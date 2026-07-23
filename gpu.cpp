#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <syslog.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm.h>
#include <drm_mode.h>

#include "config.h"
#include "gpu.h"
#include "display.h"
#include "tick.h"
#include "util.h"
#include "statistics.h"
#include "mem_alloc.h"

#include <sys/socket.h>
#include <linux/netlink.h>
#include <fcntl.h>
#include <unistd.h>

bool MarkProgramQuitting(void);

// ----------------------------------------------------------------------------
// Global variables required by gpu.h
// ----------------------------------------------------------------------------
int frameTimeHistorySize = 0;
FrameHistory frameTimeHistory[FRAME_HISTORY_MAX_SIZE] = {};

uint16_t *videoCoreFramebuffer[2] = {};
volatile int numNewGpuFrames = 0;

int displayXOffset = 0;
int displayYOffset = 0;
int gpuFrameWidth = 0;
int gpuFrameHeight = 0;
int gpuFramebufferScanlineStrideBytes = 0;
int gpuFramebufferSizeBytes = 0;

int excessPixelsLeft = 0;
int excessPixelsRight = 0;
int excessPixelsTop = 0;
int excessPixelsBottom = 0;

int eagerFastTrackToSnapshottingFramesEarlierFactor = 0;
uint64_t lastFramePollTime = 0;

pthread_t gpuPollingThread;

uint64_t frameArrivalTimes[HISTOGRAM_SIZE];
uint64_t frameArrivalTimesTail = 0;
int histogramSize = 0;

#define HISTOGRAM_MAX_SAMPLE_AGE 10000000

// ----------------------------------------------------------------------------
// DRM Private State
// ----------------------------------------------------------------------------
static int drmFd = -1;
static uint32_t drmCrtcId = 0;
static uint32_t currentFbId = 0;
static uint32_t currentHandle = 0;

static uint16_t *mappedFramebuffer = nullptr;
static size_t mappedFramebufferSize = 0;
static uint32_t mappedFbPitch = 0;

static uint32_t drmSrcWidth = 0;
static uint32_t drmSrcHeight = 0;

static int RoundUpToMultipleOf(int val, int multiple)
{
  return ((val + multiple - 1) / multiple) * multiple;
}

// ----------------------------------------------------------------------------
// Frame Content Analysis & Histogram Helpers
// ----------------------------------------------------------------------------
bool IsNewFramebuffer(uint16_t *possiblyNewFramebuffer, uint16_t *oldFramebuffer)
{
  for (uint32_t *newfb = (uint32_t *)possiblyNewFramebuffer,
                *oldfb = (uint32_t *)oldFramebuffer,
                *endfb = (uint32_t *)oldFramebuffer + gpuFramebufferSizeBytes / 4;
       oldfb < endfb;)
  {
    if (*newfb++ != *oldfb++)
      return true;
  }
  return false;
}

void AddHistogramSample(uint64_t t)
{
  frameArrivalTimes[frameArrivalTimesTail] = t;
  frameArrivalTimesTail = (frameArrivalTimesTail + 1) % HISTOGRAM_SIZE;
  if (histogramSize < HISTOGRAM_SIZE)
    ++histogramSize;

  // Expire too old entries.
  while (t - GET_HISTOGRAM(histogramSize - 1) > HISTOGRAM_MAX_SAMPLE_AGE)
    --histogramSize;
}

static int cmp(const void *e1, const void *e2)
{
  return *(uint64_t *)e1 > *(uint64_t *)e2;
}

uint64_t EstimateFrameRateInterval()
{
  if (histogramSize == 0)
    return 1000000 / TARGET_FRAME_RATE;

  uint64_t mostRecentFrame = GET_HISTOGRAM(0);
  uint64_t timeNow = tick();

#ifdef SAVE_BATTERY_BY_SLEEPING_WHEN_IDLE
  if (timeNow - mostRecentFrame > 60000000)
  {
    histogramSize = 1;
    return 500000;
  }
  if (timeNow - mostRecentFrame > 5000000)
    return lastFramePollTime + 100000;
#endif

#ifndef SAVE_BATTERY_BY_PREDICTING_FRAME_ARRIVAL_TIMES
  return 1000000 / TARGET_FRAME_RATE;
#else
  if (histogramSize < 2)
    return 100000;

  uint64_t intervals[HISTOGRAM_SIZE - 1];
  for (int i = 0; i < histogramSize - 1; ++i)
    intervals[i] = MIN(100000, GET_HISTOGRAM(i) - GET_HISTOGRAM(i + 1));
  qsort(intervals, histogramSize - 1, sizeof(uint64_t), cmp);

  int percentile = (histogramSize - 1) * 2 / 5;
  percentile = MAX(percentile - eagerFastTrackToSnapshottingFramesEarlierFactor, 0);
  uint64_t interval = intervals[percentile];

  interval = MIN(interval, GET_HISTOGRAM(0) - GET_HISTOGRAM(1));
  interval = MAX((int64_t)interval - eagerFastTrackToSnapshottingFramesEarlierFactor * 1000, (int64_t)1000000 / TARGET_FRAME_RATE);

  if (interval > 100000)
    interval = 100000;

  return MAX(interval, 1000000 / TARGET_FRAME_RATE);
#endif
}

uint64_t PredictNextFrameArrivalTime()
{
  uint64_t mostRecentFrame = histogramSize > 0 ? GET_HISTOGRAM(0) : tick();
  uint64_t timeNow = tick();

#ifdef SAVE_BATTERY_BY_SLEEPING_WHEN_IDLE
  if (timeNow - mostRecentFrame > 60000000)
  {
    histogramSize = 1;
    return lastFramePollTime + 100000;
  }
  if (timeNow - mostRecentFrame > 5000000)
    return lastFramePollTime + 100000;
#endif

  uint64_t interval = EstimateFrameRateInterval();
  uint64_t k = (timeNow - mostRecentFrame + interval - 1) / interval;
  uint64_t nextFrameArrivalTime = mostRecentFrame + k * interval;
  uint64_t timeOfPreviousMissedFrame = nextFrameArrivalTime - interval;

  if (timeNow - timeOfPreviousMissedFrame < interval / 3 && timeOfPreviousMissedFrame > mostRecentFrame)
    return timeNow;
  else
    return nextFrameArrivalTime;
}

// ----------------------------------------------------------------------------
// DRM Memory Unmapping Helper
// ----------------------------------------------------------------------------
static void UnmapCurrentDrmBuffer()
{
  if (mappedFramebuffer)
  {
    munmap(mappedFramebuffer, mappedFramebufferSize);
    mappedFramebuffer = nullptr;
    mappedFramebufferSize = 0;
    mappedFbPitch = 0;
  }
}

// ----------------------------------------------------------------------------
// Snapshot Framebuffer via DRM KMS
// ----------------------------------------------------------------------------
#include <unordered_map>

struct MappedBuffer {
  void *ptr;
  size_t size;
  uint32_t width;
  uint32_t height;
  uint32_t pitch;
};

static std::unordered_map<uint32_t, MappedBuffer> drmBufferCache;

void CleanupDrmBufferCache()
{
  for (auto &pair : drmBufferCache)
  {
    if (pair.second.ptr && pair.second.ptr != MAP_FAILED)
      munmap(pair.second.ptr, pair.second.size);
  }
  drmBufferCache.clear();
}

// ------------------------------------------------------------------------
// LUT RESCALING GLOBALS
// ------------------------------------------------------------------------
static int drmUeventFd = -1;

static uint32_t *lutX = nullptr;
static uint32_t *lutY = nullptr;
static uint32_t lutSrcWidth = 0;
static uint32_t lutSrcHeight = 0;
static bool isRescalingNeeded = false;

static uint32_t *cachedSrcRow = nullptr;
static size_t cachedSrcRowCapacityPixels = 0;

// ------------------------------------------------------------------------
// DRM NETLINK UEVENT LISTENER
// ------------------------------------------------------------------------
static void InitDrmUeventListener()
{
  drmUeventFd = socket(AF_NETLINK, SOCK_RAW, NETLINK_KOBJECT_UEVENT);
  if (drmUeventFd < 0)
  {
    printf("[DRM] Warning: Failed to create Netlink uevent socket.\n");
    return;
  }

  struct sockaddr_nl sa = {};
  sa.nl_family = AF_NETLINK;
  sa.nl_groups = 1; // Kernel multicast group for uevents

  if (bind(drmUeventFd, (struct sockaddr *)&sa, sizeof(sa)) < 0)
  {
    printf("[DRM] Warning: Failed to bind Netlink uevent socket.\n");
    close(drmUeventFd);
    drmUeventFd = -1;
    return;
  }

  // Set non-blocking mode so event checking never delays frame capture
  int flags = fcntl(drmUeventFd, F_GETFL, 0);
  fcntl(drmUeventFd, F_SETFL, flags | O_NONBLOCK);

  printf("[DRM] Kernel uevent listener initialized on fd %d.\n", drmUeventFd);
}

static bool CheckDrmUevents()
{
  if (drmUeventFd < 0)
    return false;

  char buffer[2048];
  bool eventDetected = false;

  // Drain all pending messages in socket buffer
  while (true)
  {
    ssize_t len = recv(drmUeventFd, buffer, sizeof(buffer) - 1, 0);
    if (len <= 0)
      break;

    buffer[len] = '\0';

    bool isDrm = false;
    bool isHotplugOrChange = false;

    char *ptr = buffer;
    while (ptr < buffer + len)
    {
      if (strcmp(ptr, "SUBSYSTEM=drm") == 0)
      {
        isDrm = true;
      }
      else if (strcmp(ptr, "HOTPLUG=1") == 0 || strncmp(ptr, "ACTION=change", 13) == 0)
      {
        isHotplugOrChange = true;
      }

      ptr += strlen(ptr) + 1;
    }

    if (isDrm && isHotplugOrChange)
    {
      eventDetected = true;
    }
  }

  return eventDetected;
}

// ------------------------------------------------------------------------
// DYNAMIC LUT REBUILD
// ------------------------------------------------------------------------
static void EnsureLutUpdated(uint32_t currentSrcWidth, uint32_t currentSrcHeight)
{
  // Short-circuit if current LUT mapping matches active framebuffer dimensions
  if (currentSrcWidth == lutSrcWidth && currentSrcHeight == lutSrcHeight && lutX != nullptr)
    return;

  lutSrcWidth = currentSrcWidth;
  lutSrcHeight = currentSrcHeight;

  // 1. Overscan and cropping offsets calculation
  double overscanLeft = 0.00, overscanRight = 0.00;
  double overscanTop = 0.00, overscanBottom = 0.00;

#ifdef DISPLAY_CROPPED_INSTEAD_OF_SCALING
  if (DISPLAY_DRAWABLE_WIDTH < (int)lutSrcWidth)
  {
    overscanLeft = (lutSrcWidth - DISPLAY_DRAWABLE_WIDTH) * 0.5 / lutSrcWidth;
    overscanRight = overscanLeft;
  }
  if (DISPLAY_DRAWABLE_HEIGHT < (int)lutSrcHeight)
  {
    overscanTop = (lutSrcHeight - DISPLAY_DRAWABLE_HEIGHT) * 0.5 / lutSrcHeight;
    overscanBottom = overscanTop;
  }
#endif

  uint32_t srcCropStartX = ROUND_TO_NEAREST_INT(lutSrcWidth * overscanLeft);
  uint32_t srcCropStartY = ROUND_TO_NEAREST_INT(lutSrcHeight * overscanTop);
  int relevantWidth = ROUND_TO_NEAREST_INT(lutSrcWidth * (1.0 - overscanLeft - overscanRight));
  int relevantHeight = ROUND_TO_NEAREST_INT(lutSrcHeight * (1.0 - overscanTop - overscanBottom));

  // 2. Check if spatial rescaling is required
  isRescalingNeeded = (lutSrcWidth != (uint32_t)gpuFrameWidth) ||
                      (lutSrcHeight != (uint32_t)gpuFrameHeight) ||
                      (srcCropStartX != 0) || (srcCropStartY != 0);

  // 3. Reallocate LUT arrays
  if (lutX) { free(lutX); lutX = nullptr; }
  if (lutY) { free(lutY); lutY = nullptr; }

  if (isRescalingNeeded)
  {
    lutX = (uint32_t *)Malloc(gpuFrameWidth * sizeof(uint32_t), "gpu.cpp lutX");
    lutY = (uint32_t *)Malloc(gpuFrameHeight * sizeof(uint32_t), "gpu.cpp lutY");

    for (int x = 0; x < gpuFrameWidth; ++x)
    {
      double normX = ((double)x + 0.5) / gpuFrameWidth;
      uint32_t srcX = srcCropStartX + (uint32_t)(normX * relevantWidth);
      if (srcX >= lutSrcWidth) srcX = lutSrcWidth - 1;
      lutX[x] = srcX;
    }

    for (int y = 0; y < gpuFrameHeight; ++y)
    {
      double normY = ((double)y + 0.5) / gpuFrameHeight;
      uint32_t srcY = srcCropStartY + (uint32_t)(normY * relevantHeight);
      if (srcY >= lutSrcHeight) srcY = lutSrcHeight - 1;
      lutY[y] = srcY;
    }
  }

  // 4. Dynamically size scanline cache buffer to handle current source width
  if (cachedSrcRowCapacityPixels < lutSrcWidth)
  {
    if (cachedSrcRow) free(cachedSrcRow);
    cachedSrcRowCapacityPixels = lutSrcWidth + 128; // Padding safety buffer
    cachedSrcRow = (uint32_t *)Malloc(cachedSrcRowCapacityPixels * sizeof(uint32_t), "gpu.cpp cachedSrcRow");
  }

  printf("[GPU] Rescale LUT updated: %dx%d -> %dx%d (%s)\n",
         lutSrcWidth, lutSrcHeight, gpuFrameWidth, gpuFrameHeight,
         isRescalingNeeded ? "RESCALING ACTIVE" : "1:1 DIRECT FAST-PATH");
}

// ------------------------------------------------------------------------
// SNAPSHOT FRAMEBUFFER
// ------------------------------------------------------------------------
bool SnapshotFramebuffer(uint16_t *destination)
{
  if (drmFd < 0 || drmCrtcId == 0)
    return false;

  static uint32_t lastProcessedFbId = 0;

  // ------------------------------------------------------------------------
  // 0. KERNEL EVENT DRIVEN CACHE & LUT INVALIDATION
  // ------------------------------------------------------------------------
  if (CheckDrmUevents())
  {
    printf("[DRM] Kernel reported mode change / Sway reload! Flushing buffer cache.\n");
    CleanupDrmBufferCache();
    lutSrcWidth = 0;
    lutSrcHeight = 0;
    lastProcessedFbId = 0;
  }

  // ------------------------------------------------------------------------
  // 1. QUERY CRTC AND ACTIVE DISPLAY MODE
  // ------------------------------------------------------------------------
  drmModeCrtc *crtc = drmModeGetCrtc(drmFd, drmCrtcId);
  if (!crtc)
    return false;

  uint32_t fbId = crtc->buffer_id;
  uint32_t activeWidth = crtc->mode.hdisplay;
  uint32_t activeHeight = crtc->mode.vdisplay;
  bool hasValidMode = crtc->mode_valid && (activeWidth > 0) && (activeHeight > 0);
  drmModeFreeCrtc(crtc);

  if (fbId == 0)
    return false;

  // ------------------------------------------------------------------------
  // 2. FETCH OR MAP FRAMEBUFFER WITH ACTIVE-MODE FILTERING
  // ------------------------------------------------------------------------
  void *mappedPtr = nullptr;
  auto it = drmBufferCache.find(fbId);

  if (it != drmBufferCache.end())
  {
    // Validate cached entry against active mode resolution
    if (hasValidMode && (it->second.width != activeWidth || it->second.height != activeHeight))
    {
      // Cached buffer is from old display mode! Flush cache.
      CleanupDrmBufferCache();
      mappedPtr = nullptr;
    }
    else
    {
      mappedPtr = it->second.ptr;
      drmSrcWidth = it->second.width;
      drmSrcHeight = it->second.height;
      mappedFbPitch = it->second.pitch;
    }
  }

  if (!mappedPtr)
  {
    if (drmBufferCache.size() > 16)
    {
      CleanupDrmBufferCache();
    }

    drmModeFB2 *fb = drmModeGetFB2(drmFd, fbId);
    if (!fb) return false;

    // HARDWARE FILTER: Reject transient framebuffers during mode transitions
    if (hasValidMode && (fb->width != activeWidth || fb->height != activeHeight))
    {
      drmModeFreeFB2(fb);
      return false; // Skip frame until Sway finishes switching resolutions
    }

    // Purge stale buffer cache if resolution actually changed
    if (fb->width != lutSrcWidth || fb->height != lutSrcHeight)
    {
      printf("[DRM] Resolution shift (%ux%u -> %ux%u). Purging buffer cache.\n",
             lutSrcWidth, lutSrcHeight, fb->width, fb->height);
      CleanupDrmBufferCache();
      lastProcessedFbId = 0;
    }

    drmSrcWidth = fb->width;
    drmSrcHeight = fb->height;
    mappedFbPitch = fb->pitches[0];
    size_t fbSize = fb->height * fb->pitches[0];

    drm_mode_map_dumb map = {};
    map.handle = fb->handles[0];

    if (ioctl(drmFd, DRM_IOCTL_MODE_MAP_DUMB, &map) >= 0)
    {
      void *ptr = mmap(nullptr, fbSize, PROT_READ, MAP_SHARED, drmFd, map.offset);
      if (ptr != MAP_FAILED)
      {
        drmBufferCache[fbId] = {ptr, fbSize, drmSrcWidth, drmSrcHeight, mappedFbPitch};
        mappedPtr = ptr;
      }
    }
    drmModeFreeFB2(fb);
  }

  if (!mappedPtr || drmSrcWidth == 0 || drmSrcHeight == 0)
    return false;

  // Ensure LUT state matches current framebuffer geometry
  EnsureLutUpdated(drmSrcWidth, drmSrcHeight);

  bool is32bpp = (mappedFbPitch >= drmSrcWidth * 4);

  // Short-circuit if framebuffer hasn't flipped
  if (is32bpp && fbId == lastProcessedFbId)
  {
    return false;
  }

  const uint32_t dstPitchBytes = (gpuFramebufferScanlineStrideBytes > 0) 
                                 ? gpuFramebufferScanlineStrideBytes 
                                 : (gpuFrameWidth * sizeof(uint16_t));
  const uint32_t dstStride16 = dstPitchBytes / sizeof(uint16_t);

  const auto *srcBytes = reinterpret_cast<const uint8_t *>(mappedPtr);

  if (is32bpp)
  {
    lastProcessedFbId = fbId;

    if (!isRescalingNeeded)
    {
      // --------------------------------------------------------------------
      // PATH A: 1:1 DIRECT FAST-PATH (No Scaling)
      // --------------------------------------------------------------------
      const uint32_t rowCopyBytes = gpuFrameWidth * 4;

      for (uint32_t y = 0; y < gpuFrameHeight; ++y)
      {
        const uint8_t *uncachedSrcRow = srcBytes + (y * mappedFbPitch);
        uint16_t *dstRow = destination + (y * dstStride16);

        memcpy(cachedSrcRow, uncachedSrcRow, rowCopyBytes);

        for (uint32_t x = 0; x < gpuFrameWidth; ++x)
        {
          uint32_t pixel = cachedSrcRow[x];
          uint8_t r = (pixel >> 16) & 0xFF;
          uint8_t g = (pixel >> 8)  & 0xFF;
          uint8_t b = pixel & 0xFF;

          dstRow[x] = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
        }
      }
    }
    else
    {
      // --------------------------------------------------------------------
      // PATH B: OPTIMIZED LUT RESCALED PATH
      // --------------------------------------------------------------------
      const uint32_t rowCopyBytes = drmSrcWidth * 4;

      for (uint32_t y = 0; y < gpuFrameHeight; ++y)
      {
        uint32_t srcY = lutY[y];
        const uint8_t *uncachedSrcRow = srcBytes + (srcY * mappedFbPitch);
        uint16_t *dstRow = destination + (y * dstStride16);

        // Copy entire source scanline to CPU cache to avoid uncached read stalls
        memcpy(cachedSrcRow, uncachedSrcRow, rowCopyBytes);

        for (uint32_t x = 0; x < gpuFrameWidth; ++x)
        {
          uint32_t pixel = cachedSrcRow[lutX[x]];
          uint8_t r = (pixel >> 16) & 0xFF;
          uint8_t g = (pixel >> 8)  & 0xFF;
          uint8_t b = pixel & 0xFF;

          dstRow[x] = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
        }
      }
    }
  }
  else
  {
    // 16bpp direct format fallback
    if (mappedFbPitch == dstPitchBytes)
    {
      memcpy(destination, srcBytes, gpuFrameHeight * dstPitchBytes);
    }
    else
    {
      const uint32_t copyBytes = gpuFrameWidth * sizeof(uint16_t);
      for (uint32_t y = 0; y < gpuFrameHeight; ++y)
      {
        const uint16_t *srcRow = reinterpret_cast<const uint16_t *>(srcBytes + y * mappedFbPitch);
        uint16_t *dstRow = reinterpret_cast<uint16_t *>(reinterpret_cast<uint8_t *>(destination) + y * dstPitchBytes);
        memcpy(dstRow, srcRow, copyBytes);
      }
    }
  }

  return true;
}

// ----------------------------------------------------------------------------
// Simplified DRM GPU Polling Thread (No Battery-Prediction Drift)
// ----------------------------------------------------------------------------
extern volatile bool programRunning;

void *gpu_polling_thread(void *)
{
  const uint64_t TARGET_FRAME_TIME_US = 1000000 / 60; // Hard 60 Hz target
  uint64_t lastFrameTime = tick();

  while (programRunning)
  {
    uint64_t now = tick();
    
    // Pace to 60 FPS cleanly
    if (now - lastFrameTime < TARGET_FRAME_TIME_US)
    {
      uint64_t sleepUs = TARGET_FRAME_TIME_US - (now - lastFrameTime);
      if (sleepUs > 500)
        usleep(sleepUs - 200); // Sleep with 200us margin to prevent oversleeping
    }

    uint64_t t0 = tick();

    bool gotNewFramebuffer = SnapshotFramebuffer(videoCoreFramebuffer[0]);
    gotNewFramebuffer = gotNewFramebuffer && IsNewFramebuffer(videoCoreFramebuffer[0], videoCoreFramebuffer[1]);

    if (!gotNewFramebuffer)
    {
      // Idle path: Tiny yield to avoid 100% spin without dropping frame response
      usleep(1000); 
      continue;
    }

    lastFrameTime = t0;

    // Send frame downstream
    memcpy(videoCoreFramebuffer[1], videoCoreFramebuffer[0], gpuFramebufferSizeBytes);
    __atomic_fetch_add(&numNewGpuFrames, 1, __ATOMIC_SEQ_CST);
    syscall(SYS_futex, &numNewGpuFrames, FUTEX_WAKE, 1, 0, 0, 0);
  }

  pthread_exit(0);
}

// ------------------------------------------------------------------------
// INIT GPU
// ------------------------------------------------------------------------
void InitGPU()
{
  // 1 & 2. Auto-detect DRM device node with an active CRTC
  drmModeCrtc *crtc = nullptr;
  drmFd = -1;

  for (int cardIdx = 0; cardIdx < 8; cardIdx++)
  {
    char cardPath[32];
    snprintf(cardPath, sizeof(cardPath), "/dev/dri/card%d", cardIdx);

    int fd = open(cardPath, O_RDWR | O_CLOEXEC);
    if (fd < 0)
      continue;

    drmModeRes *res = drmModeGetResources(fd);
    if (!res)
    {
      close(fd);
      continue;
    }

    // Search for an active CRTC with a bound framebuffer
    for (int i = 0; i < res->count_crtcs; i++)
    {
      drmModeCrtc *testCrtc = drmModeGetCrtc(fd, res->crtcs[i]);
      if (!testCrtc)
        continue;

      if (testCrtc->buffer_id)
      {
        crtc = testCrtc;
        drmCrtcId = testCrtc->crtc_id;
        drmFd = fd;
        break;
      }

      drmModeFreeCrtc(testCrtc);
    }

    drmModeFreeResources(res);

    // Found working display card
    if (crtc)
    {
      drmDropMaster(drmFd);
      printf("Opened DRM device %s successfully.\n", cardPath);
      printf("Found active DRM CRTC %u\n", drmCrtcId);
      break;
    }

    close(fd);
  }

  if (drmFd < 0 || !crtc)
    FATAL_ERROR("No active DRM display device or bound CRTC framebuffer found across card0-card7");

  // 3. Obtain geometry of primary framebuffer attached to CRTC
  drmModeFB2 *fb = drmModeGetFB2(drmFd, crtc->buffer_id);
  if (!fb)
    FATAL_ERROR("drmModeGetFB2 failed for CRTC buffer ID %u", crtc->buffer_id);

  uint32_t srcWidth = fb->width;
  uint32_t srcHeight = fb->height;
  drmSrcWidth = srcWidth;
  drmSrcHeight = srcHeight;

  drmModeFreeFB2(fb);
  drmModeFreeCrtc(crtc);

  // 4. Overscan and aspect ratio scaling calculation
  double overscanLeft = 0.00;
  double overscanRight = 0.00;
  double overscanTop = 0.00;
  double overscanBottom = 0.00;

#ifdef DISPLAY_CROPPED_INSTEAD_OF_SCALING
  if (DISPLAY_DRAWABLE_WIDTH < (int)srcWidth)
  {
    overscanLeft = (srcWidth - DISPLAY_DRAWABLE_WIDTH) * 0.5 / srcWidth;
    overscanRight = overscanLeft;
  }
  if (DISPLAY_DRAWABLE_HEIGHT < (int)srcHeight)
  {
    overscanTop = (srcHeight - DISPLAY_DRAWABLE_HEIGHT) * 0.5 / srcHeight;
    overscanBottom = overscanTop;
  }
#endif

  overscanLeft = (double)ROUND_TO_FLOOR_INT(srcWidth * overscanLeft) / srcWidth;
  overscanRight = (double)ROUND_TO_CEIL_INT(srcWidth * overscanRight) / srcWidth;
  overscanTop = (double)ROUND_TO_FLOOR_INT(srcHeight * overscanTop) / srcHeight;
  overscanBottom = (double)ROUND_TO_CEIL_INT(srcHeight * overscanBottom) / srcHeight;

  int relevantDisplayWidth = ROUND_TO_NEAREST_INT(srcWidth * (1.0 - overscanLeft - overscanRight));
  int relevantDisplayHeight = ROUND_TO_NEAREST_INT(srcHeight * (1.0 - overscanTop - overscanBottom));

  double scalingFactorWidth = (double)DISPLAY_DRAWABLE_WIDTH / relevantDisplayWidth;
  double scalingFactorHeight = (double)DISPLAY_DRAWABLE_HEIGHT / relevantDisplayHeight;

#ifndef DISPLAY_BREAK_ASPECT_RATIO_WHEN_SCALING
  scalingFactorWidth = scalingFactorHeight = MIN(scalingFactorWidth, scalingFactorHeight);
#endif

  int scaledWidth = ROUND_TO_NEAREST_INT(relevantDisplayWidth * scalingFactorWidth);
  int scaledHeight = ROUND_TO_NEAREST_INT(relevantDisplayHeight * scalingFactorHeight);

  displayXOffset = DISPLAY_COVERED_LEFT_SIDE + (DISPLAY_DRAWABLE_WIDTH - scaledWidth) / 2;
  displayYOffset = DISPLAY_COVERED_TOP_SIDE + (DISPLAY_DRAWABLE_HEIGHT - scaledHeight) / 2;

  excessPixelsLeft = ROUND_TO_NEAREST_INT(srcWidth * overscanLeft * scalingFactorWidth);
  excessPixelsRight = ROUND_TO_NEAREST_INT(srcWidth * overscanRight * scalingFactorWidth);
  excessPixelsTop = ROUND_TO_NEAREST_INT(srcHeight * overscanTop * scalingFactorHeight);
  excessPixelsBottom = ROUND_TO_NEAREST_INT(srcHeight * overscanBottom * scalingFactorHeight);

  gpuFrameWidth = scaledWidth;
  gpuFrameHeight = scaledHeight;
  gpuFramebufferScanlineStrideBytes = RoundUpToMultipleOf((gpuFrameWidth + excessPixelsLeft + excessPixelsRight) * sizeof(uint16_t), 32);
  gpuFramebufferSizeBytes = gpuFramebufferScanlineStrideBytes * (gpuFrameHeight + excessPixelsTop + excessPixelsBottom);

  syslog(LOG_INFO, "Source DRM display is %dx%d. Output SPI display is %dx%d (drawable %dx%d). Scaled size: %dx%d, offsets: (%d, %d)",
         srcWidth, srcHeight, DISPLAY_WIDTH, DISPLAY_HEIGHT, DISPLAY_DRAWABLE_WIDTH, DISPLAY_DRAWABLE_HEIGHT,
         scaledWidth, scaledHeight, displayXOffset, displayYOffset);

  printf("Source DRM display is %dx%d. Output SPI display is %dx%d (drawable %dx%d). Scaled size: %dx%d, offsets: (%d, %d)\n",
         srcWidth, srcHeight, DISPLAY_WIDTH, DISPLAY_HEIGHT, DISPLAY_DRAWABLE_WIDTH, DISPLAY_DRAWABLE_HEIGHT,
         scaledWidth, scaledHeight, displayXOffset, displayYOffset);

  // ------------------------------------------------------------------------
  // 5. INITIALIZE UEVENT LISTENER & BUILD INITIAL RESCALE LUT
  // ------------------------------------------------------------------------
  InitDrmUeventListener();
  EnsureLutUpdated(srcWidth, srcHeight);

  // 6. Allocate double framebuffers
  videoCoreFramebuffer[0] = (uint16_t *)Malloc(gpuFramebufferSizeBytes, "gpu.cpp framebuffer0");
  videoCoreFramebuffer[1] = (uint16_t *)Malloc(gpuFramebufferSizeBytes, "gpu.cpp framebuffer1");
  memset(videoCoreFramebuffer[0], 0, gpuFramebufferSizeBytes);
  memset(videoCoreFramebuffer[1], 0, gpuFramebufferSizeBytes);

  // 7. Warm up frame-rate histogram
  uint64_t now = tick();
  for (int i = 0; i < HISTOGRAM_SIZE; ++i)
    AddHistogramSample(now - 1000000ULL * (HISTOGRAM_SIZE - i) / TARGET_FRAME_RATE);

  // 8. Spawn GPU polling thread
  int rc = pthread_create(&gpuPollingThread, NULL, gpu_polling_thread, NULL);
  if (rc != 0)
    FATAL_ERROR("Failed to create GPU polling thread!");
}

// ----------------------------------------------------------------------------
// Deinitialization
// ----------------------------------------------------------------------------
void DeinitGPU()
{
  if (gpuPollingThread)
  {
    pthread_join(gpuPollingThread, NULL);
    gpuPollingThread = (pthread_t)0;
  }

  UnmapCurrentDrmBuffer();

  currentFbId = 0;
  currentHandle = 0;
  drmCrtcId = 0;

  if (drmFd >= 0)
  {
    close(drmFd);
    drmFd = -1;
  }

  gpuFrameWidth = 0;
  gpuFrameHeight = 0;
  gpuFramebufferScanlineStrideBytes = 0;
  gpuFramebufferSizeBytes = 0;
}
