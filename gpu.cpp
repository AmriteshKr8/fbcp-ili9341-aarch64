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
bool SnapshotFramebuffer(uint16_t *destination)
{
  lastFramePollTime = tick();

  if (drmFd < 0 || drmCrtcId == 0)
    return false;

  drmModeCrtc *crtc = drmModeGetCrtc(drmFd, drmCrtcId);
  if (!crtc)
    return false;

  uint32_t fbId = crtc->buffer_id;
  drmModeFreeCrtc(crtc);

  if (fbId == 0)
    return false;

  if (fbId != currentFbId || !mappedFramebuffer)
  {
    UnmapCurrentDrmBuffer();

    drmModeFB2 *fb = drmModeGetFB2(drmFd, fbId);
    if (!fb)
    {
      perror("drmModeGetFB2");
      return false;
    }

    currentFbId = fbId;
    currentHandle = fb->handles[0];

    drmSrcWidth = fb->width;
    drmSrcHeight = fb->height;
    mappedFbPitch = fb->pitches[0];
    mappedFramebufferSize = fb->height * fb->pitches[0];

    drm_mode_map_dumb map = {};
    map.handle = currentHandle;

    if (ioctl(drmFd, DRM_IOCTL_MODE_MAP_DUMB, &map) < 0)
    {
      perror("DRM_IOCTL_MODE_MAP_DUMB");
      drmModeFreeFB2(fb);
      return false;
    }

    void *ptr = mmap(nullptr, mappedFramebufferSize, PROT_READ, MAP_SHARED, drmFd, map.offset);
    drmModeFreeFB2(fb);

    if (ptr == MAP_FAILED)
    {
      perror("mmap");
      currentFbId = 0;
      currentHandle = 0;
      return false;
    }

    mappedFramebuffer = static_cast<uint16_t *>(ptr);
  }

  if (!mappedFramebuffer || drmSrcWidth == 0 || drmSrcHeight == 0)
    return false;

  // Calculate destination pitch & stride
  const uint32_t dstPitchBytes = (gpuFramebufferScanlineStrideBytes > 0) 
                                 ? gpuFramebufferScanlineStrideBytes 
                                 : (gpuFrameWidth * sizeof(uint16_t));
  const uint32_t dstStride16 = dstPitchBytes / sizeof(uint16_t);

  const auto *srcBytes = reinterpret_cast<const uint8_t *>(mappedFramebuffer);
  auto *dstBytes = reinterpret_cast<uint8_t *>(destination);

  // Detect 32-bit (XRGB8888/ARGB8888) vs 16-bit (RGB565) source
  bool is32bpp = (mappedFbPitch >= drmSrcWidth * 4);

  if (is32bpp)
  {
    // ------------------------------------------------------------------------
    // 32-bit (Sway / Wayland) -> 16-bit RGB565 Downsampler
    // Reads exact byte offsets (B, G, R) and safely ignores Alpha/X byte (0xFF)
    // ------------------------------------------------------------------------
    for (uint32_t y = 0; y < gpuFrameHeight; ++y)
    {
      uint32_t srcY = (y * drmSrcHeight) / gpuFrameHeight;
      const uint8_t *srcRow = srcBytes + (srcY * mappedFbPitch);
      uint16_t *dstRow = destination + (y * dstStride16);

      for (uint32_t x = 0; x < gpuFrameWidth; ++x)
      {
        uint32_t srcX = (x * drmSrcWidth) / gpuFrameWidth;
        const uint8_t *px = srcRow + (srcX * 4);

        // Little-endian memory layout for DRM_FORMAT_XRGB8888:
        uint8_t b = px[0];
        uint8_t g = px[1];
        uint8_t r = px[2];
        // px[3] is Alpha/Padding (0xFF) — ignored!

        // Pack into 16-bit RGB565
        dstRow[x] = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
      }
    }
  }
  else
  {
    // ------------------------------------------------------------------------
    // 16-bit (TTY / fbcon) Path
    // ------------------------------------------------------------------------
    if (drmSrcWidth == gpuFrameWidth && drmSrcHeight == gpuFrameHeight)
    {
      // Fast path: 1:1 scanline copy
      const uint32_t copyBytes = gpuFrameWidth * sizeof(uint16_t);
      for (uint32_t y = 0; y < gpuFrameHeight; ++y)
      {
        const uint16_t *srcRow = reinterpret_cast<const uint16_t *>(srcBytes + y * mappedFbPitch);
        uint16_t *dstRow = reinterpret_cast<uint16_t *>(dstBytes + y * dstPitchBytes);
        memcpy(dstRow, srcRow, copyBytes);
      }
    }
    else
    {
      // Resampled path
      for (uint32_t y = 0; y < gpuFrameHeight; ++y)
      {
        uint32_t srcY = (y * drmSrcHeight) / gpuFrameHeight;
        const uint16_t *srcRow = reinterpret_cast<const uint16_t *>(srcBytes + srcY * mappedFbPitch);
        uint16_t *dstRow = reinterpret_cast<uint16_t *>(dstBytes + y * dstPitchBytes);

        for (uint32_t x = 0; x < gpuFrameWidth; ++x)
        {
          uint32_t srcX = (x * drmSrcWidth) / gpuFrameWidth;
          dstRow[x] = srcRow[srcX];
        }
      }
    }
  }

  return true;
}

// ----------------------------------------------------------------------------
// GPU Polling Thread Routine
// ----------------------------------------------------------------------------
extern volatile bool programRunning;

void *gpu_polling_thread(void *)
{
  uint64_t lastNewFrameReceivedTime = tick();

  while (programRunning)
  {
#ifdef SAVE_BATTERY_BY_SLEEPING_UNTIL_TARGET_FRAME
    const int64_t earlyFramePrediction = 500;
    uint64_t earliestNextFrameArrivaltime = lastNewFrameReceivedTime + 1000000 / TARGET_FRAME_RATE - earlyFramePrediction;
    uint64_t now = tick();
    if (earliestNextFrameArrivaltime > now)
      usleep(earliestNextFrameArrivaltime - now);
#endif

#if defined(SAVE_BATTERY_BY_PREDICTING_FRAME_ARRIVAL_TIMES) || defined(SAVE_BATTERY_BY_SLEEPING_WHEN_IDLE)
    uint64_t nextFrameArrivalTime = PredictNextFrameArrivalTime();
    int64_t timeToSleep = nextFrameArrivalTime - tick();
    const int64_t minimumSleepTime = 150;
    if (timeToSleep > minimumSleepTime)
      usleep(timeToSleep - minimumSleepTime);
#endif

    uint64_t t0 = tick();

    bool gotNewFramebuffer = SnapshotFramebuffer(videoCoreFramebuffer[0]);
    gotNewFramebuffer = gotNewFramebuffer && IsNewFramebuffer(videoCoreFramebuffer[0], videoCoreFramebuffer[1]);

    if (gotNewFramebuffer)
    {
      lastNewFrameReceivedTime = t0;
      AddHistogramSample(lastNewFrameReceivedTime);
    }

    uint64_t t1 = tick();

    if (!gotNewFramebuffer)
    {
#ifdef STATISTICS
      __atomic_fetch_add(&timeWastedPollingGPU, t1 - t0, __ATOMIC_RELAXED);
#endif
      eagerFastTrackToSnapshottingFramesEarlierFactor /= 2;
      continue;
    }
    else
    {
      ++eagerFastTrackToSnapshottingFramesEarlierFactor;
      memcpy(videoCoreFramebuffer[1], videoCoreFramebuffer[0], gpuFramebufferSizeBytes);
      __atomic_fetch_add(&numNewGpuFrames, 1, __ATOMIC_SEQ_CST);
      syscall(SYS_futex, &numNewGpuFrames, FUTEX_WAKE, 1, 0, 0, 0);
    }
  }

  pthread_exit(0);
}

// ----------------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------------
void InitGPU()
{
  // 1. Open DRM device node
  drmFd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
  if (drmFd < 0)
    FATAL_ERROR("Failed to open /dev/dri/card0");
  drmDropMaster(drmFd);
  printf("Opened DRM device /dev/dri/card0 successfully.\n");

  // 2. Discover active CRTC
  drmModeRes *res = drmModeGetResources(drmFd);
  if (!res)
    FATAL_ERROR("drmModeGetResources failed");

  drmModeCrtc *crtc = nullptr;
  for (int i = 0; i < res->count_crtcs; i++)
  {
    drmModeCrtc *testCrtc = drmModeGetCrtc(drmFd, res->crtcs[i]);
    if (!testCrtc)
      continue;

    if (testCrtc->buffer_id)
    {
      crtc = testCrtc;
      drmCrtcId = testCrtc->crtc_id;
      break;
    }

    drmModeFreeCrtc(testCrtc);
  }

  drmModeFreeResources(res);

  if (!crtc)
    FATAL_ERROR("No active CRTC with bound framebuffer found on DRM card0");

  printf("Found active DRM CRTC %u\n", drmCrtcId);

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

  // 5. Allocate double framebuffers
  videoCoreFramebuffer[0] = (uint16_t *)Malloc(gpuFramebufferSizeBytes, "gpu.cpp framebuffer0");
  videoCoreFramebuffer[1] = (uint16_t *)Malloc(gpuFramebufferSizeBytes, "gpu.cpp framebuffer1");
  memset(videoCoreFramebuffer[0], 0, gpuFramebufferSizeBytes);
  memset(videoCoreFramebuffer[1], 0, gpuFramebufferSizeBytes);

  // 6. Warm up frame-rate histogram
  uint64_t now = tick();
  for (int i = 0; i < HISTOGRAM_SIZE; ++i)
    AddHistogramSample(now - 1000000ULL * (HISTOGRAM_SIZE - i) / TARGET_FRAME_RATE);

  // 7. Spawn GPU polling thread
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