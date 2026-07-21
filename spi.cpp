#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdio.h>    // printf, stderr
#include <syslog.h>   // syslog
#include <fcntl.h>    // open, O_RDWR, O_SYNC
#include <sys/mman.h> // mmap, munmap
#include <pthread.h>  // pthread_create
#include <unistd.h>   // close, syscall
#include <sys/syscall.h>

#include "config.h"
#include "spi.h"
#include "util.h"
#include "dma.h"
#include "mailbox.h"
#include "mem_alloc.h"

// Hardware peripheral address resolution without legacy bcm_host dependency
static uintptr_t get_bcm_peripheral_address() {
  uintptr_t address = 0;
  FILE *fp = fopen("/proc/device-tree/soc/ranges", "rb");
  if (fp) {
    unsigned char buf[12];
    if (fread(buf, 1, 12, fp) == 12) {
      address = ((uintptr_t)buf[4] << 24) | ((uintptr_t)buf[5] << 16) |
                ((uintptr_t)buf[6] << 8)  | ((uintptr_t)buf[7]);
      if (address == 0) {
        address = ((uintptr_t)buf[8] << 24) | ((uintptr_t)buf[9] << 16) |
                  ((uintptr_t)buf[10] << 8) | ((uintptr_t)buf[11]);
      }
    }
    fclose(fp);
  }
  // Fallback default (0xFE000000 for BCM2711 / Pi 4, 0x3F000000 for Pi 3)
  if (!address) address = 0xFE000000;
  return address;
}

static uint32_t get_bcm_peripheral_size() {
  return 0x01000000; // Standard 16MB peripheral address window
}

static uintptr_t get_bcm_sdram_address() {
  return 0xC0000000; // VC bus alias address
}

// Uncomment this to print out all bytes sent to the SPI bus
// #define DEBUG_SPI_BUS_WRITES

#ifdef DEBUG_SPI_BUS_WRITES
#define DEBUG_PRINT_WRITTEN_BYTE(byte) do { \
  printf("%02X", byte); \
  if ((writeCounter & 3) == 0) printf("\n"); \
  } while(0)
#else
#define DEBUG_PRINT_WRITTEN_BYTE(byte) ((void)0)
#endif

#ifdef CHIP_SELECT_LINE_NEEDS_REFRESHING_EACH_32BITS_WRITTEN
void ChipSelectHigh();
#define TOGGLE_CHIP_SELECT_LINE() if ((++writeCounter & 3) == 0) { ChipSelectHigh(); }
#else
#define TOGGLE_CHIP_SELECT_LINE() ((void)0)
#endif

static uint32_t writeCounter = 0;

#define WRITE_FIFO(word) do { \
  uint8_t w = (word); \
  spi->fifo = w; \
  TOGGLE_CHIP_SELECT_LINE(); \
  DEBUG_PRINT_WRITTEN_BYTE(w); \
  } while(0)

int mem_fd = -1;
volatile void *bcm2835 = 0;
volatile GPIORegisterFile *gpio = 0;
volatile SPIRegisterFile *spi = 0;

volatile uint64_t *systemTimerRegister = 0;

void DumpSPICS(uint32_t reg)
{
  PRINT_FLAG(BCM2835_SPI0_CS_CS);
  PRINT_FLAG(BCM2835_SPI0_CS_CPHA);
  PRINT_FLAG(BCM2835_SPI0_CS_CPOL);
  PRINT_FLAG(BCM2835_SPI0_CS_CLEAR_TX);
  PRINT_FLAG(BCM2835_SPI0_CS_CLEAR_RX);
  PRINT_FLAG(BCM2835_SPI0_CS_TA);
  PRINT_FLAG(BCM2835_SPI0_CS_DMAEN);
  PRINT_FLAG(BCM2835_SPI0_CS_INTD);
  PRINT_FLAG(BCM2835_SPI0_CS_INTR);
  PRINT_FLAG(BCM2835_SPI0_CS_ADCS);
  PRINT_FLAG(BCM2835_SPI0_CS_DONE);
  PRINT_FLAG(BCM2835_SPI0_CS_RXD);
  PRINT_FLAG(BCM2835_SPI0_CS_TXD);
  PRINT_FLAG(BCM2835_SPI0_CS_RXR);
  PRINT_FLAG(BCM2835_SPI0_CS_RXF);
  printf("SPI0 DLEN: %u\n", spi->dlen);
  printf("SPI0 CE0 register: %d\n", GET_GPIO(GPIO_SPI0_CE0) ? 1 : 0);
}

#ifdef RUN_WITH_REALTIME_THREAD_PRIORITY

#include <pthread.h>
#include <sched.h>

void SetRealtimeThreadPriority()
{
  sched_param params;
  params.sched_priority = sched_get_priority_max(SCHED_FIFO);

  int failed = pthread_setschedparam(pthread_self(), SCHED_FIFO, &params);
  if (failed) FATAL_ERROR("pthread_setschedparam() failed!");

  int policy = 0;
  failed = pthread_getschedparam(pthread_self(), &policy, &params);
  if (failed) FATAL_ERROR("pthread_getschedparam() failed!");

  if (policy != SCHED_FIFO) FATAL_ERROR("Failed to set realtime thread policy!");
  printf("Set fbcp-ili9341 thread scheduling priority to maximum (%d)\n", sched_get_priority_max(SCHED_FIFO));
}

#endif

#define UNLOCK_FAST_8_CLOCKS_SPI() (spi->dlen = 2)

#ifdef ALL_TASKS_SHOULD_DMA
bool previousTaskWasSPI = true;
#endif

#ifdef SPI_3WIRE_PROTOCOL

uint32_t NumBytesNeededFor32BitSPITask(uint32_t byteSizeFor8BitTask)
{
  return byteSizeFor8BitTask * 2 + 4;
}

uint32_t NumBytesNeededFor9BitSPITask(uint32_t byteSizeFor8BitTask)
{
  uint32_t numOutBits = (byteSizeFor8BitTask + 1) * 9;
  numOutBits = ((numOutBits + 71) / 72) * 72;
  uint32_t numOutBytes = numOutBits >> 3;
  return numOutBytes;
}

void Interleave8BitSPITaskTo9Bit(SPITask *task)
{
  const uint32_t size8BitTask = task->size - task->sizeExpandedTaskWithPadding;
  uint8_t *dst = task->data + size8BitTask;

  memset(dst + task->sizeExpandedTaskWithPadding - 9 - SPI_9BIT_TASK_PADDING_BYTES, 0, 9 + SPI_9BIT_TASK_PADDING_BYTES);

  dst[0] = task->cmd >> 1;
  dst[1] = task->cmd << 7;
  int dstByte = 1;
  int dstBitsUsed = 1;

  int src = 0;

  if (size8BitTask >= 7)
  {
    dst[1] |= 0x40 |                        (task->data[0] >> 2);
    dst[2]  = 0x20 | (task->data[0] << 6) | (task->data[1] >> 3);
    dst[3]  = 0x10 | (task->data[1] << 5) | (task->data[2] >> 4);
    dst[4]  = 0x08 | (task->data[2] << 4) | (task->data[3] >> 5);
    dst[5]  = 0x04 | (task->data[3] << 3) | (task->data[4] >> 6);
    dst[6]  = 0x02 | (task->data[4] << 2) | (task->data[5] >> 7);
    dst[7]  = 0x01 | (task->data[5] << 1);
    dst[8]  =        (task->data[6]     );
    dstByte = 9;
    dstBitsUsed = 0;
    src = 7;

    while(src <= size8BitTask - 8)
    {
      uint8_t *d = dst + dstByte;
      dstByte += 9;
      const uint8_t *s = task->data + src;
      src += 8;

      d[0] = 0x80 |               (s[0] >> 1);
      d[1] = 0x40 | (s[0] << 7) | (s[1] >> 2);
      d[2] = 0x20 | (s[1] << 6) | (s[2] >> 3);
      d[3] = 0x10 | (s[2] << 5) | (s[3] >> 4);
      d[4] = 0x08 | (s[3] << 4) | (s[4] >> 5);
      d[5] = 0x04 | (s[4] << 3) | (s[5] >> 6);
      d[6] = 0x02 | (s[5] << 2) | (s[6] >> 7);
      d[7] = 0x01 | (s[6] << 1);
      d[8] = (s[7]     );
    }

    dst[dstByte] = 0;
  }

  while(src < size8BitTask)
  {
    uint8_t data = task->data[src++];

    dst[dstByte] |= 1 << (7 - dstBitsUsed);
    ++dstBitsUsed;
    if (dstBitsUsed == 8)
    {
      ++dstByte;
      dstBitsUsed = 0;
      dst[dstByte++] = data;
      dst[dstByte] = 0;
    }
    else
    {
      dst[dstByte++] |= data >> dstBitsUsed;
      dst[dstByte] = data << (8 - dstBitsUsed);
    }
  }
}

void Interleave16BitSPITaskTo32Bit(SPITask *task)
{
  const uint32_t size8BitTask = task->size - task->sizeExpandedTaskWithPadding;

  uint32_t *dst = (uint32_t *)(task->data + size8BitTask);
  *dst++ = task->cmd;

  const uint32_t taskSizeU16 = size8BitTask >> 1;
  uint16_t *src = (uint16_t*)task->data;
  for(uint32_t i = 0; i < taskSizeU16; ++i)
    dst[i] = 0x1500 | (src[i] << 16);
}

#endif // ~SPI_3WIRE_PROTOCOL

void WaitForPolledSPITransferToFinish()
{
  uint32_t cs;
  while (!(((cs = spi->cs) ^ BCM2835_SPI0_CS_TA) & (BCM2835_SPI0_CS_DONE | BCM2835_SPI0_CS_TA)))
    if ((cs & (BCM2835_SPI0_CS_RXR | BCM2835_SPI0_CS_RXF)))
      spi->cs = BCM2835_SPI0_CS_CLEAR_RX | BCM2835_SPI0_CS_TA | DISPLAY_SPI_DRIVE_SETTINGS;

  if ((cs & BCM2835_SPI0_CS_RXD)) spi->cs = BCM2835_SPI0_CS_CLEAR_RX | BCM2835_SPI0_CS_TA | DISPLAY_SPI_DRIVE_SETTINGS;
}

#ifdef ALL_TASKS_SHOULD_DMA

#ifndef USE_DMA_TRANSFERS
#error When building with #define ALL_TASKS_SHOULD_DMA enabled, -DUSE_DMA_TRANSFERS=ON should be set in CMake command line!
#endif

void RunSPITask(SPITask *task)
{
  uint32_t cs;
  uint8_t *tStart = task->PayloadStart();
  uint8_t *tEnd = task->PayloadEnd();
  const uint32_t payloadSize = tEnd - tStart;
  uint8_t *tPrefillEnd = tStart + MIN(15, payloadSize);

#define TASK_SIZE_TO_USE_DMA 4
  if (payloadSize >= TASK_SIZE_TO_USE_DMA && (task->cmd == DISPLAY_WRITE_PIXELS || task->cmd == DISPLAY_SET_CURSOR_X || task->cmd == DISPLAY_SET_CURSOR_Y))
  {
    if (previousTaskWasSPI)
      WaitForPolledSPITransferToFinish();
    SPIDMATransfer(task);
    previousTaskWasSPI = false;
  }
  else
  {
    if (!previousTaskWasSPI)
    {
      WaitForDMAFinished();
      spi->cs = BCM2835_SPI0_CS_TA | BCM2835_SPI0_CS_CLEAR_TX | DISPLAY_SPI_DRIVE_SETTINGS;
      UNLOCK_FAST_8_CLOCKS_SPI();
    }
    else
      WaitForPolledSPITransferToFinish();

#ifndef SPI_3WIRE_PROTOCOL
    CLEAR_GPIO(GPIO_TFT_DATA_CONTROL);

#ifdef DISPLAY_SPI_BUS_IS_16BITS_WIDE
    WRITE_FIFO(0x00);
#endif
    WRITE_FIFO(task->cmd);

#ifdef DISPLAY_SPI_BUS_IS_16BITS_WIDE
    while(!(spi->cs & (BCM2835_SPI0_CS_DONE))) /*nop*/;
    spi->fifo;
    spi->fifo;
#else
    while(!(spi->cs & (BCM2835_SPI0_CS_RXD|BCM2835_SPI0_CS_DONE))) /*nop*/;
#endif

    SET_GPIO(GPIO_TFT_DATA_CONTROL);
#endif

    while(tStart < tPrefillEnd) WRITE_FIFO(*tStart++);
    while(tStart < tEnd)
    {
      cs = spi->cs;
      if ((cs & BCM2835_SPI0_CS_TXD)) WRITE_FIFO(*tStart++);
      if ((cs & (BCM2835_SPI0_CS_RXR|BCM2835_SPI0_CS_RXF))) spi->cs = BCM2835_SPI0_CS_CLEAR_RX | BCM2835_SPI0_CS_TA | DISPLAY_SPI_DRIVE_SETTINGS;
    }

    previousTaskWasSPI = true;
  }
}
#else

void RunSPITask(SPITask *task)
{
  WaitForPolledSPITransferToFinish();

#ifdef DISPLAY_NEEDS_CHIP_SELECT_SIGNAL
  BEGIN_SPI_COMMUNICATION();
#endif

  uint8_t *tStart = task->PayloadStart();
  uint8_t *tEnd = task->PayloadEnd();
  const uint32_t payloadSize = tEnd - tStart;
  uint8_t *tPrefillEnd = tStart + MIN(15, payloadSize);

#ifndef SPI_3WIRE_PROTOCOL
  CLEAR_GPIO(GPIO_TFT_DATA_CONTROL);

#ifdef DISPLAY_SPI_BUS_IS_16BITS_WIDE
  WRITE_FIFO(0x00);
#endif
  WRITE_FIFO(task->cmd);

#ifdef DISPLAY_SPI_BUS_IS_16BITS_WIDE
  while(!(spi->cs & (BCM2835_SPI0_CS_DONE))) /*nop*/;
  spi->fifo;
  spi->fifo;
#else
  while(!(spi->cs & (BCM2835_SPI0_CS_RXD|BCM2835_SPI0_CS_DONE))) /*nop*/;
#endif

  SET_GPIO(GPIO_TFT_DATA_CONTROL);
#endif // ~!SPI_3WIRE_PROTOCOL

#define DMA_IS_FASTER_THAN_POLLED_SPI 140
#ifdef USE_DMA_TRANSFERS
  if (tEnd - tStart > DMA_IS_FASTER_THAN_POLLED_SPI)
  {
    SPIDMATransfer(task);
    UNLOCK_FAST_8_CLOCKS_SPI();
  }
  else
#endif
  {
    while(tStart < tPrefillEnd) WRITE_FIFO(*tStart++);
    while(tStart < tEnd)
    {
      uint32_t cs = spi->cs;
      if ((cs & BCM2835_SPI0_CS_TXD)) WRITE_FIFO(*tStart++);
      if ((cs & (BCM2835_SPI0_CS_RXR|BCM2835_SPI0_CS_RXF))) spi->cs = BCM2835_SPI0_CS_CLEAR_RX | BCM2835_SPI0_CS_TA | DISPLAY_SPI_DRIVE_SETTINGS;
    }
  }

#ifdef DISPLAY_NEEDS_CHIP_SELECT_SIGNAL
  END_SPI_COMMUNICATION();
#endif
}
#endif

SharedMemory *spiTaskMemory = 0;
volatile uint64_t spiThreadIdleUsecs = 0;
volatile uint64_t spiThreadSleepStartTime = 0;
volatile int spiThreadSleeping = 0;
double spiUsecsPerByte;

SPITask *GetTask()
{
  uint32_t head = spiTaskMemory->queueHead;
  uint32_t tail = spiTaskMemory->queueTail;
  if (head == tail) return 0;
  SPITask *task = (SPITask*)(spiTaskMemory->buffer + head);
  if (task->cmd == 0)
  {
    spiTaskMemory->queueHead = 0;
    __sync_synchronize();
    if (tail == 0) return 0;
    task = (SPITask*)spiTaskMemory->buffer;
  }
  return task;
}

void DoneTask(SPITask *task)
{
  __atomic_fetch_sub(&spiTaskMemory->spiBytesQueued, task->PayloadSize()+1, __ATOMIC_RELAXED);
  spiTaskMemory->queueHead = (uint32_t)((uint8_t*)task - spiTaskMemory->buffer) + sizeof(SPITask) + task->size;
  __sync_synchronize();
}

extern volatile bool programRunning;

void ExecuteSPITasks()
{
#ifndef USE_DMA_TRANSFERS
  BEGIN_SPI_COMMUNICATION();
#endif
  {
    while(programRunning && spiTaskMemory->queueTail != spiTaskMemory->queueHead)
    {
      SPITask *task = GetTask();
      if (task)
      {
        RunSPITask(task);
        DoneTask(task);
      }
    }
  }
#ifndef USE_DMA_TRANSFERS
  END_SPI_COMMUNICATION();
#endif
}

#if !defined(KERNEL_MODULE) && defined(USE_SPI_THREAD)
pthread_t spiThread;

void *spi_thread(void *unused)
{
#ifdef RUN_WITH_REALTIME_THREAD_PRIORITY
  SetRealtimeThreadPriority();
#endif
  while(programRunning)
  {
    if (spiTaskMemory->queueTail != spiTaskMemory->queueHead)
    {
      ExecuteSPITasks();
    }
    else
    {
#ifdef STATISTICS
      uint64_t t0 = tick();
      spiThreadSleepStartTime = t0;
      __atomic_store_n(&spiThreadSleeping, 1, __ATOMIC_RELAXED);
#endif
      if (programRunning) syscall(SYS_futex, &spiTaskMemory->queueTail, FUTEX_WAIT, spiTaskMemory->queueHead, 0, 0, 0);
#ifdef STATISTICS
      __atomic_store_n(&spiThreadSleeping, 0, __ATOMIC_RELAXED);
      uint64_t t1 = tick();
      __sync_fetch_and_add(&spiThreadIdleUsecs, t1-t0);
#endif
    }
  }
  pthread_exit(0);
}
#endif

int InitSPI()
{
#ifdef KERNEL_MODULE

#define BCM2835_PERI_BASE               0x3F000000
#define BCM2835_GPIO_BASE               0x200000
#define BCM2835_SPI0_BASE               0x204000
  printk("ioremapping %p\n", (void*)(BCM2835_PERI_BASE+BCM2835_GPIO_BASE));
  void *bcm2835 = ioremap(BCM2835_PERI_BASE+BCM2835_GPIO_BASE, 32768);
  printk("Got bcm address %p\n", bcm2835);
  if (!bcm2835) FATAL_ERROR("Failed to map BCM2835 address!");
  spi = (volatile SPIRegisterFile*)((uintptr_t)bcm2835 + BCM2835_SPI0_BASE - BCM2835_GPIO_BASE);
  gpio = (volatile GPIORegisterFile*)((uintptr_t)bcm2835);

#else // Userland version
  mem_fd = open("/dev/mem", O_RDWR|O_SYNC);
  if (mem_fd < 0) FATAL_ERROR("can't open /dev/mem (run as sudo)");

  uintptr_t peri_addr = get_bcm_peripheral_address();
  uint32_t peri_size = get_bcm_peripheral_size();
  uintptr_t sdram_addr = get_bcm_sdram_address();

  printf("bcm_peripheral_address: %p, bcm_peripheral_size: %u, bcm_sdram_address: %p\n", (void*)peri_addr, peri_size, (void*)sdram_addr);
  
  bcm2835 = mmap(NULL, peri_size, (PROT_READ | PROT_WRITE), MAP_SHARED, mem_fd, peri_addr);
  if (bcm2835 == MAP_FAILED) FATAL_ERROR("mapping /dev/mem failed");
  spi = (volatile SPIRegisterFile*)((uintptr_t)bcm2835 + BCM2835_SPI0_BASE);
  gpio = (volatile GPIORegisterFile*)((uintptr_t)bcm2835 + BCM2835_GPIO_BASE);
  systemTimerRegister = (volatile uint64_t*)((uintptr_t)bcm2835 + BCM2835_TIMER_BASE + 0x04);
#endif

  uint32_t currentBcmCoreSpeed = MailboxRet2(0x00030002/*Get Clock Rate*/, 0x4/*CORE*/);
  uint32_t maxBcmCoreTurboSpeed = MailboxRet2(0x00030004/*Get Max Clock Rate*/, 0x4/*CORE*/);

  spiUsecsPerByte = 1000000.0 * 8.0 * SPI_BUS_CLOCK_DIVISOR / maxBcmCoreTurboSpeed;

  printf("BCM core speed: current: %uhz, max turbo: %uhz. SPI CDIV: %d, SPI max frequency: %.0fhz\n", currentBcmCoreSpeed, maxBcmCoreTurboSpeed, SPI_BUS_CLOCK_DIVISOR, (double)maxBcmCoreTurboSpeed / SPI_BUS_CLOCK_DIVISOR);

#if !defined(KERNEL_MODULE_CLIENT) || defined(KERNEL_MODULE_CLIENT_DRIVES)
#ifdef GPIO_TFT_DATA_CONTROL
  SET_GPIO_MODE(GPIO_TFT_DATA_CONTROL, 0x01);
#endif
#if !defined(GPIO_TFT_DATA_CONTROL) || GPIO_TFT_DATA_CONTROL != GPIO_SPI0_MISO
  SET_GPIO_MODE(GPIO_SPI0_MISO, 0x04);
#endif
  SET_GPIO_MODE(GPIO_SPI0_MOSI, 0x04);
  SET_GPIO_MODE(GPIO_SPI0_CLK, 0x04);

#ifdef DISPLAY_NEEDS_CHIP_SELECT_SIGNAL
  SET_GPIO_MODE(GPIO_SPI0_CE0, 0x04);
#ifdef DISPLAY_USES_CS1
  SET_GPIO_MODE(GPIO_SPI0_CE1, 0x04);
#endif
#else
  SET_GPIO_MODE(GPIO_SPI0_CE0, 0x01);
  CLEAR_GPIO(GPIO_SPI0_CE0);
#ifdef DISPLAY_USES_CS1
  SET_GPIO_MODE(GPIO_SPI0_CE1, 0x01);
#endif
#endif

  spi->cs = BCM2835_SPI0_CS_CLEAR | DISPLAY_SPI_DRIVE_SETTINGS;
  spi->clk = SPI_BUS_CLOCK_DIVISOR;
#endif

#ifdef KERNEL_MODULE_CLIENT
  int driverfd = open("/proc/bcm2835_spi_display_bus", O_RDWR|O_SYNC);
  if (driverfd < 0) FATAL_ERROR("Could not open SPI ring buffer - kernel driver module not running?");
  spiTaskMemory = (SharedMemory*)mmap(NULL, SHARED_MEMORY_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, driverfd, 0);
  close(driverfd);
  if (spiTaskMemory == MAP_FAILED) FATAL_ERROR("Could not mmap SPI ring buffer!");
  printf("Got shared memory block %p, ring buffer head %p, ring buffer tail %p, shared memory block phys address: %p\n", (const char *)spiTaskMemory, spiTaskMemory->queueHead, spiTaskMemory->queueTail, spiTaskMemory->sharedMemoryBaseInPhysMemory);

#ifdef USE_DMA_TRANSFERS
  printf("DMA TX channel: %d, DMA RX channel: %d\n", spiTaskMemory->dmaTxChannel, spiTaskMemory->dmaRxChannel);
#endif

#else

#ifdef KERNEL_MODULE
  spiTaskMemory = (SharedMemory*)kmalloc(SHARED_MEMORY_SIZE, GFP_KERNEL | GFP_DMA);
  dmaSourceMemory = (SharedMemory*)dma_alloc_writecombine(0, SHARED_MEMORY_SIZE, &spiTaskMemoryPhysical, GFP_KERNEL);
  LOG("Allocated DMA memory: mem: %p, phys: %p", spiTaskMemory, (void*)spiTaskMemoryPhysical);
  memset((void*)spiTaskMemory, 0, SHARED_MEMORY_SIZE);
#else
  spiTaskMemory = (SharedMemory*)Malloc(SHARED_MEMORY_SIZE, "spi.cpp shared task memory");
#endif

  spiTaskMemory->queueHead = spiTaskMemory->queueTail = spiTaskMemory->spiBytesQueued = 0;
#endif

#ifdef USE_DMA_TRANSFERS
  InitDMA();
#endif

  UNLOCK_FAST_8_CLOCKS_SPI();

#if !defined(KERNEL_MODULE) && (!defined(KERNEL_MODULE_CLIENT) || defined(KERNEL_MODULE_CLIENT_DRIVES))
  printf("Initializing display\n");
  InitSPIDisplay();

#ifdef USE_SPI_THREAD
  printf("Creating SPI task thread\n");
  int rc = pthread_create(&spiThread, NULL, spi_thread, NULL);
  if (rc != 0) FATAL_ERROR("Failed to create SPI thread!");
#else
  BEGIN_SPI_COMMUNICATION();
#endif

#endif

  LOG("InitSPI done");
  return 0;
}

void DeinitSPI()
{
#ifdef USE_SPI_THREAD
  pthread_join(spiThread, NULL);
  spiThread = (pthread_t)0;
#endif
  DeinitSPIDisplay();
#ifdef USE_DMA_TRANSFERS
  DeinitDMA();
#endif

  spi->cs = BCM2835_SPI0_CS_CLEAR | DISPLAY_SPI_DRIVE_SETTINGS;

#ifndef KERNEL_MODULE_CLIENT
#ifdef GPIO_TFT_DATA_CONTROL
  SET_GPIO_MODE(GPIO_TFT_DATA_CONTROL, 0);
#endif
  SET_GPIO_MODE(GPIO_SPI0_CE1, 0);
  SET_GPIO_MODE(GPIO_SPI0_CE0, 0);
  SET_GPIO_MODE(GPIO_SPI0_MISO, 0);
  SET_GPIO_MODE(GPIO_SPI0_MOSI, 0);
  SET_GPIO_MODE(GPIO_SPI0_CLK, 0);
#endif

  if (bcm2835)
  {
    munmap((void*)bcm2835, get_bcm_peripheral_size());
    bcm2835 = 0;
  }

  if (mem_fd >= 0)
  {
    close(mem_fd);
    mem_fd = -1;
  }

#ifndef KERNEL_MODULE_CLIENT

#ifdef KERNEL_MODULE
  kfree(spiTaskMemory);
  dma_free_writecombine(0, SHARED_MEMORY_SIZE, dmaSourceMemory, spiTaskMemoryPhysical);
  spiTaskMemoryPhysical = 0;
#else
  free(spiTaskMemory);
#endif
#endif
  spiTaskMemory = 0;
}
