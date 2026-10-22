/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Object detection demo: a YOLO26 detection model, by default YOLO26n with
 * the 80 COCO classes, on the Ethos-U85 of the Alif Ensemble E8 (M55-HP,
 * DevKit-E8), live from the camera to the panel.
 *
 * Each frame:
 *   1. the input: the newest camera frame (the MT9M114's 640x480 RGB565),
 *      its centre square scaled to the 416x416 RGB888 model input by the
 *      camera thread; without a camera, the test image (detect/test_image.c)
 *   2. the detector: the model on the NPU, the head's decode on the CPU (detector.cpp)
 *   3. the panel: the picture in a 480x480 view; the boxes with class and
 *      score, the frame rate and the timing around it in the display thread,
 *      while the NPU works on the next frame; double-buffered.
 *
 * The console is a log buffer that the debugger reads (board layer); the
 * latest result and the frame timing are in the global `detect_status`.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "RTE_Components.h"
#include CMSIS_device_header

#include "cmsis_os2.h"
#include "se_services_port.h"
#include "services_lib_api.h"
extern "C" {
#include "DSI_DCS.h"  // the pack header has no C++ guard
}

#include "board_display.h"
#include "camera.h"
#include "detector.h"
#include "image.h"

#include "test_image.h"

#ifndef DETECT_CAMERA_AUTOSTART
#define DETECT_CAMERA_AUTOSTART 1
#endif

// ---------------------------------------------------------------------------
// What the debugger reads.
// ---------------------------------------------------------------------------
struct DetectStatus {
  uint32_t frames;          // frames processed
  int32_t source;           // 0 test image, 1 camera
  int32_t camera_status;    // camera_init(): 0 ok, else the failing step; -2 not started
  uint32_t camera_frames;   // frames the camera delivered
  uint32_t camera_errors;
  int32_t detector_status;  // last ExecuTorch error, 0 ok
  uint32_t frame_us;        // last frame, input to handed to the display thread
  uint32_t input_us;        // waiting for the camera input
  uint32_t convert_us;      // camera thread: camera frame to model input
  uint32_t detect_us;       // detector_run: execute and decode
  uint32_t npu_us;          // the NPU job alone
  uint32_t decode_us;       // the decode alone
  uint32_t picture_us;      // the input into the frame buffer
  uint32_t display_us;      // display thread: boxes, text and cache clean
  float fps;                // frames per second, averaged over one second
  detections_t result;      // the detections of the last frame
};

extern "C" {
volatile DetectStatus detect_status;
// Debugging aid: with DETECT_CAMERA_AUTOSTART 0 the camera stays off until
// the debugger sets detect_camera_enable before app_main gets to it.
volatile int detect_camera_enable = DETECT_CAMERA_AUTOSTART;
}

namespace {

constexpr int kSize = TEST_IMAGE_SIZE;
constexpr uint32_t kInputBytes = kSize * kSize * 3;
constexpr int kSlots = 2;

// The NPU the board layer names: for the console banner, and upper case for
// the panel's header (the font has no lower case).
#ifdef ETHOSU85
#define DETECT_NPU_NAME "Ethos-U85"
#define DETECT_NPU_LABEL "ETHOS-U85"
#else
#define DETECT_NPU_NAME "Ethos-U55"
#define DETECT_NPU_LABEL "ETHOS-U55"
#endif

// The model inputs: two slots in SRAM1 that the camera
// thread fills in turn; without a camera slot 0 holds the test image.
alignas(32) uint8_t g_slot[kSlots][kInputBytes] __attribute__((section(".bss.sram1")));

// Two frame buffers in SRAM0 (the CDC200 cannot read SRAM1): the picture and
// the overlays go into the one the panel does not show, then the panel switches.
constexpr uint32_t kFrameBytes = IMAGE_PANEL_W * IMAGE_PANEL_H * 3;
alignas(32) uint8_t g_framebuffer0[kFrameBytes] __attribute__((section(APP_FRAMEBUFFER_SECTION)));
alignas(32) uint8_t g_framebuffer1[kFrameBytes] __attribute__((section(APP_FRAMEBUFFER_SECTION)));
uint8_t* const g_framebuffer[2] = {g_framebuffer0, g_framebuffer1};

// SRAM1 holds the input slots and the NPU scratch. Power and clock it through
// the Secure Enclave before any use; a store to an unpowered SRAM hangs the bus.
bool sram1_power_on() {
  uint32_t error = 0;
  if (SERVICES_power_memory_req(se_services_s_handle, POWER_MEM_SRAM_0_ENABLE | POWER_MEM_SRAM_1_ENABLE, &error) !=
          SERVICES_REQ_SUCCESS ||
      error != 0) {
    printf("SRAM1: power request failed (error %lu)\n", static_cast<unsigned long>(error));
    return false;
  }
  if (SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_SRAM1, true, &error) != SERVICES_REQ_SUCCESS || error != 0) {
    printf("SRAM1: clock request failed (error %lu)\n", static_cast<unsigned long>(error));
    return false;
  }
  return true;
}

inline uint32_t cycles() { return DWT->CYCCNT; }
inline uint32_t us(uint32_t c) { return static_cast<uint32_t>(static_cast<uint64_t>(c) * 1000000U / SystemCoreClock); }

enum Source { kTestImage = 0, kCamera = 1 };

// ---------------------------------------------------------------------------
// The camera thread: every camera frame turned into a model input in the
// slot the vision thread is not reading. The vision thread takes the newest
// slot and holds it for the frame; the camera thread writes the other one,
// so the vision thread always finds the newest complete frame. A frame that
// arrives while a converted one still waits unread is skipped. The thread
// takes the frames above the vision thread but converts below it, so the
// conversion goes into the vision thread's wait for the NPU.
// ---------------------------------------------------------------------------
volatile int g_slot_newest = -1;   // the newest complete input, -1 when taken
volatile int g_slot_reading = -1;  // the vision thread's input
osEventFlagsId_t g_slot_event;
uint64_t g_camera_stack[512] __attribute__((section(APP_POOL_SECTION)));
volatile uint32_t g_convert_cycles;

__NO_RETURN void camera_thread(void*) {
  for (;;) {
    const void* frame = camera_frame(200);
    if (frame == nullptr || g_slot_newest >= 0) continue;
    osKernelLock();
    int slot = 0;
    while (slot == g_slot_reading) ++slot;
    osKernelUnlock();
    osThreadSetPriority(osThreadGetId(), osPriorityBelowNormal);
    const uint32_t t0 = cycles();
    image_rgb565_to_input(static_cast<const uint16_t*>(frame), CAMERA_WIDTH, CAMERA_HEIGHT, g_slot[slot], kSize,
                          CAMERA_QUARTER_TURNS);
    g_convert_cycles = cycles() - t0;
    osThreadSetPriority(osThreadGetId(), osPriorityAboveNormal);
    g_slot_newest = slot;
    osEventFlagsSet(g_slot_event, 1U);
  }
}

// The newest camera input, waiting up to `timeout_ms`; nullptr on a timeout.
const uint8_t* camera_input(uint32_t timeout_ms) {
  for (;;) {
    osKernelLock();
    const int slot = g_slot_newest;
    if (slot >= 0) {
      g_slot_reading = slot;
      g_slot_newest = -1;
    }
    osKernelUnlock();
    if (slot >= 0) return g_slot[slot];
    if (osEventFlagsWait(g_slot_event, 1U, osFlagsWaitAny, timeout_ms) & osFlagsError) return nullptr;
  }
}

void camera_input_done() { g_slot_reading = -1; }

// ---------------------------------------------------------------------------
// The panel.
// ---------------------------------------------------------------------------
constexpr uint32_t kGrey = 0xC0C0C0U, kDim = 0x707070U, kGreen = 0x00FF00U, kWhite = 0xFFFFFFU, kBlack = 0x000000U;

// Black or white text, whichever reads better on `rgb`.
uint32_t text_on(uint32_t rgb) {
  const uint32_t luma = (299U * ((rgb >> 16) & 0xFFU) + 587U * ((rgb >> 8) & 0xFFU) + 114U * (rgb & 0xFFU)) / 1000U;
  return luma > 150U ? kBlack : kWhite;
}

// The detections on the picture (`size` pixels square, 1:1, centred in the
// view): a box in the class colour, the class and the score on a label in
// the same colour at its top. Everything stays inside the picture: the
// margins around it are never redrawn.
void draw_detections(uint8_t* fb, const detections_t& det, int size) {
  const int x0 = (IMAGE_VIEW - size) / 2, y0 = IMAGE_VIEW_TOP + (IMAGE_VIEW - size) / 2;
  char label[32];
  // Weakest first, so that the strongest box and label end up on top.
  for (int i = static_cast<int>(det.count) - 1; i >= 0; --i) {
    const detection_t& d = det.det[i];
    int x1 = x0 + static_cast<int>(d.x1), x2 = x0 + static_cast<int>(d.x2);
    int y1 = y0 + static_cast<int>(d.y1), y2 = y0 + static_cast<int>(d.y2);
    x1 = x1 < x0 ? x0 : x1;
    y1 = y1 < y0 ? y0 : y1;
    x2 = x2 > x0 + size ? x0 + size : x2;
    y2 = y2 > y0 + size ? y0 + size : y2;
    if (x2 - x1 < 4 || y2 - y1 < 4) continue;
    const uint32_t colour = image_class_colour(d.cls);
    image_box(fb, x1, y1, x2, y2, 2, colour);
    snprintf(label, sizeof(label), "%s %.2f", DETECTOR_CLASS_NAMES[d.cls], static_cast<double>(d.score));
    // Above the box, or inside it at the top edge; shifted left and up to
    // stay inside the picture.
    const int lw = 12 * static_cast<int>(strlen(label)) + 4;
    int lx = x1 + lw <= x0 + size ? x1 : x0 + size - lw;
    lx = lx < x0 ? x0 : lx;
    int ly = y1 - 18 >= y0 ? y1 - 18 : y1;
    ly = ly + 18 <= y0 + size ? ly : y0 + size - 18;
    image_fill(fb, lx, ly, lx + lw, ly + 18, colour);
    image_text(fb, lx + 2, ly + 2, 2, label, text_on(colour));
  }
}

// Around the picture (480 x 160 above, 480 x 160 below):
//   above: the title, the frame rate, the NPU, detector and frame times
//   below: how many of each class are in the picture, the input source
void draw_status(uint8_t* fb, const detections_t& det, Source source, float fps, uint32_t frame_us) {
  image_fill(fb, 0, 0, IMAGE_PANEL_W, IMAGE_VIEW_TOP, kBlack);
  image_fill(fb, 0, IMAGE_VIEW_TOP + IMAGE_VIEW, IMAGE_PANEL_W, IMAGE_PANEL_H, kBlack);

  char line[48];
  image_text(fb, 12, 6, 2, "OBJECT DETECTION  " MODEL_DETECT_NAME "  " DETECT_NPU_LABEL, kDim);
  snprintf(line, sizeof(line), "%.1f", static_cast<double>(fps));
  const int x = image_text(fb, 12, 40, 7, line, kWhite);
  image_text(fb, x + 4, 76, 2, "FPS", kGrey);
  snprintf(line, sizeof(line), "NPU    %5.1f MS", det.npu_us / 1000.0);
  image_text(fb, 276, 40, 2, line, kGrey);
  snprintf(line, sizeof(line), "DETECT %5.1f MS", det.total_us / 1000.0);
  image_text(fb, 276, 62, 2, line, kGrey);
  snprintf(line, sizeof(line), "FRAME  %5.1f MS", frame_us / 1000.0);
  image_text(fb, 276, 84, 2, line, kGrey);
  snprintf(line, sizeof(line), "FORMAT=ETHOS  INT8 IO  SCORE > %.2f", static_cast<double>(DETECTOR_SCORE_THRESHOLD));
  image_text(fb, 12, 128, 2, line, kDim);

  // The classes in the picture, strongest first, with their counts.
  int count[DETECTOR_CLASSES] = {};
  for (uint32_t i = 0; i < det.count; ++i) ++count[det.det[i].cls];
  int cx = 12, cy = IMAGE_VIEW_TOP + IMAGE_VIEW + 12;
  for (uint32_t i = 0; i < det.count && cy < IMAGE_PANEL_H - 50; ++i) {
    const int cls = det.det[i].cls;
    if (count[cls] == 0) continue;  // listed already
    snprintf(line, sizeof(line), "%s %d", DETECTOR_CLASS_NAMES[cls], count[cls]);
    count[cls] = 0;
    const int w = 18 + 12 * static_cast<int>(strlen(line)) + 18;
    if (cx + w > IMAGE_PANEL_W) {
      cx = 12;
      cy += 24;
      if (cy >= IMAGE_PANEL_H - 50) break;
    }
    image_fill(fb, cx, cy + 2, cx + 12, cy + 14, image_class_colour(cls));
    image_text(fb, cx + 18, cy, 2, line, kWhite);
    cx += w;
  }
  if (det.count == 0) image_text(fb, 12, IMAGE_VIEW_TOP + IMAGE_VIEW + 12, 2, "NOTHING ABOVE THE THRESHOLD", kDim);

  const int y = IMAGE_PANEL_H - 30;
  image_fill(fb, 12, y, 28, y + 16, kGreen);
  image_text(fb, 36, y + 2, 2, source == kCamera ? "CAMERA" : "TEST IMAGE", kGreen);
  snprintf(line, sizeof(line), "%lu OBJECT%s", static_cast<unsigned long>(det.count), det.count == 1 ? "" : "S");
  image_text(fb, IMAGE_PANEL_W - 12 - 12 * static_cast<int>(strlen(line)), y + 2, 2, line, kGrey);
}

// The display thread draws everything but the picture while the vision
// thread runs the NPU on the next frame: the picture goes into the back
// buffer in the vision thread, the rest here, from a copy of the frame's results.
struct DisplayJob {
  uint8_t* fb;
  detections_t det;
  Source source;
  float fps;
  uint32_t frame_us;
};
DisplayJob g_job;
osSemaphoreId_t g_display_go;    // a job is ready
osSemaphoreId_t g_display_idle;  // the last job is on the panel: the other buffer is free
uint64_t g_display_stack[1024] __attribute__((section(APP_POOL_SECTION)));

__NO_RETURN void display_thread(void*) {
  for (;;) {
    osSemaphoreAcquire(g_display_go, osWaitForever);
    const uint32_t t0 = cycles();
    uint8_t* fb = g_job.fb;
    draw_detections(fb, g_job.det, kSize);
    draw_status(fb, g_job.det, g_job.source, g_job.fps, g_job.frame_us);
    SCB_CleanDCache_by_Addr(fb, static_cast<int32_t>(kFrameBytes));
    detect_status.display_us = us(cycles() - t0);
    display_present(fb);
    // Presented takes effect at the panel's next refresh: until then the
    // other buffer is still on the panel and must not be drawn over.
    display_wait_shown(fb);
    osSemaphoreRelease(g_display_idle);
  }
}

}  // namespace

extern "C" int app_main(void) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  memset(const_cast<DetectStatus*>(&detect_status), 0, sizeof(detect_status));
  printf("Object detection: " MODEL_DETECT_NAME " (%d classes) on the " DETECT_NPU_NAME ", input %dx%d RGB888, core %lu MHz\n",
         DETECTOR_CLASSES, kSize, kSize, static_cast<unsigned long>(SystemCoreClock / 1000000U));

  if (!sram1_power_on()) return 1;
  int32_t status = detector_init();
  if (status != 0) {
    detect_status.detector_status = status;
    printf("detector: method load failed (err=%ld)\n", static_cast<long>(status));
    return 1;
  }

  const int32_t camera_status = detect_camera_enable ? camera_init() : -2;
  printf("camera: %s (%ld)\n", camera_status == 0 ? "640x480 RGB565" : "not available, the test image instead",
         static_cast<long>(camera_status));
  detect_status.camera_status = camera_status;
  const bool camera_on = camera_status == 0;
  if (camera_on) {
    g_slot_event = osEventFlagsNew(nullptr);
    static const osThreadAttr_t attr = {.name = "camera", .stack_mem = g_camera_stack, .stack_size = sizeof(g_camera_stack),
                                        .priority = osPriorityAboveNormal};
    osThreadNew(camera_thread, nullptr, &attr);
  } else {
    memcpy(g_slot[0], test_image, kInputBytes);
  }

  for (uint8_t* fb : g_framebuffer) {
    memset(fb, 0, kFrameBytes);
    SCB_CleanDCache_by_Addr(fb, static_cast<int32_t>(kFrameBytes));
  }
  bool display_on = display_init() == 0;
  // The panel's memory access control (MADCTL, 0x36) before the video stream
  // starts: the board layer's value shows the frame buffer as drawn.
  if (display_on) DSI_DCS_Short_Write(0x36, APP_LCD_MADCTL);
  if (display_on) display_on = display_start(g_framebuffer[1]) == 0;
  if (!display_on) printf("display: not available\n");
  int back = 0;
  if (display_on) {
    g_display_go = osSemaphoreNew(1, 0, nullptr);
    g_display_idle = osSemaphoreNew(1, 1, nullptr);
    // Below the vision thread: when the NPU is done, the vision thread goes on at once.
    static const osThreadAttr_t attr = {.name = "display", .stack_mem = g_display_stack,
                                        .stack_size = sizeof(g_display_stack), .priority = osPriorityBelowNormal};
    osThreadNew(display_thread, nullptr, &attr);
  }

  uint32_t fps_frames = 0, fps_start = osKernelGetTickCount();
  float fps = 0.0f;
  uint32_t frame_us = 0;
  detections_t det{};
  for (uint32_t frame = 0;; ++frame) {
    const uint32_t t_frame = cycles();

    // 1. The input.
    const Source source = camera_on ? kCamera : kTestImage;
    const uint8_t* input = g_slot[0];
    uint32_t t0 = cycles();
    if (source == kCamera) {
      input = camera_input(200);  // the camera thread converted it already
      if (input == nullptr) {
        printf("camera: no frame for 200 ms (%lu frames, %lu errors)\n", static_cast<unsigned long>(camera_frame_count()),
               static_cast<unsigned long>(camera_error_count()));
        continue;
      }
    }
    const uint32_t t_input = cycles() - t0;

    // 2. The detector.
    t0 = cycles();
    status = detector_run(input, &det);
    const uint32_t t_detect = cycles() - t0;
    det.frame = frame;
    if (status != 0) {
      detect_status.detector_status = status;
      printf("detector: run failed (err=%ld)\n", static_cast<long>(status));
      if (source == kCamera) camera_input_done();
      osDelay(1000);
      continue;
    }

    // 3. The panel: the picture here, the rest in the display thread while
    //    the NPU works on the next frame.
    t0 = cycles();
    if (display_on) {
      osSemaphoreAcquire(g_display_idle, osWaitForever);  // the back buffer is off the panel
      uint8_t* fb = g_framebuffer[back];
      image_input_to_view(input, kSize, fb);
      g_job.fb = fb;
      g_job.det = det;
      g_job.source = source;
      g_job.fps = fps;
      g_job.frame_us = frame_us;
      osSemaphoreRelease(g_display_go);
      back ^= 1;
    }
    const uint32_t t_picture = cycles() - t0;
    if (source == kCamera) camera_input_done();

    // The status for the debugger, the console every 100 frames.
    ++fps_frames;
    const uint32_t now = osKernelGetTickCount();
    if (now - fps_start >= 1000U) {
      fps = fps_frames * 1000.0f / static_cast<float>(now - fps_start);
      fps_frames = 0;
      fps_start = now;
    }
    frame_us = us(cycles() - t_frame);
    detect_status.frames = frame + 1;
    detect_status.source = source;
    if (camera_on) {
      detect_status.camera_frames = camera_frame_count();
      detect_status.camera_errors = camera_error_count();
      detect_status.convert_us = us(g_convert_cycles);
    }
    detect_status.detector_status = 0;
    detect_status.frame_us = frame_us;
    detect_status.input_us = us(t_input);
    detect_status.detect_us = us(t_detect);
    detect_status.npu_us = det.npu_us;
    detect_status.decode_us = det.decode_us;
    detect_status.picture_us = us(t_picture);
    detect_status.fps = fps;
    memcpy(const_cast<detections_t*>(&detect_status.result), &det, sizeof(det));
    if (frame % 100 == 0) {
      printf("frame %lu: %lu object%s, %.1f fps; frame %lu us = input %lu + detect %lu (NPU %lu, decode %lu) + picture %lu; display %lu, convert %lu\n",
             static_cast<unsigned long>(frame), static_cast<unsigned long>(det.count), det.count == 1 ? "" : "s",
             static_cast<double>(fps), static_cast<unsigned long>(frame_us), static_cast<unsigned long>(detect_status.input_us),
             static_cast<unsigned long>(detect_status.detect_us), static_cast<unsigned long>(det.npu_us),
             static_cast<unsigned long>(det.decode_us), static_cast<unsigned long>(detect_status.picture_us),
             static_cast<unsigned long>(detect_status.display_us), static_cast<unsigned long>(detect_status.convert_us));
      for (uint32_t i = 0; i < det.count; ++i) {
        const detection_t& d = det.det[i];
        printf("  %-14s %.2f  x1 %.0f y1 %.0f x2 %.0f y2 %.0f\n", DETECTOR_CLASS_NAMES[d.cls], static_cast<double>(d.score),
               static_cast<double>(d.x1), static_cast<double>(d.y1), static_cast<double>(d.x2), static_cast<double>(d.y2));
      }
    }
  }
}
