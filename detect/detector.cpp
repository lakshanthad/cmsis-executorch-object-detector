/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The object detector: the program of `yolo export format=ethos` for the
 * Ethos-U85-256, imported by detect/make_ai_layer.py into ai_layer_detect/,
 * the single method `forward` with the whole graph on the Ethos-U85 (its two
 * attention blocks included). It returns, per anchor, the box distances and
 * one score per COCO class (int8). The CPU does what is left of the YOLO26
 * head: the best class of each anchor, the threshold, the distances of the
 * survivors turned into boxes. The one-to-one head of
 * YOLO26 is NMS-free; an IoU check still drops the rare duplicate.
 * traffic/detector.cpp is the same with the five vehicle classes.
 *
 * The Ethos-U backend copies the input into the NPU scratch with
 * arm_ethos_io_memcpy(); the override below turns RGB888 into the int8 input
 * on the way (the input scale is 1/255 with zero point -128, so
 * q = pixel - 128 = pixel ^ 0x80): no staging buffer, no conversion pass.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include <arm_mve.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include "arm_embedded_module.hpp"
#include "detector.h"
#include "model_io.h"
#include "model_pte.h"

using arm::embedded::EmbeddedModule;
using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;

namespace {

// Score threshold of a detection (detector.h), and the overlap above which
// the weaker of two boxes of a class is a duplicate.
constexpr float kScoreThreshold = DETECTOR_SCORE_THRESHOLD;
constexpr float kIouDuplicate = 0.70f;

constexpr int32_t kInputShape[] = MODEL_DETECT_INPUT0_SHAPE;  // {1, S, S, 3}
constexpr int kSize = kInputShape[1];
constexpr int32_t kBoxShape[] = MODEL_DETECT_OUTPUT0_SHAPE;  // {1, N, 4}
constexpr int kAnchors = kBoxShape[1];
constexpr int32_t kClsShape[] = MODEL_DETECT_OUTPUT1_SHAPE;  // {1, N, C}
constexpr int kClasses = kClsShape[2];
constexpr int kStrides[] = {8, 16, 32};
static_assert(kAnchors == (kSize / 8) * (kSize / 8) + (kSize / 16) * (kSize / 16) + (kSize / 32) * (kSize / 32),
              "anchor count");
static_assert(kClasses == DETECTOR_CLASSES, "class count of the program");

// Method pool: the runtime's structures and the method's planned buffer, the
// two outputs (box 14 kB, the 80 class scores per anchor 284 kB at 416 x 416);
// 298,460 bytes used at 416 x 416 (detector_method_pool_used). Temp pool:
// Vela's scratch for the whole graph, 1223.7 KiB at 416 x 416.
constexpr size_t kMethodPoolSize = 0x50000;
constexpr size_t kTempPoolSize = 0x140000;
static_assert(MODEL_DETECT_PLANNED_BYTES < kMethodPoolSize, "the program's planned buffer does not fit the method pool");
static_assert(MODEL_DETECT_SCRATCH_BYTES <= kTempPoolSize, "Vela's scratch does not fit the temp pool");
alignas(16) uint8_t g_method_pool[kMethodPoolSize] __attribute__((section(APP_POOL_SECTION)));
#ifndef APP_TEMP_POOL_SECTION
#define APP_TEMP_POOL_SECTION APP_POOL_SECTION
#endif
// The NPU's scratch in SRAM1 (the board layer's APP_TEMP_POOL_SECTION): SRAM0
// holds the panel buffers and the camera frames.
alignas(16) uint8_t g_temp_pool[kTempPoolSize] __attribute__((section(APP_TEMP_POOL_SECTION)));

const uint8_t* g_input_rgb;  // the frame the input copy converts
uint32_t g_npu_begin, g_npu_cycles;

inline uint32_t cycles() { return DWT->CYCCNT; }
inline uint32_t us(uint32_t c) { return static_cast<uint32_t>(static_cast<uint64_t>(c) * 1000000U / SystemCoreClock); }

// The module and the input tensor live as long as the application.
alignas(EmbeddedModule) uint8_t g_module_storage[sizeof(EmbeddedModule)];
EmbeddedModule* g_module;
std::array<SizesType, 4> g_sizes;
std::array<DimOrderType, 4> g_dim_order;
alignas(TensorImpl) uint8_t g_input_storage[sizeof(TensorImpl)];
TensorImpl* g_input;

float iou(const detection_t& a, const detection_t& b) {
  const float w = fminf(a.x2, b.x2) - fmaxf(a.x1, b.x1);
  const float h = fminf(a.y2, b.y2) - fmaxf(a.y1, b.y1);
  if (w <= 0.0f || h <= 0.0f) return 0.0f;
  const float inter = w * h;
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

// The rest of the YOLO26 head: the best class of each anchor, the anchors
// above the threshold, their distances to boxes. Both outputs are
// anchor-major: box[a * 4 + side], score[a * kClasses + class].
uint32_t decode(const int8_t* box, const int8_t* score, detection_t* out) {
  constexpr float kBoxScale = MODEL_DETECT_OUTPUT0_SCALE;
  constexpr int kBoxZp = MODEL_DETECT_OUTPUT0_ZERO_POINT;
  constexpr float kScoreScale = MODEL_DETECT_OUTPUT1_SCALE;
  constexpr int kScoreZp = MODEL_DETECT_OUTPUT1_ZERO_POINT;
  // Score > threshold, as Ultralytics' predict: the highest int8 step at or below it is dropped.
  const int threshold_q = static_cast<int>(floorf(kScoreThreshold / kScoreScale)) + kScoreZp;
  detection_t found[DETECTOR_MAX_DETECTIONS * 2];
  int n = 0;
  int base = 0;
  for (int stride : kStrides) {
    const int cells = kSize / stride;
    for (int a = base; a < base + cells * cells; ++a) {
      const int8_t* scores = score + a * kClasses;
      int8_t best = scores[0];
      int cls = 0;
      for (int c = 1; c < kClasses; ++c) {
        if (scores[c] > best) {
          best = scores[c];
          cls = c;
        }
      }
      if (best <= threshold_q) continue;
      const int cell = a - base;
      const float cx = static_cast<float>(cell % cells) + 0.5f;
      const float cy = static_cast<float>(cell / cells) + 0.5f;
      const float s = static_cast<float>(stride);
      detection_t d;
      const int8_t* ltrb = box + a * 4;
      d.x1 = (cx - (ltrb[0] - kBoxZp) * kBoxScale) * s;
      d.y1 = (cy - (ltrb[1] - kBoxZp) * kBoxScale) * s;
      d.x2 = (cx + (ltrb[2] - kBoxZp) * kBoxScale) * s;
      d.y2 = (cy + (ltrb[3] - kBoxZp) * kBoxScale) * s;
      d.score = (best - kScoreZp) * kScoreScale;
      d.cls = cls;
      if (n < DETECTOR_MAX_DETECTIONS * 2) {
        found[n++] = d;
      } else {  // full: replace the weakest
        int weakest = 0;
        for (int i = 1; i < n; ++i)
          if (found[i].score < found[weakest].score) weakest = i;
        if (d.score > found[weakest].score) found[weakest] = d;
      }
    }
    base += cells * cells;
  }

  // Strongest first, drop what overlaps a stronger box of the same class.
  for (int i = 1; i < n; ++i)
    for (int j = i; j > 0 && found[j].score > found[j - 1].score; --j) std::swap(found[j], found[j - 1]);
  uint32_t kept = 0;
  for (int i = 0; i < n && kept < DETECTOR_MAX_DETECTIONS; ++i) {
    bool duplicate = false;
    for (uint32_t k = 0; k < kept && !duplicate; ++k)
      duplicate = found[i].cls == out[k].cls && iou(found[i], out[k]) > kIouDuplicate;
    if (!duplicate) out[kept++] = found[i];
  }
  return kept;
}

// The method allocator, with its high-water mark for the debugger
// (detector_method_pool_used).
class TrackingAllocator : public MemoryAllocator {
 public:
  using MemoryAllocator::MemoryAllocator;
  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    void* p = MemoryAllocator::allocate(size, alignment);
    if (p != nullptr) used = static_cast<size_t>(static_cast<uint8_t*>(p) + size - base_address());
    return p;
  }
  size_t used = 0;
};

}  // namespace

extern "C" {
// For the debugger: the method pool's high-water mark.
volatile uint32_t detector_method_pool_used;
}

extern "C" const int DETECTOR_INPUT_SIZE = kSize;

// The Ethos-U backend's IO copy hook (strong override of the weak default).
// The input: RGB888 to int8 (x ^ 0x80), 16 bytes per Helium load / store,
// tail by predicate. Everything else (the two outputs) is a memcpy.
extern "C" void arm_ethos_io_memcpy(void* dst, const void* src, size_t size) {
  if (src == g_input_rgb && size == MODEL_DETECT_INPUT0_NUMEL) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint8x16_t offset = vdupq_n_u8(0x80);
    for (int32_t left = static_cast<int32_t>(size); left > 0; left -= 16, s += 16, d += 16) {
      mve_pred16_t pred = vctp8q(static_cast<uint32_t>(left));
      vstrbq_p_u8(d, veorq_u8(vldrbq_z_u8(s, pred), offset), pred);
    }
    return;
  }
  memcpy(dst, src, size);
}

// The driver's hooks around the NPU job: command stream start to interrupt.
extern "C" void ethosu_inference_begin(struct ethosu_driver* drv, void* user_arg) {
  (void)drv;
  (void)user_arg;
  g_npu_begin = cycles();
}

extern "C" void ethosu_inference_end(struct ethosu_driver* drv, void* user_arg) {
  (void)drv;
  (void)user_arg;
  g_npu_cycles += cycles() - g_npu_begin;
}

extern "C" int32_t detector_init(void) {
  executorch::runtime::runtime_init();
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  auto method_allocator = std::make_unique<TrackingAllocator>(kMethodPoolSize, g_method_pool);
  TrackingAllocator* tracking = method_allocator.get();
  g_module = new (g_module_storage)
      EmbeddedModule(model_pte, model_pte_size, std::make_unique<BufferDataLoader>(model_pte, model_pte_size),
                     std::move(method_allocator), std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));

  for (int i = 0; i < 4; ++i) {
    g_sizes[i] = kInputShape[i];
    g_dim_order[i] = static_cast<DimOrderType>(i);
  }
  g_input = new (g_input_storage) TensorImpl(ScalarType::Char, 4, g_sizes.data(), nullptr, g_dim_order.data());

  const int32_t err = static_cast<int32_t>(g_module->load_method(MODEL_DETECT_METHOD));
  detector_method_pool_used = tracking->used;
  return err;
}

extern "C" int32_t detector_run(const uint8_t* rgb, detections_t* out) {
  g_input_rgb = rgb;
  g_input->set_data(const_cast<uint8_t*>(rgb));
  std::vector<EValue> inputs{EValue(Tensor(g_input))};
  g_npu_cycles = 0;
  const uint32_t t0 = cycles();
  auto result = g_module->execute(MODEL_DETECT_METHOD, inputs);
  if (!result.ok()) return static_cast<int32_t>(result.error());
  if (result->size() != MODEL_DETECT_NUM_OUTPUTS) return -1;
  const uint32_t t1 = cycles();
  out->count = decode((*result)[0].toTensor().const_data_ptr<int8_t>(), (*result)[1].toTensor().const_data_ptr<int8_t>(),
                      out->det);
  const uint32_t t2 = cycles();
  out->decode_us = us(t2 - t1);
  out->total_us = us(t2 - t0);
  out->npu_us = us(g_npu_cycles);
  for (uint32_t i = out->count; i < DETECTOR_MAX_DETECTIONS; ++i) out->det[i] = detection_t{};
  return 0;
}
