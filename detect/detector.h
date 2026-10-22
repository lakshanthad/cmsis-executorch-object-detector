/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The object detector of the detection demo: YOLO26n with the 80 COCO
 * classes on the Ethos-U85 (the program of `yolo export format=ethos` in
 * ai_layer_detect/) and the rest of its head on the CPU: the best class of
 * each anchor, the threshold, the boxes.
 */
#ifndef DETECT_DETECTOR_H_
#define DETECT_DETECTOR_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DETECTOR_MAX_DETECTIONS 32
#define DETECTOR_CLASSES 80  /* COCO */
#define DETECTOR_SCORE_THRESHOLD 0.25f  /* the predict default of `yolo detect predict` (conf=0.25) */

typedef struct {
    float   x1, y1, x2, y2;  /* input pixels (DETECTOR_INPUT_SIZE square) */
    float   score;
    int32_t cls;             /* 0 .. DETECTOR_CLASSES - 1, see DETECTOR_CLASS_NAMES */
} detection_t;

/* The result of one frame, strongest first. */
typedef struct {
    uint32_t    frame;     /* frame counter of the application */
    uint32_t    count;     /* valid entries in det[] */
    uint32_t    npu_us;    /* NPU job, command stream start to interrupt */
    uint32_t    total_us;  /* module.execute() and the decode */
    uint32_t    decode_us; /* the decode alone */
    detection_t det[DETECTOR_MAX_DETECTIONS];
} detections_t;

/* Input size in pixels: the model takes DETECTOR_INPUT_SIZE x DETECTOR_INPUT_SIZE RGB888. */
extern const int DETECTOR_INPUT_SIZE;

/* Upper-case COCO class names (the panel's font has no lower case), in the
   order of the model's class channels. */
extern const char *const DETECTOR_CLASS_NAMES[DETECTOR_CLASSES];

/* Load the program and the method. 0 on success. */
int32_t detector_init(void);

/* Run the detector on an interleaved RGB888 image of the input size (in the
   bulk SRAM or the MRAM); fills `out` (frame is left alone). 0 on success,
   else the ExecuTorch error. */
int32_t detector_run(const uint8_t *rgb, detections_t *out);

#ifdef __cplusplus
}
#endif

#endif /* DETECT_DETECTOR_H_ */
