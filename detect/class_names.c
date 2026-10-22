/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The class names of the detector (detector.cpp): the names of the export's
 * metadata.yaml, which detect/make_ai_layer.py writes into
 * ai_layer_detect/model_io.h, upper case for the panel's font.
 */

#include "detector.h"

const char *const DETECTOR_CLASS_NAMES[DETECTOR_CLASSES] = MODEL_DETECT_CLASS_NAMES;

/* A short list would leave NULL names, which the labels would print: count
   the list itself, since the array above takes its size from the class count. */
_Static_assert(sizeof((const char *[])MODEL_DETECT_CLASS_NAMES) / sizeof(const char *) == DETECTOR_CLASSES,
               "MODEL_DETECT_CLASS_NAMES has one name per class");
