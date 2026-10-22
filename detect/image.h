/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Pixel work of the detection demo on the CPU: the camera frame to the model
 * input, the picture to the panel, boxes and text.
 *
 * The panel frame is APP_DISPLAY_WIDTH x APP_DISPLAY_HEIGHT RGB888 (480 x 800,
 * portrait), in the CDC200's byte order B, G, R; colours given as uint32_t are
 * 0xRRGGBB. The picture goes in a square view of the panel width, 480 x 480,
 * centred vertically; above and below it are the status lines.
 */
#ifndef DETECT_IMAGE_H_
#define DETECT_IMAGE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IMAGE_PANEL_W   APP_DISPLAY_WIDTH
#define IMAGE_PANEL_H   APP_DISPLAY_HEIGHT
#define IMAGE_VIEW      APP_DISPLAY_WIDTH                     /* the square picture on the panel */
#define IMAGE_VIEW_TOP  ((APP_DISPLAY_HEIGHT - APP_DISPLAY_WIDTH) / 2)

/* The centre square of an RGB565 frame (w x h, h <= w), scaled to the model
   input (RGB888, size x size). quarter_turns turns the picture that many
   times 90 degrees counter-clockwise on the way. */
void image_rgb565_to_input(const uint16_t *camera, int w, int h, uint8_t *rgb, int size, int quarter_turns);

/* A model input (RGB888, size x size), 1:1 in the middle of the view of the panel. */
void image_input_to_view(const uint8_t *rgb, int size, uint8_t *panel);

/* The colour of a class (0xRRGGBB): the Ultralytics palette. */
uint32_t image_class_colour(int cls);

/* Filled rectangle on the panel, clipped: [x0, x1) x [y0, y1). */
void image_fill(uint8_t *panel, int x0, int y0, int x1, int y1, uint32_t rgb);

/* Rectangle outline `t` pixels thick, inside [x0, x1) x [y0, y1). */
void image_box(uint8_t *panel, int x0, int y0, int x1, int y1, int t, uint32_t rgb);

/* Text in a 5x7 font scaled by `scale` (upper case, digits, " .:%/-+"). Returns the x after it. */
int image_text(uint8_t *panel, int x, int y, int scale, const char *text, uint32_t rgb);

#ifdef __cplusplus
}
#endif

#endif /* DETECT_IMAGE_H_ */
