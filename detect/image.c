/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Pixel work of the detection demo on the CPU, see image.h: the camera frame
 * to the model input, the input to the panel, rectangles and text, and boxes
 * in the Ultralytics colours.
 *
 * RGB565 as the CPI stores an MT9M114's frames: one halfword per pixel, B in
 * bits 4:0, G in 10:5, R in 15:11. The model input is RGB888, three bytes
 * per pixel, R first. The panel frame is the CDC200's RGB888, one 24-bit
 * word per pixel with R in bits 23:16 and B in 7:0, so in memory B first:
 * image_input_to_view() and image_fill() write B, G, R.
 */

#include <stdint.h>

#include "image.h"

static inline void rgb565_to_888(uint16_t p, uint8_t *d)
{
    const uint32_t r = (p >> 11) & 0x1FU, g = (p >> 5) & 0x3FU, b = p & 0x1FU;
    d[0] = (uint8_t)((r << 3) | (r >> 2));
    d[1] = (uint8_t)((g << 2) | (g >> 4));
    d[2] = (uint8_t)((b << 3) | (b >> 2));
}

/* Where output pixel (x, y) of a size x size picture turned `quarter_turns`
   times counter-clockwise lands: the index into the turned picture. */
static inline int turned_index(int x, int y, int size, int quarter_turns)
{
    switch (quarter_turns & 3) {
    case 1:  return (size - 1 - x) * size + y;
    case 2:  return (size - 1 - y) * size + (size - 1 - x);
    case 3:  return x * size + (size - 1 - y);
    default: return y * size + x;
    }
}

void image_rgb565_to_input(const uint16_t *camera, int w, int h, uint8_t *rgb, int size, int quarter_turns)
{
    const int side = w < h ? w : h;
    const int x0 = (w - side) / 2, y0 = (h - side) / 2;
    /* Source column of each output column, and one destination step per row
       (as in image_bayer_to_input): no division or index per pixel. */
    static uint16_t col[1024];
    for (int x = 0; x < size; x++) {
        col[x] = (uint16_t)(x0 + x * side / size);
    }
    for (int y = 0; y < size; y++) {
        const uint16_t *row = camera + (y0 + y * side / size) * w;
        uint8_t *d = rgb + turned_index(0, y, size, quarter_turns) * 3;
        const int dstep = (quarter_turns & 3) == 0 ? 3 : (turned_index(1, y, size, quarter_turns) - turned_index(0, y, size, quarter_turns)) * 3;
        for (int x = 0; x < size; x++, d += dstep) {
            rgb565_to_888(row[col[x]], d);
        }
    }
}

void image_input_to_view(const uint8_t *rgb, int size, uint8_t *panel)
{
    /* 1:1, centred in the view, no scaling. The margins stay as they are. */
    const int x0 = (IMAGE_VIEW - size) / 2, y0 = IMAGE_VIEW_TOP + (IMAGE_VIEW - size) / 2;
    for (int y = 0; y < size; y++) {
        const uint8_t *s = rgb + y * size * 3;
        uint8_t *d = panel + ((y0 + y) * IMAGE_PANEL_W + x0) * 3;
        for (int x = 0; x < size; x++, s += 3, d += 3) {
            d[0] = s[2]; /* R, G, B to the panel's B, G, R */
            d[1] = s[1];
            d[2] = s[0];
        }
    }
}

uint32_t image_class_colour(int cls)
{
    /* The Ultralytics palette (ultralytics.utils.plotting.Colors), by class index. */
    static const uint32_t colours[20] = {
        0x042AFFU, 0x0BDBEBU, 0xF3F3F3U, 0x00DFB7U, 0x111F68U, 0xFF6FDDU, 0xFF444FU, 0xCCED00U, 0x00F344U, 0xBD00FFU,
        0x00B4FFU, 0xDD00BAU, 0x00FFFFU, 0x26C000U, 0x01FFB3U, 0x7D24FFU, 0x7B0068U, 0xFF1B6CU, 0xFC6D2FU, 0xA2FF0BU,
    };
    return colours[(cls < 0 ? 0 : cls) % 20];
}

void image_fill(uint8_t *panel, int x0, int y0, int x1, int y1, uint32_t rgb)
{
    const uint8_t r = (uint8_t)(rgb >> 16), g = (uint8_t)(rgb >> 8), b = (uint8_t)rgb;
    x0 = x0 < 0 ? 0 : x0;
    y0 = y0 < 0 ? 0 : y0;
    x1 = x1 > IMAGE_PANEL_W ? IMAGE_PANEL_W : x1;
    y1 = y1 > IMAGE_PANEL_H ? IMAGE_PANEL_H : y1;
    for (int y = y0; y < y1; y++) {
        uint8_t *p = panel + (y * IMAGE_PANEL_W + x0) * 3;
        for (int x = x0; x < x1; x++, p += 3) {
            p[0] = b; /* the panel's byte order */
            p[1] = g;
            p[2] = r;
        }
    }
}

void image_box(uint8_t *panel, int x0, int y0, int x1, int y1, int t, uint32_t rgb)
{
    image_fill(panel, x0, y0, x1, y0 + t, rgb);
    image_fill(panel, x0, y1 - t, x1, y1, rgb);
    image_fill(panel, x0, y0, x0 + t, y1, rgb);
    image_fill(panel, x1 - t, y0, x1, y1, rgb);
}

/* The classic 5x7 font, ASCII 0x20 to 0x5A: five columns per glyph, bit 0 at the top. */
static const uint8_t font5x7[][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x5F, 0x00, 0x00}, {0x00, 0x07, 0x00, 0x07, 0x00},
    {0x14, 0x7F, 0x14, 0x7F, 0x14}, {0x24, 0x2A, 0x7F, 0x2A, 0x12}, {0x23, 0x13, 0x08, 0x64, 0x62},
    {0x36, 0x49, 0x55, 0x22, 0x50}, {0x00, 0x05, 0x03, 0x00, 0x00}, {0x00, 0x1C, 0x22, 0x41, 0x00},
    {0x00, 0x41, 0x22, 0x1C, 0x00}, {0x08, 0x2A, 0x1C, 0x2A, 0x08}, {0x08, 0x08, 0x3E, 0x08, 0x08},
    {0x00, 0x50, 0x30, 0x00, 0x00}, {0x08, 0x08, 0x08, 0x08, 0x08}, {0x00, 0x60, 0x60, 0x00, 0x00},
    {0x20, 0x10, 0x08, 0x04, 0x02}, {0x3E, 0x51, 0x49, 0x45, 0x3E}, {0x00, 0x42, 0x7F, 0x40, 0x00},
    {0x42, 0x61, 0x51, 0x49, 0x46}, {0x21, 0x41, 0x45, 0x4B, 0x31}, {0x18, 0x14, 0x12, 0x7F, 0x10},
    {0x27, 0x45, 0x45, 0x45, 0x39}, {0x3C, 0x4A, 0x49, 0x49, 0x30}, {0x01, 0x71, 0x09, 0x05, 0x03},
    {0x36, 0x49, 0x49, 0x49, 0x36}, {0x06, 0x49, 0x49, 0x29, 0x1E}, {0x00, 0x36, 0x36, 0x00, 0x00},
    {0x00, 0x56, 0x36, 0x00, 0x00}, {0x00, 0x08, 0x14, 0x22, 0x41}, {0x14, 0x14, 0x14, 0x14, 0x14},
    {0x41, 0x22, 0x14, 0x08, 0x00}, {0x02, 0x01, 0x51, 0x09, 0x06}, {0x32, 0x49, 0x79, 0x41, 0x3E},
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, {0x7F, 0x49, 0x49, 0x49, 0x36}, {0x3E, 0x41, 0x41, 0x41, 0x22},
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, {0x7F, 0x49, 0x49, 0x49, 0x41}, {0x7F, 0x09, 0x09, 0x01, 0x01},
    {0x3E, 0x41, 0x41, 0x51, 0x32}, {0x7F, 0x08, 0x08, 0x08, 0x7F}, {0x00, 0x41, 0x7F, 0x41, 0x00},
    {0x20, 0x40, 0x41, 0x3F, 0x01}, {0x7F, 0x08, 0x14, 0x22, 0x41}, {0x7F, 0x40, 0x40, 0x40, 0x40},
    {0x7F, 0x02, 0x04, 0x02, 0x7F}, {0x7F, 0x04, 0x08, 0x10, 0x7F}, {0x3E, 0x41, 0x41, 0x41, 0x3E},
    {0x7F, 0x09, 0x09, 0x09, 0x06}, {0x3E, 0x41, 0x51, 0x21, 0x5E}, {0x7F, 0x09, 0x19, 0x29, 0x46},
    {0x46, 0x49, 0x49, 0x49, 0x31}, {0x01, 0x01, 0x7F, 0x01, 0x01}, {0x3F, 0x40, 0x40, 0x40, 0x3F},
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, {0x7F, 0x20, 0x18, 0x20, 0x7F}, {0x63, 0x14, 0x08, 0x14, 0x63},
    {0x03, 0x04, 0x78, 0x04, 0x03}, {0x61, 0x51, 0x49, 0x45, 0x43},
};

int image_text(uint8_t *panel, int x, int y, int scale, const char *text, uint32_t rgb)
{
    for (; *text != '\0'; text++) {
        int c = (unsigned char)*text;
        if (c >= 'a' && c <= 'z') {
            c -= 'a' - 'A';
        }
        if (c < 0x20 || c > 0x5A) {
            c = '?';
        }
        const uint8_t *glyph = font5x7[c - 0x20];
        for (int col = 0; col < 5; col++) {
            for (int row = 0; row < 7; row++) {
                if (glyph[col] & (1U << row)) {
                    image_fill(panel, x + col * scale, y + row * scale, x + (col + 1) * scale, y + (row + 1) * scale, rgb);
                }
            }
        }
        x += 6 * scale;
    }
    return x;
}
