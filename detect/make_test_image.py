#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Write the test image of the detection demo as a C array: detect/test_image.c.

    python detect/make_test_image.py [<image file or URL>] [--weights <model.pt>]

The image is letterboxed to the model input (the Ultralytics grey padding)
and stored as interleaved RGB888, what the camera path hands over. The input
size is the program's, from ai_layer_detect/model_io.h. The default is the
bus.jpg that comes with the Ultralytics package (people and a bus). The
generated file is not committed (.gitignore); run this once before the first
build, and again after detect/make_ai_layer.py imported a program of another
input size (the build stops otherwise).

It also prints what Ultralytics `predict` finds in the letterboxed image with
the one-to-one head that the program runs (nms=False) and the board's score
threshold: the float reference for the board. That reference is only right
for the weights the program was exported from: --weights names them, and
weights with other classes than the program are refused. Without --weights
it is the COCO-pretrained yolo26n.pt at the repository root, which
Ultralytics downloads on the first run (and `yolo export model=yolo26n.pt`
run there uses too); when the program has other classes, the image is
written without the reference. For a model trained on other classes, pass
an image of its own data.
"""

from __future__ import annotations

import argparse
import re
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEIGHTS = ROOT / "yolo26n.pt"


def fetch(source: str) -> bytes:
    if "://" not in source:
        return Path(source).read_bytes()
    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(source, context=context) as response:
        return response.read()


def letterbox(bgr: np.ndarray, size: int) -> np.ndarray:
    """Resize the longer side to `size` and pad to a square with the Ultralytics grey: RGB uint8."""
    import cv2

    h, w = bgr.shape[:2]
    r = size / max(h, w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py : py + nh, px : px + nw] = resized[:, :, ::-1]
    return out


def define(path: Path, pattern: str) -> str:
    """The value of a #define in a header, by a regex for the rest of its line."""
    found = re.search(rf"^#define {pattern}", path.read_text(), re.MULTILINE)
    if not found:
        sys.exit(f"{path}: no #define {pattern}")
    return found.group(1)


def program_names(model_io: Path) -> list[str]:
    """The class names of the program, as detect/make_ai_layer.py wrote them into model_io.h."""
    found = re.search(r"^#define MODEL_DETECT_CLASS_NAMES \{(.*?)^\}", model_io.read_text(), re.MULTILINE | re.DOTALL)
    if not found:
        sys.exit(f"{model_io}: no MODEL_DETECT_CLASS_NAMES; import the program again with detect/make_ai_layer.py")
    return [re.sub(r"\\(.)", r"\1", s) for s in re.findall(r'"((?:[^"\\]|\\.)*)"', found.group(1))]


def main() -> None:
    import cv2
    from make_ai_layer import panel_label
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", help="image file or URL (default: the bus.jpg of the ultralytics package)")
    parser.add_argument(
        "--weights", type=Path, help="the weights the program was exported from, for the reference (default: yolo26n.pt)"
    )
    args = parser.parse_args()

    model_io = ROOT / "ai_layer_detect/model_io.h"
    size = int(define(model_io, r"MODEL_DETECT_INPUT0_SHAPE \{1, (\d+), \d+, 3\}"))
    names = program_names(model_io)
    threshold = float(define(HERE / "detector.h", r"DETECTOR_SCORE_THRESHOLD ([0-9.]+)f"))
    source = args.image or str(ASSETS / "bus.jpg")
    bgr = cv2.imdecode(np.frombuffer(fetch(source), np.uint8), cv2.IMREAD_COLOR)
    rgb = letterbox(bgr, size)

    weights = args.weights or WEIGHTS
    model = YOLO(str(weights))  # Ultralytics downloads the COCO-pretrained yolo26n.pt
    if [panel_label(model.names[i]) for i in range(len(model.names))] != names:
        if args.weights:
            sys.exit(f"{weights} has other classes than the program in ai_layer_detect/: pass the weights it was exported from")
        print(f"no reference on the host: the program has other classes than {weights.name}; --weights gives one")
        reference_note = "// No reference on the host: the weights the program was exported from were not given (--weights).\n"
    else:
        # The square letterboxed image is already the input size, so predict adds no padding of its own.
        result = model.predict(np.ascontiguousarray(rgb[..., ::-1]), imgsz=size, conf=threshold, nms=False, verbose=False)[0]
        reference = [
            f"{result.names[int(c)]} {s:.2f} x1 {x1:.0f} y1 {y1:.0f} x2 {x2:.0f} y2 {y2:.0f}"
            for (x1, y1, x2, y2), s, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())
        ]
        print(f"predict on the host (nms=False, score > {threshold:.2f}):", *reference, sep="\n  ")
        comment = "\n".join(f"//   {r}" for r in reference)
        reference_note = f"// Ultralytics predict on the host finds (nms=False, score > {threshold:.2f}):\n{comment}\n"

    data = rgb.tobytes()
    rows = [", ".join(f"0x{b:02x}" for b in data[i : i + 24]) for i in range(0, len(data), 24)]
    (HERE / "test_image.c").write_text(
        "// Generated by detect/make_test_image.py -- do not edit, do not commit.\n"
        f"// {Path(source).name}, letterboxed to {size} x {size}, RGB888.\n"
        f"{reference_note}"
        '#include "test_image.h"\n'
        f"__attribute__((aligned(16))) const unsigned char test_image[{len(data)}] = {{\n  "
        + ",\n  ".join(rows)
        + "\n};\n",
        newline="\n",
    )
    (HERE / "test_image.h").write_text(
        "// Generated by detect/make_test_image.py -- do not edit, do not commit.\n"
        "#pragma once\n"
        f"#define TEST_IMAGE_SIZE {size}\n"
        "#ifdef __cplusplus\n"
        'extern "C" {\n'
        "#endif\n"
        "extern const unsigned char test_image[];  // TEST_IMAGE_SIZE x TEST_IMAGE_SIZE x 3, RGB888\n"
        "#ifdef __cplusplus\n"
        "}\n"
        "#endif\n",
        newline="\n",
    )
    print(f"wrote {HERE / 'test_image.c'} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
