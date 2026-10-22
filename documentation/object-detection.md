# Object detection demo with YOLO26n on the Ethos-U85 (DevKit-E8)

The COCO-pretrained Ultralytics YOLO26n with all 80 classes runs on the
Ethos-U85-256 of the Alif Ensemble E8 DevKit, and the panel shows the camera
picture with a box, the class and the score for each object, and the frame
rate. A YOLO26 detection model trained on other classes runs the same way
([Models trained on other classes](#models-trained-on-other-classes)). The
project is `detect/detect.cproject.yml`, target-type `DevKit-E8`. Build and
run are in the [README](../README.md).

## The model

The Ultralytics exporter (`format=ethos`, ultralytics PR 24171) exports the
COCO-pretrained `yolo26n.pt` for the DevKit-E8's NPU:

```bash
yolo export model=yolo26n.pt format=ethos name=ethos-u85-256 imgsz=416 data=coco128.yaml
```

It writes `yolo26n_ethos_model/`: the program, `yolo26n.pte`, and
`metadata.yaml` with the class names, the input layout and the scales of the
INT8 input and outputs. For YOLO26 detection the exporter cuts the NMS-free
one-to-one head before its final top-k, which TOSA does not have, and
quantizes the input and the outputs, so the program is:

```text
image  int8 (1, 416, 416, 3)  RGB in [0, 1], interleaved
-> box int8 (1, 3549, 4)      left, top, right, bottom distances from each anchor, in anchor strides
-> cls int8 (1, 3549, 80)     sigmoid score per COCO class
```

The Ethos-U85 has a matrix multiply, so the two attention blocks stay on the
NPU and the whole graph is one Ethos-U delegate in the method `forward`; the
two `getitem` nodes that the export log counts on the CPU only pick the two
outputs from the delegate's results. Vela compiles it with the Ethos-U85
configuration of the ExecuTorch Arm backend (`Ethos_U85_SYS_DRAM_Mid`,
memory mode `Shared_Sram`), not with the `ensemble_vela.ini` of Alif's
Ensemble pack: the weights and the command stream stay in the program, in the
MRAM, and the 1223.7 KiB of scratch goes to the SRAM. The
program is 2.40 MB. The calibration images are the 128 of COCO128
(`data=coco128.yaml`; without it the exporter takes the 8 of COCO8), in a
random order, so two exports are equivalent but not byte-identical.

[`detect/make_ai_layer.py`](../detect/make_ai_layer.py) turns the export
into the AI layer `ai_layer_detect/`: the program as a C array, the
ExecuTorch components it needs (the runtime and the Ethos-U backend, no CPU
operators), and `model_io.h` with the shapes, the INT8 scales of
`metadata.yaml`, the planned buffer and Vela's scratch, which
`detect/detector.cpp` checks against its memory pools when it compiles, and
the model name and class names of `metadata.yaml`. It refuses a program for
another NPU than `ethos-u85-256` or with another interface. The program has
to come from the ExecuTorch release of the `PyTorch::ExecuTorch` pack, 1.5.1.

## Models trained on other classes

Nothing in the application is specific to COCO. `make_ai_layer.py` writes
the class names of the export's `metadata.yaml` into `model_io.h` as
`MODEL_DETECT_CLASSES` and `MODEL_DETECT_CLASS_NAMES`, in upper case with an
underscore as a space: the panel's 5x7 font has no lower case, and it shows
a character outside its range as `?` (the script warns about such a name).
`detect/class_names.c` takes the names from there, the decode takes the
class count from the program's output shape, and `detector.cpp` checks that
the two agree when it compiles. `MODEL_DETECT_NAME`, the model from the
description in `metadata.yaml`, goes into the title. The colours repeat the
20 of the Ultralytics palette.

The class count sets the size of the class scores, 3549 x C bytes at
416 x 416, and the time of the decode, which reads every score: about
0.05 ms per class. The two outputs fit the method pool up to 88 classes; the
[README](../README.md#6-export-another-model) lists the limits.

As a check, `yolo26n` cut down to its first six COCO classes, with class 5
renamed `city_bus`, was exported, imported and run on the board with no
change to the application. On the test image it finds the bus as
`CITY BUS` (0.95) and the four people (0.91, 0.88, 0.87 and 0.29) with the
boxes of the 80-class program within 2 pixels, at 46.2 fps, the decode down
from 3.8 ms to 0.28 ms.

## Quantization of the class scores

With the default calibration the int8 model scored every object at 0.50 at
most, which is sigmoid(0). The Ethos-U quantizer observes activations with a
histogram observer, which fits the range to where most values are. Nearly all
of the 80 class logits of an anchor are far below 0, so:

- the logits (the three class convs, through their permute, flatten and the
  cat into the sigmoid, one shared range) got a range that ended near 0;
- the scores out of the sigmoid got a range that ended at 0.35.

The exporter gives both fixed ranges instead (`_fix_score_ranges` in
`ultralytics/utils/export/ethos.py`; the export log says
`fixed INT8 ranges: class logits (-8.0, 8.0), scores (0.0, 1.0)`):

| Tensor | Range   | Scale | Zero point |
| ------ | ------- | ----- | ---------- |
| logits | -8 to 8 | 1/16  | 0          |
| scores | 0 to 1  | 1/256 | -128       |

A logit below -8 is a score below 0.0003, so nothing of use is clipped. The
box distances keep their calibrated range (scale 0.0509, zero point -115).
The int8 model on the host then scores the objects of the test image close
to the board (see [On the board](#on-the-board)).

## The application

[`detect/app_detect.cpp`](../detect/app_detect.cpp) runs three threads:

- the camera thread turns each MT9M114 frame (640x480 RGB565) into the
  416x416 RGB888 input, its centre square scaled down, in one of two slots;
- the vision thread runs the detector on the newest slot and copies the
  picture into the panel's back buffer;
- the display thread draws the boxes, labels and status lines while the NPU
  works on the next frame, cleans the D-cache and switches the panel's buffer.

[`detect/detector.cpp`](../detect/detector.cpp) runs the program: the input
copy hook turns RGB888 into int8 on the way into the NPU's scratch, and the
CPU takes the best class of each anchor, keeps it when its score is above
0.25 (the `conf` default of Ultralytics' predict), turns the distances into
boxes and drops a box that overlaps a stronger one of the same class by more
than 0.7 IoU. Overlaps of different classes stay: a person on a bicycle or a
cup on a table is two objects. [`detect/image.c`](../detect/image.c) does the
pixel work (camera frame to input, input to panel, rectangles and the 5x7
font), with the Ultralytics palette for the class colours. The camera driver
([`detect/camera.c`](../detect/camera.c)) takes one snapshot per frame from
the MT9M114 through the CPI, and
[`detect/ethosu_rtos.c`](../detect/ethosu_rtos.c) lets the thread that runs
the model block on an RTX semaphore during the NPU job instead of spinning,
so the display thread runs meanwhile.

## Memory

| Region                | Size used       | Contents                                                                                                                                        |
| --------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| MRAM, from 0x80200000 | 3.09 MB         | code 0.68 MB (0.52 MB of it the test image), the program 2.40 MB                                                                                |
| SRAM0                 | 3.91 of 4.19 MB | the two panel buffers (2.30 MB), the two camera frames (1.23 MB), the method pool (320 kB, 298 kB used: the planned outputs), the thread stacks |
| SRAM1                 | 2.35 of 4.19 MB | the NPU scratch (1.31 MB), the two input slots (1.04 MB)                                                                                        |

The image ends about 500 kB below the top of the MRAM user region that the
scatter file allows.

## Performance on the board

Measured on the DevKit-E8, M55 HP core at 400 MHz:

| Step                                                | Time                         |
| --------------------------------------------------- | ---------------------------- |
| NPU job                                             | 17.0 ms                      |
| Detector (input copy, NPU job, output copy, decode) | 22.5 ms                      |
| Decode on the CPU                                   | 3.8 ms                       |
| Picture into the panel buffer                       | 3.6 ms                       |
| Overlays and cache clean (display thread)           | 2.5 to 2.9 ms                |
| Camera frame to model input (camera thread)         | 7.4 ms                       |
| Loop with the camera                                | 25 fps, the camera's rate    |
| Loop on the test image                              | 38.3 fps (26.1 ms per frame) |

The decode is plain C over 284,000 scores per frame; with the camera the
loop waits for frames anyway, so it is left that way.

## On the board

The test image (`detect/make_test_image.py`, the `bus.jpg` of the
`ultralytics` package) with the camera off (`detect_camera_enable` set to 0
at `main`), the same objects in each column. The host int8 column is the
exporter's quantized graph run in PyTorch before Vela compiles it; the float
column is Ultralytics `predict` with `nms=False`, which `make_test_image.py`
prints:

| Object                   | Board | Host, int8 | Host, float |
| ------------------------ | ----- | ---------- | ----------- |
| bus                      | 0.93  | 0.88       | 0.88        |
| person, left             | 0.92  | 0.91       | 0.88        |
| person, right            | 0.88  | 0.87       | 0.72        |
| person, middle           | 0.80  | 0.87       | 0.91        |
| person, at the left edge | 0.32  | 0.31       | 0.50        |

The boxes of the board and the host int8 model agree within 2 pixels. The
float model also gives the right-hand person a second box (0.58) that
`predict` keeps and the board's duplicate check drops; the host int8 model
has the same duplicate at 0.38.

## Status

Built and run on the DevKit-E8 with the program in `ai_layer_detect/`. The
test image gives the objects above, and with the camera the loop runs at
25 fps on a live picture. Boxes from the camera and the panel layout are yet
to be checked by eye.
