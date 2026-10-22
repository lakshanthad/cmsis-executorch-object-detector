# Object detection on the Alif DevKit-E8: YOLO26n on the Ethos-U85

The COCO-pretrained Ultralytics YOLO26n with all 80 classes runs on the
Ethos-U85-256 of the Alif Ensemble E8 DevKit (DK-E8), from the camera to the
panel. The model is exported for the NPU by Ultralytics
(`yolo export format=ethos`) with the whole network on the Ethos-U85, its two
attention blocks included, and runs with ExecuTorch next to the Cortex-M55 HP
core. Each object gets a box with its class and score in the Ultralytics
colours; above the picture are the frame rate and the NPU, detector and frame
times, below it how many of each class are in view. A YOLO26 detection model
trained on other classes runs the same way, with no change to the
application ([section 6](#6-export-another-model)).

|                      |                                                                 |
| -------------------- | --------------------------------------------------------------- |
| NPU                  | 17.0 ms per 416x416 frame (2.40 MB program in MRAM)             |
| Decode on the CPU    | 3.8 ms: the best of the 80 class scores of each of 3549 anchors |
| Loop with the camera | 25 fps, the camera's rate; 38.3 fps on the test image           |

One CMSIS solution, `cmsis-executorch.csolution.yml`, with one project and
one target-type, **DevKit-E8**:

| Path                   | Contents                                                                                                                   |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `detect/`              | the application (`detect.cproject.yml`), the scripts that import an exported program and make the test image               |
| `ai_layer_detect/`     | the AI layer: the program of `yolo export format=ethos` as a C array, its IO header and the ExecuTorch components it needs |
| `board/DevKit-E8/`     | the board layer: display, camera, console, Ethos-U85 and the MRAM layout                                                   |
| `src/`                 | `EmbeddedModule`, ExecuTorch's `Module` without the POSIX file loading                                                     |
| `tools/console_log.py` | prints the application's console through the J-Link                                                                        |
| `.alif/`               | the boot table configuration that Alif SETOOLS writes once                                                                 |

The board layer, the camera driver and the ExecuTorch glue come from the
traffic counter example
[cmsis-executorch-traffic-counter](https://github.com/MatthiasHertelArm/cmsis-executorch-traffic-counter),
which is built on the Arm example
[cmsis-executorch](https://github.com/Arm-Examples/cmsis-executorch).
[documentation/object-detection.md](documentation/object-detection.md)
describes the demo in detail.

## 1. Host tools

1. [VS Code](https://code.visualstudio.com/) with the extensions **Keil
   Studio Pack** (`Arm.keil-studio-pack`, CMSIS Solution 1.70.0 or newer) and
   **Python** (`ms-python.python`). Sign in with an Arm account when Keil
   Studio asks; the free Keil MDK Community license is enough. When the
   folder opens, accept the tools that the Arm Tools Environment Manager
   offers from `vcpkg-configuration.json` (CMSIS-Toolbox, Arm Compiler 6,
   CMake, Ninja). The demo builds with Arm Compiler 6 only.
2. **Alif SETOOLS** from the
   [Alif software and tools page](https://alifsemi.com/support/software-tools/ensemble/)
   (login required). Set its root directory, the one with `app-gen-toc` and
   `app-write-mram`, in your VS Code user settings:

   ```json
   "alif.setools.root": "/absolute/path/to/setools"
   ```

3. **SEGGER J-Link Software** V8.42 or later from
   [segger.com](https://www.segger.com/downloads/jlink/), for the board's
   on-board J-Link. The DevKit-E8 was tested with V9.74, which knows its
   `AE822FA0E5597LS0`.
4. Python `>=3.10,<3.15` for the model export, the test image and the
   console tool.

## 2. Board

- **PRG USB (J3)** to the host: power, the on-board J-Link and the Secure
  Enclave's UART.
- The MT9M114 camera module on the MIPI CSI-2 connector and the 480x800 MIPI
  DSI panel (ILI9806E).
- **SW4** on `SEUART` while SETOOLS runs. The application does not use a
  UART: its console is a log buffer in the DTCM that the debugger reads.

## 3. Project and build

In a clone of this repository:

```bash
./setup_venv.sh && .venv/bin/python -m pip install ultralytics
.venv/bin/python detect/make_test_image.py
```

`setup_venv.sh` creates `.venv` with the ExecuTorch release of the
`PyTorch::ExecuTorch` pack (1.5.1) and the Vela compiler; `setup_venv.bat`
does the same on Windows. `make_test_image.py` letterboxes the `bus.jpg` that
comes with the `ultralytics` package into `detect/test_image.c`, the image
the detector runs on when the camera is off, at the program's input size. It
also prints what Ultralytics `predict` finds in it with the one-to-one head
that the program runs (`nms=False`). The image is not ours to redistribute,
so it is not in the repository, and the build needs it. The AI layer
`ai_layer_detect/` is committed.

Open the folder in VS Code and accept the pack installation
(`PyTorch::ExecuTorch`, `AlifSemiconductor::Ensemble`, `ARM::SDS`,
`Keil::MDK-Middleware`). In the CMSIS view open **Manage Solution**, choose
the target-type **DevKit-E8** and click **Apply**, then **Build**. From the
command line:

```bash
cbuild cmsis-executorch.csolution.yml --active DevKit-E8 --packs
```

The image is `out/detect/DevKit-E8/Release/detect.axf`.

## 4. Prepare the board once

The Secure Enclave boots the HP core from a table of contents in MRAM, which
must point at the HP core's MRAM region.

1. SW4 on **SEUART**, PRG USB attached.
2. **Terminal > Run Task > Alif: Install M55_HP debug stubs (DevKit-E8,
   single core configuration)**. It installs `.alif/M55_HP_mram_cfg_e8.json`
   into SETOOLS, selects the E8 with its default device configuration and
   writes the table. Choose COM port discovery (`-d`) the first time. If
   `app-write-mram` gets no answer, press the board's reset button while it
   waits.

A board whose table already boots the HP core from 0x80200000 needs no
change. The task writes a stub over the start of the HP core's MRAM region,
so load the application again after it.

## 5. Run and debug

In the CMSIS view click **Run** or **Debug**: the J-Link loads the image into
MRAM and starts it. The panel shows the camera picture with the boxes after
a few seconds.

- **Console:** `.venv/bin/python tools/console_log.py out/detect/DevKit-E8/Release/detect.axf.map`
  prints the timing and the detections every 100 frames.
- **Status:** the global `detect_status` holds the frame counters, the time
  of each step in microseconds, the camera state and the latest detections.
- **Test image:** with the debugger stopped at `main`, set
  `detect_camera_enable` to 0 and continue: the demo runs on the test image.
  The board finds the bus and the four people that `predict` finds on the
  host (scores side by side in
  [documentation/object-detection.md](documentation/object-detection.md#on-the-board)).

## 6. Export another model

The committed AI layer is the program that

```bash
yolo export model=yolo26n.pt format=ethos name=ethos-u85-256 imgsz=416 data=coco128.yaml
```

writes to `yolo26n_ethos_model/`, imported by `detect/make_ai_layer.py`. Any
YOLO26 detection model runs the same way, including one trained on your own
classes. Export it in the venv with your dataset's YAML as `data=` (the
export calibrates on its images), import it, and make the test image from
the same weights and an image of your data:

```bash
yolo export model=best.pt format=ethos name=ethos-u85-256 imgsz=416 data=my_data.yaml
.venv/bin/python detect/make_ai_layer.py best_ethos_model
.venv/bin/python detect/make_test_image.py my_image.jpg --weights best.pt
```

`make_ai_layer.py` writes the class count and the class names of the
export's `metadata.yaml` into `ai_layer_detect/model_io.h`, and the
application labels the boxes with them. The panel's font has upper case
letters, digits and common punctuation only, so the names are shown in upper
case, an underscore as a space and any other character as `?`.
`make_test_image.py` refuses weights with another class count than the
program; without `--weights` it takes the COCO-pretrained `yolo26n.pt`.

The build checks that the program fits the board:

- **Classes:** up to 88 at `imgsz=416`. The two outputs, 3549 x (4 + classes)
  bytes, go in the 320 kB method pool (`kMethodPoolSize` in
  `detect/detector.cpp`, in SRAM0, which has about 280 kB to spare). A
  smaller `imgsz` has fewer anchors and leaves room for more classes.
- **Input size:** `imgsz` up to 416. Vela's scratch, 1223.7 KiB at 416, has
  to fit the 1280 KiB temp pool in SRAM1.
- **Model size:** about `yolo26n`. The program is in the MRAM, and the image
  ends about 500 kB below the top of the region the scatter file allows.

`format=ethos` is not in an Ultralytics release yet (ultralytics PR 24171);
until it is, install the pull request's branch into the venv:

```bash
.venv/bin/python -m pip install "ultralytics @ git+https://github.com/ultralytics/ultralytics@ethos-export"
```

The program has to come from the ExecuTorch release of the
`PyTorch::ExecuTorch` pack, 1.5.1, which the venv has; `name=ethos-u85-256`
is the E8's NPU, and `make_ai_layer.py` refuses a program for another. The
export calibrates on the images in a random order, so a new export is
equivalent to the committed program, not identical to it.

## 7. If something does not work

- **The application never gets past the Secure Enclave handshake** (the log
  shows no frames, `SRAM1: power request failed`): the Secure Enclave has
  stopped answering. A reset does not help; power-cycle the board (unplug PRG
  USB), wait about 40 s, and load again.
- **The build fails in CMake** after a CMSIS Solution extension update
  (`toolchain.cmake ... could not find requested file`): delete
  `tmp/DevKit-E8` and build again.
- **CMSIS Load or Debug says there is no CMSIS Load task** (no
  `.vscode/tasks.json`): open **Manage Solution** in the CMSIS view and
  confirm the J-Link debugger of the target-type, or run **CMSIS: Update
  Debug Tasks and Launch Configurations**. The extension writes
  `.vscode/tasks.json` and `.vscode/launch.json` for the active target-type.
- **The debugger cannot start its GDB server:** another debug session holds
  the port. Stop it, or change the ports in the launch configuration.
- **The first inference after a debugger reset reports error 35:** the NPU
  was still busy with the job from before the reset. The loop recovers on its
  own.
- **The picture never changes, or the board and its debug port die while the
  camera runs:** the table of contents carries another device's
  configuration. Repeat step 4. Keep the camera frames and the panel buffers
  in SRAM0: neither the camera DMA nor the display controller reaches SRAM1,
  and SRAM8 hangs the bus.

## Licences

The code is under the Apache License 2.0 ([LICENSE](LICENSE)). The YOLO26n
weights come from Ultralytics and are under the AGPL-3.0, and so is the
program compiled into `ai_layer_detect/`; the export and
`detect/make_test_image.py` fetch the COCO-pretrained `yolo26n.pt` with the
`ultralytics` package. `src/LICENSE-ExecuTorch` covers the ExecuTorch sources
(BSD-3-Clause).

## Read on

- [documentation/object-detection.md](documentation/object-detection.md):
  the demo in detail (model, quantization of the class scores, application,
  memory, performance).
- [documentation/pack-provenance.md](documentation/pack-provenance.md): where
  the `PyTorch::ExecuTorch` pack comes from and how to move to a newer
  ExecuTorch release.
