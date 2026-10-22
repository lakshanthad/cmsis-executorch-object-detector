#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Import the program of `yolo export format=ethos` into the detection demo's AI layer, ai_layer_detect/.

    python detect/make_ai_layer.py <export dir or .pte>

The program is what the Ultralytics exporter (ultralytics PR 24171) writes
for the DevKit-E8's NPU, for example:

    yolo export model=yolo26n.pt format=ethos name=ethos-u85-256 imgsz=416 data=coco128.yaml

A YOLO26 detection export runs whole on the NPU: the one-to-one head cut
before its top-k, fixed logit and score ranges, an INT8 NHWC input
(1, S, S, 3) that the runtime does not plan, INT8 box (1, N, 4) and scores
(1, N, 80). detect/detector.cpp runs it and decodes the two outputs with the
scales of the export's metadata.yaml, which this script writes into
model_io.h.

Run it in the solution's .venv (it reads the program with ExecuTorch). The
program has to come from the ExecuTorch release of the csolution's
PyTorch::ExecuTorch pack. Then `python detect/make_test_image.py` writes the
test image at the program's input size.

Writes:
    ai_layer_detect/ai_layer.clayer.yml   runtime, Ethos-U backend and the CPU operators the program uses
    ai_layer_detect/model_pte.c / .h      the program as a C array
    ai_layer_detect/model_io.h            its shapes, IO quantization, planned memory and NPU scratch
    ai_layer_detect/model.pte             the program itself
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LAYER = ROOT / "ai_layer_detect"
TARGET = "ethos-u85-256"  # the DevKit-E8's NPU, the `name` of the export
DTYPES = {1: "int8", 6: "float32"}  # ExecuTorch ScalarType: Char, Float
sys.path.insert(0, str(ROOT))

import create_ai_layer  # noqa: E402


def scratch_size(pte: bytes) -> int:
    """The NPU scratch of the program's Ethos-U delegate: the scratch_size block of its VelaBinStream."""
    start = pte.find(b"vela_bin_stream\0")
    if start < 0 or pte.find(b"vela_bin_stream\0", start + 1) >= 0:
        sys.exit("expected exactly one Ethos-U delegate in the program")
    pos = start
    while True:
        name = pte[pos : pos + 16].split(b"\0")[0]
        size = struct.unpack_from("<I", pte, pos + 16)[0]
        if name == b"scratch_size":
            return struct.unpack_from("<I", pte, pos + 32)[0]
        if name == b"vela_end_stream":
            sys.exit("no scratch_size block in the Ethos-U delegate")
        pos += 32 + size
        pos += -pos % 16


def program_io(pte: bytes) -> dict:
    """Method name, IO shapes and dtypes, and planned memory of the program's single method."""
    from executorch.exir._serialize._program import deserialize_pte_binary

    program = deserialize_pte_binary(pte)
    program = getattr(program, "program", program)
    (plan,) = program.execution_plan
    (inp,) = (plan.values[i].val for i in plan.inputs)
    outs = [plan.values[i].val for i in plan.outputs]
    return {
        "method": plan.name,
        "input": [int(d) for d in inp.sizes],
        "input_dtype": DTYPES.get(inp.scalar_type),
        "outputs": [[int(d) for d in o.sizes] for o in outs],
        "output_dtypes": [DTYPES.get(o.scalar_type) for o in outs],
        "planned": [int(s) for s in plan.non_const_buffer_sizes[1:]],
    }


def quantization(io: dict, export_dir: Path) -> dict:
    """The IO scales from the export's metadata.yaml, checked against what detect/detector.cpp expects."""
    import yaml

    metadata = yaml.safe_load((export_dir / "metadata.yaml").read_text())
    target = metadata.get("args", {}).get("name")
    if target != TARGET:
        sys.exit(f"the program is for {target}, the DevKit-E8 needs name={TARGET}")
    q = metadata.get("quantization")
    size = io["input"][1]
    anchors = sum((size // s) ** 2 for s in (8, 16, 32))
    if (
        q is None
        or io["input_dtype"] != "int8"
        or io["output_dtypes"] != ["int8", "int8"]
        or io["input"] != [1, size, size, 3]
        or io["outputs"] != [[1, anchors, 4], [1, anchors, 80]]
        or q["inputs"][0]["zero_point"] != -128
        or abs(q["inputs"][0]["scale"] * 255 - 1) > 0.01
        or len(io["planned"]) != 1
    ):
        sys.exit(
            "expected a YOLO26 detection export with INT8 IO: an NHWC input (1, S, S, 3) with scale ~1/255 and zero "
            "point -128 (detect/detector.cpp converts RGB888 with x ^ 0x80), box (1, N, 4) and scores (1, N, 80), one "
            f"planned buffer; found {io}, {q}"
        )
    return q


def io_header(io: dict, q: dict, scratch: int) -> str:
    """model_io.h in create_ai_layer.py's format, under the names detect/detector.cpp uses."""
    import torch

    def desc(shape, tensor_q):
        return create_ai_layer._tensor_desc(shape, torch.int8, tensor_q["scale"], tensor_q["zero_point"], -128, 127)

    header = create_ai_layer.io_header(
        {
            "detect": {
                "inputs": [desc(io["input"], q["inputs"][0])],
                "outputs": [desc(s, tensor_q) for s, tensor_q in zip(io["outputs"], q["outputs"])],
            }
        }
    )
    lines = header.splitlines()
    lines[0] = "// Generated by detect/make_ai_layer.py -- do not edit."
    lines = [
        f'#define MODEL_DETECT_METHOD "{io["method"]}"  // the program\'s method, under detect/detector.cpp\'s names'
        if line.startswith("#define MODEL_DETECT_METHOD ")
        else line
        for line in lines
    ]
    lines.append(f"#define MODEL_DETECT_PLANNED_BYTES {io['planned'][0]}  // the planned buffer: the two outputs")
    lines.append(f"#define MODEL_DETECT_SCRATCH_BYTES {scratch}  // Vela's scratch for the delegate")
    lines.append("")
    return "\n".join(lines)


def clayer(runtime: list[str], operators: list[str], version: str, source: str) -> str:
    return "\n".join(
        [
            "# Generated by detect/make_ai_layer.py -- do not edit.",
            f"# From {source}, a program of `yolo export format=ethos`.",
            "layer:",
            "  type: AI",
            "  description: YOLO26n object detector (80 COCO classes) from `yolo export format=ethos` for the Ethos-U85 of the Ensemble E8",
            "",
            "  packs:",
            f"    - pack: {create_ai_layer.PACK}@{version}",
            "",
            "  define:",
            "    - ET_LOG_ENABLED: 0",
            "",
            "  add-path:",
            "    - .",
            "",
            "  components:",
            *[f"    - component: Machine Learning:ExecuTorch:{c}" for c in runtime],
            *[f"    - component: Machine Learning:ExecuTorch Operators:{c}" for c in operators],
            "",
            "  groups:",
            "    - group: ObjectDetector",
            "      files:",
            f"        - file: ./{create_ai_layer.SYMBOL}.c",
            f"        - file: ./{create_ai_layer.SYMBOL}.h",
            "        - file: ./model_io.h",
            "",
        ]
    )


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(__file__).name} <export dir or .pte>")
    source = Path(sys.argv[1]).resolve()
    pte_file = source if source.suffix == ".pte" else next(source.glob("*.pte"))
    pte = pte_file.read_bytes()
    io = program_io(pte)
    q = quantization(io, pte_file.parent)
    scratch = scratch_size(pte)

    version = create_ai_layer.executorch_version(ROOT / "cmsis-executorch.cbuild-mlops.yml")
    runtime, operators = create_ai_layer.components(pte, create_ai_layer.executorch_pack(version))

    LAYER.mkdir(exist_ok=True)
    (LAYER / "model.pte").write_bytes(pte)
    (LAYER / f"{create_ai_layer.SYMBOL}.c").write_text(create_ai_layer.c_array(pte), newline="\n")
    (LAYER / f"{create_ai_layer.SYMBOL}.h").write_text(create_ai_layer.HEADER, newline="\n")
    (LAYER / "model_io.h").write_text(io_header(io, q, scratch), newline="\n")
    (LAYER / "ai_layer.clayer.yml").write_text(clayer(runtime, operators, version, pte_file.name), newline="\n")

    print(f"[detect] {io['method']}: input {io['input']} outputs {io['outputs']}, INT8")
    print(f"[detect] planned {io['planned'][0]} bytes, NPU scratch {scratch} bytes")
    print(f"[detect] {len(pte)} byte program, CPU operators: {operators or 'none'}")
    print(f"[detect] wrote {LAYER}; now run detect/make_test_image.py")


if __name__ == "__main__":
    main()
