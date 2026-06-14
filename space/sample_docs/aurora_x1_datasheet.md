# Aurora X1 Edge AI Accelerator — Technical Datasheet (Sample)

> Synthetic document for demonstrating retrieval over technical specifications.

## Overview

The Aurora X1 is a low-power inference accelerator for edge devices. It targets
computer-vision and speech workloads at the network edge, delivering high
throughput within a strict 15 W thermal envelope.

## Key Specifications

| Parameter | Value |
|-----------|-------|
| Peak INT8 throughput | 64 TOPS |
| Peak FP16 throughput | 16 TFLOPS |
| On-chip SRAM | 32 MB |
| External memory | LPDDR5, up to 16 GB |
| Memory bandwidth | 102 GB/s |
| Typical power | 15 W |
| Idle power | 0.9 W |
| Process node | 5 nm |
| Operating temperature | -40°C to 85°C |
| Host interface | PCIe Gen4 x4 |

## Performance Benchmarks

Measured throughput on common models at INT8 precision, batch size 1:

| Model | Throughput | Latency |
|-------|------------|---------|
| ResNet-50 | 3,200 img/s | 0.31 ms |
| YOLOv8-n | 1,850 img/s | 0.54 ms |
| BERT-base | 940 seq/s | 1.06 ms |
| Whisper-tiny | 22x real-time | — |

## Software Support

The Aurora X1 ships with the AuroraRT runtime, which supports models exported
from PyTorch and TensorFlow via ONNX. Quantization-aware training and
post-training INT8 quantization are both supported. The SDK includes a model
compiler that performs operator fusion and memory planning for the on-chip SRAM.

## Power Modes

Three power modes are available:

- **Burst** — up to 18 W for short windows, 64 TOPS peak.
- **Balanced** — 15 W sustained, the default mode, ~58 TOPS effective.
- **Eco** — capped at 8 W, ~34 TOPS, for battery-powered deployments.

## Ordering Information

| Part number | Memory | Temperature grade |
|-------------|--------|-------------------|
| AX1-8C | 8 GB | Commercial (0–70°C) |
| AX1-16I | 16 GB | Industrial (-40–85°C) |
