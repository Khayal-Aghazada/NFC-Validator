# NFC Optical Validator Demo

Python proof-of-concept for automatic visual NFC terminal validation using camera input.

## Supported modes

- `led` → green/red LED terminal
- `multi_led` → 4-LED payment-terminal style indicator
- `ocr` → screen text detection

## Features

- ROI-based processing
- multi-frame stabilization
- PASS / FAIL / UNKNOWN decision
- overlay UI
- evidence saving:
  - final annotated frame
  - ROI crop
  - JSON log
  - optional output video

## Install

```bash
pip install -r requirements.txt