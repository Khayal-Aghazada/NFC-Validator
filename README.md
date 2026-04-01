# NFC Optical Validator

Computer vision proof-of-concept for automatic NFC terminal validation using camera input.

## What it does

This project analyzes the **optical output of NFC / payment terminals** and decides whether the interaction result is:

- **PASS**
- **FAIL**
- **UNKNOWN**

It works from **camera or video input** and supports:

- **LED mode** → green / red LED detection
- **Multi-LED mode** → payment-terminal-style indicator analysis
- **OCR mode** → screen text detection

---

## Features

- ROI-based processing
- Real-time webcam / video input
- Multi-frame stabilization
- Overlay UI
- Evidence saving:
  - annotated frame
  - ROI crop
  - JSON result log
  - optional recorded video

---

## How it works

1. Read frame from camera or video
2. Select terminal ROI
3. Run selected detector:
   - LED
   - Multi-LED
   - OCR
4. Stabilize results across multiple frames
5. Produce final result
6. Save evidence

---

## Run

### Webcam
```bash
python main.py --mode led --source 0
```

### Multi:
```bash
python main.py --mode multi_led --source 0
```

### OCR:
```bash
python main.py --mode multi_led --source 0
```

### Video file
```bash
python main.py --mode ocr --source test_data/sample.mp4
```

---

## Output

Saved outputs include:

- final annotated frame
- ROI crop
- JSON log
- optional output video

---

## Notes

This is a **proof-of-concept demo**, not a finished industrial product.

The project focuses on:

- practical computer vision
- modular design
- explainable logic
- traceable results

---

## Tech Stack

- Python
- OpenCV
- NumPy
- EasyOCR

---

## Author

Khayal Aghazada
