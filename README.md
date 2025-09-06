# Batch White Balance

A simple Python CLI to batch white-balance multiple images to a common white point using OpenCV.

## Features
- Gray-World or Shades-of-Gray (Minkowski p-norm) illuminant estimation
- Pick a reference image or use the median/mean across inputs
- Saturation-aware estimation to avoid blown highlights
- Output directory, suffix, and format control

## Quick start

Install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run on a folder of images and write results next to inputs:

```bash
python balance_whites.py images/*.jpg
```

Use median target white (default) and write to `out/`:

```bash
python balance_whites.py images --recursive --out-dir out
```

Use a specific reference image as the white target:

```bash
python balance_whites.py images --recursive --target reference --reference images/ref.jpg
```

Increase white stability (more max-like) with higher p:

```bash
python balance_whites.py images --recursive --method shades --p 10
```

Force output format and suffix:

```bash
python balance_whites.py images --recursive --out-dir out --format jpg --suffix _wb
```

## Notes
- Supported inputs: files, directories, and glob patterns.
- Supported formats: .jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp
- Gains normalization uses geometric mean by default to keep overall brightness stable.
