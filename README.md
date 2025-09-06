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

---

## Bright Pixel Removal (`remove_bright_whites.py`)

Remove (or mask / make transparent) pixels brighter than a threshold.

### Usage

```bash
python remove_bright_whites.py inputs/*.jpg --threshold 0.97
```

Write results to an output directory with custom suffix and force PNG:

```bash
python remove_bright_whites.py images --recursive --out-dir out --format png --suffix _dim --threshold 0.99
```

Generate just a mask (white where bright):

```bash
python remove_bright_whites.py images/*.png --mode mask --threshold 0.98
```

Add / modify alpha channel to make bright pixels transparent:

```bash
python remove_bright_whites.py images/*.png --mode alpha --threshold 0.995
```

Use per-pixel max channel directly (equivalent to HSV Value) explicitly:

```bash
python remove_bright_whites.py images/*.jpg --use-max --threshold 0.99
```

### Arguments

```
inputs        One or more files / dirs / globs
--out-dir     Output directory (created if missing)
--suffix      Suffix for output file name (default: _dim)
--format      Output extension (keep input if omitted)
--overwrite   Allow overwriting existing outputs
--recursive   Recurse into directories
--threshold   Brightness threshold in [0,1] (default 0.98)
--mode        zero | mask | alpha (default zero)
--use-max     Use max channel for brightness (HSV Value equivalent)
```

### Behavior
Brightness is computed in [0,1]; pixels with brightness >= threshold are:
- zero: set to black (RGB/gray set to 0) preserving alpha if present
- mask: output a single-channel 8-bit mask (255 for bright, 0 else)
- alpha: ensure an alpha channel exists and set alpha=0 for bright pixels

Supports 8-bit and 16-bit inputs (alpha preserved). Grayscale inputs are handled.

