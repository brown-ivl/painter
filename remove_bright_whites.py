#!/usr/bin/env python3
"""Remove (zero out or make transparent) pixels brighter than a threshold.

Similar CLI style to `balance_whites.py`:
 - Accept files / dirs / globs
 - Output directory / suffix handling
 - Format, overwrite, recursive options

Bright pixel definition:
 By default evaluates brightness on the Value channel of HSV (range 0-1 when
 normalized). A pixel is considered "bright" if V >= --threshold.

Actions for bright pixels:
  --mode zero     : set pixel to black (default)
  --mode mask     : write a single-channel mask image (255 where bright)
  --mode alpha    : add/replace alpha channel setting bright pixels alpha=0

Threshold input is in [0,1] for convenience; internally scaled according to
bit depth. Supports 8-bit and 16-bit images, as well as existing alpha.
"""

from __future__ import annotations

import argparse
import os
import sys
import glob
from typing import List, Optional, Sequence

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def find_images(inputs: Sequence[str], recursive: bool = True) -> List[str]:
	paths: List[str] = []
	for inp in inputs:
		if any(ch in inp for ch in ("*", "?", "[")):
			paths.extend(sorted(glob.glob(inp, recursive=recursive)))
			continue
		if os.path.isdir(inp):
			for root, _, files in os.walk(inp):
				for f in files:
					if f.lower().endswith(IMG_EXTS):
						paths.append(os.path.join(root, f))
			continue
		if os.path.isfile(inp):
			paths.append(inp)
	# de-duplicate preserving order
	seen = set()
	out: List[str] = []
	for p in paths:
		ap = os.path.abspath(p)
		if ap not in seen:
			seen.add(ap)
			out.append(ap)
	return out


def ensure_out_path(out_dir: Optional[str], in_path: str, suffix: str, out_format: Optional[str]) -> str:
	base = os.path.basename(in_path)
	stem, ext = os.path.splitext(base)
	ext_out = out_format if out_format else ext
	if not ext_out.startswith('.'):
		ext_out = '.' + ext_out
	out_base = f"{stem}{suffix}{ext_out}"
	if out_dir:
		return os.path.join(out_dir, out_base)
	return os.path.join(os.path.dirname(in_path), out_base)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Remove bright pixels above threshold.")
	p.add_argument("inputs", nargs="+", help="Input image files, dirs, or globs.")
	p.add_argument("--out-dir", default=None, help="Output directory (will be created). Default: alongside inputs.")
	p.add_argument("--suffix", default="_dim", help="Suffix for output filenames (before extension). Default: _dim")
	p.add_argument("--format", default=None, help="Output format/extension (e.g., png). Default: keep input extension.")
	p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files.")
	p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories for dir/glob inputs.")

	p.add_argument("--threshold", type=float, default=0.98, help="Brightness threshold in [0,1] for removal.")
	p.add_argument("--mode", choices=["zero", "mask", "alpha"], default="zero", help="How to handle bright pixels.")
	p.add_argument("--use-max", action="store_true", help="Use max channel instead of HSV Value.")

	return p.parse_args(argv)


def compute_brightness(img: np.ndarray, use_max: bool) -> np.ndarray:
	"""Return brightness in float32 [0,1]."""
	if img.dtype == np.uint8:
		scale = 255.0
	elif img.dtype == np.uint16:
		scale = 65535.0
	else:
		scale = 1.0

	# Work on BGR (ignore alpha if present)
	if img.ndim == 2:
		ch = img.astype(np.float32) / scale
		return ch
	bgr = img[..., :3].astype(np.float32) / scale
	if use_max:
		return np.max(bgr, axis=2)
	# HSV Value via OpenCV expects uint8 or float32 in 0-255? We'll convert manually using formula max channel when not use_max.
	# Since Value == max channel, using max already replicates HSV V, but we keep branch for clarity.
	return np.max(bgr, axis=2)


def apply_mode(img: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
	if mode == "mask":
		# Return 8-bit mask regardless of input depth
		return (mask.astype(np.uint8) * 255)
	if mode == "zero":
		out = img.copy()
		if out.ndim == 2:  # grayscale
			out[mask] = 0
		else:
			out[mask, ...] = 0
		return out
	if mode == "alpha":
		# Ensure we have alpha channel
		if img.ndim == 2:
			# Promote to BGRA with repeated channel
			base = img
			if img.dtype == np.uint8:
				alpha_full = 255
			elif img.dtype == np.uint16:
				alpha_full = 65535
			else:
				alpha_full = 1.0
			a = np.full_like(base, alpha_full)
			out = np.stack([base, base, base, a], axis=2)
		else:
			if img.shape[2] == 4:
				out = img.copy()
			else:
				# Add opaque alpha
				if img.dtype == np.uint8:
					alpha_full = 255
				elif img.dtype == np.uint16:
					alpha_full = 65535
				else:
					alpha_full = 1.0
				alpha = np.full(img.shape[:2], alpha_full, dtype=img.dtype)
				out = np.concatenate([img, alpha[..., None]], axis=2)
		# Zero alpha where mask
		if out.dtype == np.uint8:
			out[mask, 3] = 0
		elif out.dtype == np.uint16:
			out[mask, 3] = 0
		else:
			out[mask, 3] = 0.0
		return out
	raise ValueError(f"Unknown mode: {mode}")


def process_image(path: str, threshold: float, mode: str, use_max: bool) -> Optional[np.ndarray]:
	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	if img is None:
		print(f"[WARN] Unreadable: {path}")
		return None
	if not (0.0 <= threshold <= 1.0):
		raise ValueError("threshold must be in [0,1]")
	brightness = compute_brightness(img, use_max=use_max)
	mask = brightness >= threshold
	out = apply_mode(img, mask, mode)
	return out


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)

	img_paths = find_images(args.inputs, recursive=args.recursive)
	if not img_paths:
		print("No images found", file=sys.stderr)
		return 2

	if args.out_dir:
		os.makedirs(args.out_dir, exist_ok=True)

	written = 0
	for pth in img_paths:
		out_img = process_image(pth, args.threshold, args.mode, args.use_max)
		if out_img is None:
			continue
		out_path = ensure_out_path(args.out_dir, pth, args.suffix, args.format)
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		if (not args.overwrite) and os.path.exists(out_path):
			print(f"[SKIP] Exists: {out_path}")
			continue
		ok = cv2.imwrite(out_path, out_img)
		if not ok:
			print(f"[ERROR] Failed to write: {out_path}")
			continue
		written += 1
		print(f"[OK] {pth} -> {out_path}")

	if written == 0:
		print("No files written.")
		return 1
	print(f"Done. Wrote {written} file(s).")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

