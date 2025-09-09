#!/usr/bin/env python3
"""
Batch white balance images to a specified color temperature in Kelvin using OpenCV.

Features:
- Specify a target color temperature (Kelvin) and balance all images to that white.
- Robust illuminant estimation (shades-of-gray) per image to compute gains.
- Saturation-aware estimation to avoid blown highlights.
- Simple CLI with dir/glob support and safe output handling.
"""

from __future__ import annotations

import argparse
import os
import sys
import glob
from typing import List, Optional, Sequence

import cv2
import numpy as np


# ----------------------------- Core algorithms ----------------------------- #


def _to_float01(bgr8: np.ndarray) -> np.ndarray:
	if bgr8.dtype == np.uint8:
		return bgr8.astype(np.float32) / 255.0
	if bgr8.dtype == np.uint16:
		# Normalize assuming full range
		return (bgr8.astype(np.float32)) / 65535.0
	# Assume already float in [0,1]
	return bgr8.astype(np.float32)


def _valid_mask(img_f: np.ndarray, sat_thresh: float, low_thresh: float) -> np.ndarray:
	# keep pixels where all channels are below saturation and above low threshold
	# shape: (H, W)
	below_top = (img_f < sat_thresh).all(axis=2)
	above_low = (img_f > low_thresh).all(axis=2)
	return below_top & above_low


def estimate_white(img_bgr: np.ndarray, p: float = 6.0, sat_thresh: float = 0.98, low_thresh: float = 0.0) -> np.ndarray:
	"""Estimate scene white (B,G,R) using shades-of-gray (Minkowski p-norm)."""
	img_f = _to_float01(img_bgr)
	mask = _valid_mask(img_f, sat_thresh, low_thresh)
	if not np.any(mask):
		mask = np.ones(img_f.shape[:2], dtype=bool)
	ch = [img_f[..., c][mask] for c in range(3)]  # B, G, R
	pf = float(p)
	def minkowski_mean(x: np.ndarray) -> float:
		if x.size == 0:
			return 1.0
		return float(np.power(np.mean(np.power(x, pf)), 1.0 / pf))
	vals = np.array([minkowski_mean(c) for c in ch], dtype=np.float32)
	vals = np.clip(vals, 1e-6, None)
	return vals  # B, G, R


def normalize_gains(gains_bgr: np.ndarray, mode: str = "geometric") -> np.ndarray:
	g = gains_bgr.astype(np.float32)
	if mode == "none":
		return g
	if mode == "green":
		anchor = g[1]
		if anchor <= 0:
			return g
		return g / anchor
	# geometric mean normalization keeps overall brightness stable
	gm = float(np.prod(g)) ** (1.0 / 3.0)
	if gm <= 0:
		return g
	return g / gm


# ----------------------------- Kelvin utilities --------------------------- #


def _srgb_gamma_encode(linear_rgb: np.ndarray) -> np.ndarray:
	"""Convert linear RGB in [0, +inf) to sRGB gamma-encoded [0,1]."""
	a = 0.055
	rgb = np.clip(linear_rgb, 0.0, None).astype(np.float32)
	low = rgb <= 0.0031308
	out = np.empty_like(rgb)
	out[low] = 12.92 * rgb[low]
	out[~low] = (1 + a) * np.power(rgb[~low], 1 / 2.4) - a
	return np.clip(out, 0.0, 1.0)


def kelvin_to_srgb_bgr_white(kelvin: float) -> np.ndarray:
	"""Approximate the sRGB gamma-encoded BGR white vector for a given CCT.

	Steps:
	- Convert CCT (Kelvin) to CIE xy chromaticity (McCamy-esque piecewise fit).
	- Set Y=1 to get XYZ, then convert to linear sRGB via XYZ->RGB.
	- Gamma-encode to sRGB, clamp to [0,1], and return in BGR order.

	Returns: np.ndarray of shape (3,) in BGR order.
	"""
	# Clamp reasonable range to keep approximation valid
	T = float(np.clip(kelvin, 1667.0, 25000.0))

	# Compute x from CCT
	if T <= 4000.0:
		x = (
			-0.2661239e9 / (T ** 3)
			- 0.2343580e6 / (T ** 2)
			+ 0.8776956e3 / T
			+ 0.179910
		)
	else:
		x = (
			-3.0258469e9 / (T ** 3)
			+ 2.1070379e6 / (T ** 2)
			+ 0.2226347e3 / T
			+ 0.240390
		)

	# Compute y from x depending on range
	if T <= 2222.0:
		y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
	elif T <= 4000.0:
		y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
	else:
		y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483

	# Convert xyY (Y=1) to XYZ
	if y <= 1e-6:
		X, Y, Z = 1.0, 1.0, 1.0
	else:
		X = x / y
		Y = 1.0
		Z = (1.0 - x - y) / y
	XYZ = np.array([X, Y, Z], dtype=np.float32)

	# XYZ to linear sRGB
	M = np.array([
		[3.2406, -1.5372, -0.4986],
		[-0.9689, 1.8758, 0.0415],
		[0.0557, -0.2040, 1.0570],
	], dtype=np.float32)
	rgb_lin = M @ XYZ
	# Remove negatives (out-of-gamut) before gamma
	rgb_lin = np.clip(rgb_lin, 0.0, None)
	rgb = _srgb_gamma_encode(rgb_lin)
	# Return BGR order
	bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
	# Avoid zeros to keep division safe; scale not critical (later normalized)
	bgr = np.clip(bgr, 1e-6, None)
	return bgr


def apply_gains(img_bgr: np.ndarray, gains_bgr: np.ndarray) -> np.ndarray:
	img = _to_float01(img_bgr)
	out = img * gains_bgr.reshape(1, 1, 3)
	out = np.clip(out, 0.0, 1.0)
	# Convert back to original bit depth
	if img_bgr.dtype == np.uint16:
		return (out * 65535.0 + 0.5).astype(np.uint16)
	return (out * 255.0 + 0.5).astype(np.uint8)


# No data-driven target aggregation is needed; target white comes from Kelvin


# ----------------------------- I/O and CLI -------------------------------- #


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
	# de-duplicate while preserving order
	seen = set()
	deduped = []
	for p in paths:
		ap = os.path.abspath(p)
		if ap not in seen:
			seen.add(ap)
			deduped.append(ap)
	return deduped


def ensure_out_path(out_dir: Optional[str], in_path: str, suffix: str, out_format: Optional[str]) -> str:
	base = os.path.basename(in_path)
	stem, ext = os.path.splitext(base)
	ext_out = out_format if out_format else ext
	if not ext_out.startswith("."):
		ext_out = "." + ext_out
	out_base = f"{stem}{suffix}{ext_out}"
	if out_dir:
		return os.path.join(out_dir, out_base)
	return os.path.join(os.path.dirname(in_path), out_base)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="White balance images to a specified color temperature (Kelvin).")
	p.add_argument("--inputs", nargs="+", required=True, help="Input image files, dirs, or globs.")
	p.add_argument("--kelvin", type=float, required=True, help="Target color temperature in Kelvin (e.g., 6500).")
	p.add_argument("--out-dir", default=None, help="Output directory (will be created). Default: alongside inputs.")
	p.add_argument("--suffix", default="_wb", help="Suffix for output filenames (before extension). Default: _wb")
	p.add_argument("--format", default=None, help="Output format/extension (e.g., jpg, png). Default: keep input extension.")
	p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files.")

	# Estimation and application knobs
	p.add_argument("--p", type=float, default=6.0, help="Minkowski p for shades-of-gray (higher is more like max).")
	p.add_argument("--sat-thresh", type=float, default=0.98, help="Exclude pixels with any channel >= this (0-1).")
	p.add_argument("--low-thresh", type=float, default=0.0, help="Exclude pixels with any channel <= this (0-1).")
	p.add_argument("--gains-norm", default="geometric", choices=["geometric", "none", "green"], help="Normalize channel gains to stabilize brightness.")

	p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories for dir/glob inputs.")
	return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)

	# Collect inputs
	img_paths = find_images(args.inputs, recursive=args.recursive)
	if not img_paths:
		print("No images found for given inputs", file=sys.stderr)
		return 2

	os.makedirs(args.out_dir, exist_ok=True) if args.out_dir else None

	# Target white from specified Kelvin
	target_white = kelvin_to_srgb_bgr_white(args.kelvin)

	# Process and write outputs
	num_done = 0
	for pth in img_paths:
		img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
		if img is None:
			print(f"[WARN] Skipping unreadable: {pth}")
			continue
		# Estimate scene white and compute gains to move it to the target white
		w = estimate_white(img, p=args.p, sat_thresh=args.sat_thresh, low_thresh=args.low_thresh)
		gains = target_white / np.clip(w, 1e-6, None)
		gains = normalize_gains(gains, mode=args.gains_norm)
		out_img = apply_gains(img, gains)
		out_path = ensure_out_path(args.out_dir, pth, args.suffix, args.format)
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		if (not args.overwrite) and os.path.exists(out_path):
			print(f"[SKIP] Exists: {out_path}")
			continue
		# Note: cv2.imwrite expects BGR order; out_img is BGR
		ok = cv2.imwrite(out_path, out_img)
		if not ok:
			print(f"[ERROR] Failed to write: {out_path}")
			continue
		num_done += 1
		print(f"[OK] {pth} -> {out_path}")

	if num_done == 0:
		print("No files were written.")
		return 1
	print(f"Done. Wrote {num_done} file(s).")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

