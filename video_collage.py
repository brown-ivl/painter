#!/usr/bin/env python3
"""Build a collage (grid) of randomly selected video clips.

Features:
- Recursively discover video files (common extensions)
- Randomly select N clips (optionally shuffle first) with reproducible seed
- Uniformly resize clips to a target height while preserving aspect ratio
- Trim or pad (freeze last frame) to a common duration window
- Arrange in a grid with configurable columns (rows computed automatically)
- Optional audio strategies: none, first clip, simple mix
- Writes output MP4 with H.264 codec by default

Examples:
    python video_collage.py videos/ --recursive -n 9 --cols 3 --out collage.mp4
    python video_collage.py videos --recursive -n 8 --cols 4 --resize-height 240 --duration 5 \
        --seed 123 --audio first --out collage_short.mp4

Limitations:
- Simple audio mixing (average) may clip; advanced mixing not implemented
- moviepy can be memory intensive for many large clips; prefer small N or low height
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".wmv", ".flv")


def find_videos(inputs: Sequence[str], recursive: bool) -> List[str]:
    paths: List[str] = []
    for inp in inputs:
        if os.path.isdir(inp):
            if recursive:
                for root, _, files in os.walk(inp):
                    for f in files:
                        if f.lower().endswith(VIDEO_EXTS):
                            paths.append(os.path.join(root, f))
            else:
                for f in os.listdir(inp):
                    p = os.path.join(inp, f)
                    if os.path.isfile(p) and f.lower().endswith(VIDEO_EXTS):
                        paths.append(p)
        elif os.path.isfile(inp):
            if inp.lower().endswith(VIDEO_EXTS):
                paths.append(inp)
        else:
            # ignore unmatched patterns (no globbing support here to keep simple)
            pass
    # de-duplicate preserving order
    seen = set()
    out: List[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    return out


@dataclass
class CollageConfig:
    num: int
    cols: int
    duration: Optional[float]
    resize_height: int
    seed: Optional[int]
    audio: str  # none|first|mix
    mute: bool
    fps: Optional[int]
    shuffle_pool: bool


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random video collage builder")
    p.add_argument("--inputs", nargs="+", help="Input video files or directories")
    p.add_argument("-n", "--num", type=int, required=True, help="Number of videos to select")
    p.add_argument("--recursive", action="store_true", help="Recurse into directories")
    p.add_argument("--cols", type=int, default=0, help="Number of columns in grid (0=auto optimal)")
    p.add_argument("--max-width", type=int, default=0, help="Max output width; scale down if exceeded (0=disable)")
    p.add_argument("--duration", type=float, default=None, help="Target duration seconds (trim or freeze)" )
    p.add_argument("--resize-height", type=int, default=240, help="Uniform height for each cell")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--audio", choices=["none", "first", "mix"], default="none", help="Audio strategy")
    p.add_argument("--mute", action="store_true", help="Force mute output (overrides --audio)")
    p.add_argument("--fps", type=int, default=None, help="Override output FPS (default: first clip fps)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle candidate list before selection")
    p.add_argument("--out", required=True, help="Output video path (mp4)")
    return p.parse_args(argv)


def select_paths(all_paths: List[str], n: int, seed: Optional[int], shuffle_pool: bool) -> List[str]:
    if seed is not None:
        random.seed(seed)
    if not all_paths:
        return []
    pool = list(all_paths)
    if shuffle_pool:
        random.shuffle(pool)
    if n >= len(pool):
        return pool
    return random.sample(pool, n)


def compute_grid(n: int, cols: int) -> Tuple[int, int]:
    """Compute grid rows, cols.

    If cols <= 0, choose an approximately square grid using ceil(sqrt(n)).
    """
    if cols is None or cols <= 0:
        cols = max(1, math.ceil(math.sqrt(max(1, n))))
    rows = math.ceil(n / cols)
    return rows, cols


def load_and_prepare_clip_cv2(path: str, target_h: int, duration: Optional[float]) -> Tuple[list, float, int, int, float]:
    """
    Loads video frames using OpenCV, resizes to target_h, and trims/pads to duration (in seconds).
    Returns: (frames, fps, width, height, actual_duration)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    max_frames = total_frames
    if duration is not None:
        max_frames = int(round(fps * duration))
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to target_h
        if target_h > 0 and frame.shape[0] != target_h:
            aspect = frame.shape[1] / frame.shape[0]
            new_w = int(round(target_h * aspect))
            frame = cv2.resize(frame, (new_w, target_h))
        frames.append(frame)
    cap.release()
    # Pad with last frame if needed
    if duration is not None and len(frames) < max_frames and frames:
        last = frames[-1]
        while len(frames) < max_frames:
            frames.append(last.copy())
    if frames:
        h, w = frames[0].shape[:2]
    else:
        h, w = target_h, target_h
    actual_duration = len(frames) / fps
    return frames, fps, w, h, actual_duration


def build_audio(*args, **kwargs):
    # No audio support in OpenCV version
    return None


def make_collage_video(video_frames_list, grid_shape, max_width=0, fps=24, out_path=None):
    """
    video_frames_list: list of lists of frames (each is a list of np.ndarray)
    grid_shape: (nRows, nCols)
    max_width: int, if >0, scale collage to this width
    fps: output fps
    out_path: if set, write video to this path
    """
    n_videos = len(video_frames_list)
    nRows, nCols = grid_shape
    # Find the minimum number of frames across all videos
    min_frames = min(len(frames) for frames in video_frames_list)
    # Truncate all to min_frames
    video_frames_list = [frames[:min_frames] for frames in video_frames_list]
    # Assume all frames are same size as first
    h, w = video_frames_list[0][0].shape[:2]
    # Pad with black videos if needed
    if n_videos < nRows * nCols:
        blank = np.zeros_like(video_frames_list[0][0])
        for _ in range(nRows * nCols - n_videos):
            video_frames_list.append([blank.copy() for _ in range(min_frames)])
    # Build each frame of the collage
    collage_frames = []
    for i in range(min_frames):
        row_imgs = []
        for r in range(nRows):
            row = []
            for c in range(nCols):
                idx = r * nCols + c
                row.append(video_frames_list[idx][i])
            row_imgs.append(np.hstack(row))
        collage = np.vstack(row_imgs)
        collage_frames.append(collage)
    # Optionally resize
    if max_width > 0 and collage_frames[0].shape[1] > max_width:
        scale = max_width / collage_frames[0].shape[1]
        new_h = int(round(collage_frames[0].shape[0] * scale))
        collage_frames = [cv2.resize(f, (max_width, new_h)) for f in collage_frames]
    # Write video
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (collage_frames[0].shape[1], collage_frames[0].shape[0]))
        for f in collage_frames:
            out.write(f)
        out.release()
    return collage_frames





    args = parse_args(argv)

    all_videos = find_videos(args.inputs, recursive=args.recursive)
    if not all_videos:
        print("No videos found", file=sys.stderr)
        return 2

    chosen = select_paths(all_videos, args.num, args.seed, args.shuffle)
    if not chosen:
        print("Failed to select videos", file=sys.stderr)
        return 3

    cfg = CollageConfig(
        num=args.num,
        cols=args.cols,
        duration=args.duration,
        resize_height=args.resize_height,
        seed=args.seed,
        audio=args.audio,
        mute=args.mute,
        fps=args.fps,
        shuffle_pool=args.shuffle,
    )

    prepared: list = []
    tried: set[str] = set()

    def try_load(path: str):
        tried.add(path)
        try:
            frames, fps, w, h, actual_duration = load_and_prepare_clip_cv2(path, cfg.resize_height, cfg.duration)
            return dict(frames=frames, fps=fps, w=w, h=h, duration=actual_duration)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            return None

    for p in chosen:
        clip = try_load(p)
        if clip is not None:
            prepared.append(clip)

    # If we didn't reach desired count due to unreadable files, pull more from the pool
    if len(prepared) < cfg.num:
        remaining = [p for p in all_videos if p not in tried]
        if cfg.seed is not None:
            random.seed(cfg.seed + 1)
        random.shuffle(remaining)
        for p in remaining:
            if len(prepared) >= cfg.num:
                break
            clip = try_load(p)
            if clip is not None:
                prepared.append(clip)
    if not prepared:
        print("No clips loaded", file=sys.stderr)
        return 4

    rows, cols = compute_grid(len(prepared), cfg.cols)
    # Use the minimum fps of all videos for safety
    out_fps = cfg.fps or min(int(round(c['fps'])) for c in prepared if c['fps'] > 0) or 24
    # Build the collage
    video_frames_list = [c['frames'] for c in prepared]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    make_collage_video(video_frames_list, (rows, cols), max_width=args.max_width, fps=out_fps, out_path=args.out)
    print(f"Wrote collage to {args.out} ({rows}x{cols} grid, fps={out_fps})")
    return 0


def main():
    import sys
    return _main(sys.argv[1:])

def _main(argv=None):
    args = parse_args(argv)

    all_videos = find_videos(args.inputs, recursive=args.recursive)
    if not all_videos:
        print("No videos found", file=sys.stderr)
        return 2

    chosen = select_paths(all_videos, args.num, args.seed, args.shuffle)
    if not chosen:
        print("Failed to select videos", file=sys.stderr)
        return 3

    cfg = CollageConfig(
        num=args.num,
        cols=args.cols,
        duration=args.duration,
        resize_height=args.resize_height,
        seed=args.seed,
        audio=args.audio,
        mute=args.mute,
        fps=args.fps,
        shuffle_pool=args.shuffle,
    )

    prepared: list = []
    tried: set[str] = set()

    def try_load(path: str):
        tried.add(path)
        try:
            frames, fps, w, h, actual_duration = load_and_prepare_clip_cv2(path, cfg.resize_height, cfg.duration)
            return dict(frames=frames, fps=fps, w=w, h=h, duration=actual_duration)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            return None

    for p in chosen:
        clip = try_load(p)
        if clip is not None:
            prepared.append(clip)

    # If we didn't reach desired count due to unreadable files, pull more from the pool
    if len(prepared) < cfg.num:
        remaining = [p for p in all_videos if p not in tried]
        if cfg.seed is not None:
            random.seed(cfg.seed + 1)
        random.shuffle(remaining)
        for p in remaining:
            if len(prepared) >= cfg.num:
                break
            clip = try_load(p)
            if clip is not None:
                prepared.append(clip)
    if not prepared:
        print("No clips loaded", file=sys.stderr)
        return 4

    rows, cols = compute_grid(len(prepared), cfg.cols)
    # Use the minimum fps of all videos for safety
    out_fps = cfg.fps or min(int(round(c['fps'])) for c in prepared if c['fps'] > 0) or 24
    # Build the collage
    video_frames_list = [c['frames'] for c in prepared]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    make_collage_video(video_frames_list, (rows, cols), max_width=args.max_width, fps=out_fps, out_path=args.out)
    print(f"Wrote collage to {args.out} ({rows}x{cols} grid, fps={out_fps})")
    return 0

if __name__ == "__main__":
    SystemExit(main())
