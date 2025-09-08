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

from moviepy.editor import (  # type: ignore
    VideoFileClip,
    CompositeAudioClip,
    concatenate_audioclips,
    concatenate_videoclips,
    AudioFileClip,
    afx,
    clips_array,
)

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


def load_and_prepare_clip(path: str, target_h: int, duration: Optional[float]) -> VideoFileClip:
    clip = VideoFileClip(path)
    # Resize while preserving aspect
    if target_h > 0:
        clip = clip.resize(height=target_h)
    if duration is not None:
        if clip.duration > duration:
            clip = clip.subclip(0, duration)
        elif clip.duration < duration:
            # pad by freezing last frame (create an ImageClip of the last frame)
            freeze_time = max(0.0, clip.duration - 1e-3)
            freeze_v = clip.to_ImageClip(t=freeze_time).set_duration(duration - clip.duration)
            # Keep FPS consistent for smoother concat
            try:
                freeze_v = freeze_v.set_fps(clip.fps)
            except Exception:
                pass
            clip = concatenate_videoclips([clip, freeze_v])
    return clip


def build_audio(clips: List[VideoFileClip], strategy: str) -> Optional[AudioFileClip]:
    if strategy == "none":
        return None
    # Filter clips with audio
    auds = [c.audio for c in clips if c.audio is not None]
    if not auds:
        return None
    if strategy == "first":
        return auds[0]
    if strategy == "mix":
        # Simple average mix to avoid clipping significantly
        # Normalize each audio to avoid huge differences
        normed = []
        for a in auds:
            try:
                # Apply volumex after measuring max (approx) -- moviepy lacks direct RMS quickly; keep simple
                normed.append(a.volumex(1.0 / len(auds)))
            except Exception:
                normed.append(a.volumex(1.0 / len(auds)))
        return CompositeAudioClip(normed)
    raise ValueError(f"Unknown audio strategy: {strategy}")


def build_grid(clips: List[VideoFileClip], rows: int, cols: int):
    # Pad list with black clips if needed to fill grid
    total = rows * cols
    if len(clips) < total:
        w, h = clips[0].w, clips[0].h
        from moviepy.editor import ColorClip  # lazy import
        blanks = total - len(clips)
        for _ in range(blanks):
            clips.append(ColorClip(size=(w, h), color=(0, 0, 0)).set_duration(clips[0].duration))
    grid = []
    for r in range(rows):
        row_clips = clips[r*cols:(r+1)*cols]
        grid.append(row_clips)
    return clips_array(grid)


def ensure_ffmpeg(prefer_imageio: bool = True) -> None:
    """Ensure a recent ffmpeg binary is used.

    If prefer_imageio is True and imageio-ffmpeg is available, point MoviePy/ImageIO
    to that binary to avoid distro-provided outdated ffmpeg.
    """
    if not prefer_imageio:
        return
    try:
        import imageio_ffmpeg  # type: ignore

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        # For imageio-ffmpeg consumers
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg_exe)
        # For moviepy (older versions read config setting)
        try:
            from moviepy.config import change_settings  # type: ignore

            change_settings({"FFMPEG_BINARY": ffmpeg_exe})
        except Exception:
            pass
    except Exception:
        # imageio-ffmpeg not available: will fall back to system ffmpeg
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Prefer a recent ffmpeg to avoid first-frame decode errors
    ensure_ffmpeg(prefer_imageio=True)
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

    prepared: List[VideoFileClip] = []
    tried: set[str] = set()

    def try_load(path: str) -> Optional[VideoFileClip]:
        tried.add(path)
        try:
            return load_and_prepare_clip(path, cfg.resize_height, cfg.duration)
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
    grid_clip = build_grid(prepared, rows, cols)

    if not cfg.mute:
        audio_clip = build_audio(prepared, cfg.audio)
        if audio_clip is not None:
            grid_clip = grid_clip.set_audio(audio_clip)
    else:
        grid_clip = grid_clip.set_audio(None)

    try:
        base_fps = getattr(prepared[0], "fps", None)
    except Exception:
        base_fps = None
    out_fps = cfg.fps or (base_fps if isinstance(base_fps, (int, float)) and base_fps > 0 else 24)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Optionally scale final collage to a max width (preserving aspect)
    if getattr(args, "max_width", 0) and grid_clip.w > args.max_width:
        scale = args.max_width / float(grid_clip.w)
        new_h = max(1, int(round(grid_clip.h * scale)))
        grid_clip = grid_clip.resize(newsize=(args.max_width, new_h))

    print(f"Writing collage to {args.out} ({rows}x{cols} grid, fps={out_fps}, size={grid_clip.w}x{grid_clip.h})")
    try:
        grid_clip.write_videofile(args.out, fps=out_fps, codec="libx264", audio_codec="aac")
    finally:
        # Close all clips to release resources
        for c in prepared:
            c.close()
        grid_clip.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
