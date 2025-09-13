from __future__ import annotations
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
"""
# (Deduplicated header below)
"""Build a collage (grid) of randomly selected video clips using OpenCV.

Features:
- Recursively discover video files (common extensions)
- Randomly select N clips (optionally shuffle first) with reproducible seed
- Uniformly resize clips to a target height while preserving aspect ratio
- Trim or pad (freeze last frame) to a common duration window
- Arrange in a grid with configurable columns (rows computed automatically; sqrt-based optimal when 0)
- Writes silent MP4 using OpenCV (no audio to avoid MoviePy dependency)

Examples:
  python video_collage.py --inputs videos/ --recursive -n 9 --cols 0 --out collage.mp4
  python video_collage.py --inputs videos --recursive -n 8 --resize-height 240 --duration 5 --seed 123 --out collage.mp4
"""
import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any
import json
import shlex
import subprocess

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".wmv", ".flv")


# ---------- Discovery ----------

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
    # de-duplicate preserving order
    seen = set()
    out: List[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    return out


# ---------- Config ----------

@dataclass
class CollageConfig:
    num: int
    cols: int
    duration: Optional[float]
    resize_height: int
    seed: Optional[int]
    fps: Optional[int]
    shuffle_pool: bool
    max_width: int
    cinema_zoom: bool
    focus_index: int


# ---------- CLI ----------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random video collage builder (OpenCV)")
    p.add_argument("--inputs", nargs="+", required=True, help="Input video files or directories")
    p.add_argument("-n", "--num", type=int, required=True, help="Number of videos to select")
    p.add_argument("--recursive", action="store_true", help="Recurse into directories")
    p.add_argument("--cols", type=int, default=0, help="Number of columns (0=auto optimal)")
    p.add_argument("--max-width", type=int, default=0, help="Max output width; scale down if exceeded (0=disable)")
    p.add_argument("--duration", type=float, default=None, help="Target duration seconds (trim/pad)")
    p.add_argument("--resize-height", type=int, default=240, help="Uniform height for each cell")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--fps", type=int, default=None, help="Override output FPS (default: min of inputs or 24)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle candidate list before selection")
    p.add_argument("--cinema-zoom", action="store_true", help="Start on one cell and zoom out to full grid by mid-duration (requires --duration)")
    p.add_argument("--focus-index", type=int, default=0, help="Which selected clip index to focus at start (default: 0)")
    p.add_argument("--out", required=True, help="Output video path (mp4)")
    return p.parse_args(argv)


# ---------- Helpers ----------

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
    # If cols <= 0, choose an approximately square grid using ceil(sqrt(n)).
    if cols <= 0:
        cols = max(1, math.ceil(math.sqrt(max(1, n))))
    rows = math.ceil(n / cols)
    return rows, cols


# ---------- Probing and ffmpeg-based pipeline ----------

def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)


def ffprobe_video(path: str) -> Optional[Dict[str, Any]]:
    """Return width, height, duration (float), and avg_frame_rate for the first video stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json",
        path,
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format", {})
        w = int(stream.get("width") or 0)
        h = int(stream.get("height") or 0)
        afr = stream.get("avg_frame_rate") or "0/0"
        # avg_frame_rate like '30000/1001'
        try:
            num, den = afr.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except Exception:
            fps = 0.0
        try:
            duration = float(fmt.get("duration") or 0.0)
        except Exception:
            duration = 0.0
        if w <= 0 or h <= 0:
            return None
        return {"width": w, "height": h, "fps": fps, "duration": duration}
    except Exception:
        return None


def even(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def make_filter_complex(meta_list: List[Dict[str, Any]], rows: int, cols: int, target_h: int, out_fps: int, duration: Optional[float], max_width: int, cinema_zoom: bool, focus_index: int) -> Tuple[str, Tuple[int, int], str]:
    """Build filter_complex for xstack grid and return (filter, (out_w, out_h), final_label)."""
    # Compute scaled width per input and max column width
    scaled_w = []
    for m in meta_list:
        w, h = int(m["width"]), int(m["height"])
        sw = max(2, int(round(target_h * (w / h))))
        sw = even(sw)
        scaled_w.append(sw)
    cell_w = max(scaled_w) if scaled_w else target_h
    cell_h = even(target_h)
    total = rows * cols

    # Build per-input chains
    parts: List[str] = []
    labels: List[str] = []
    d = duration if duration and duration > 0 else None
    for i in range(len(meta_list)):
        chain = f"[{i}:v:0]scale={scaled_w[i]}:{cell_h},pad={cell_w}:{cell_h}:(ow-iw)/2:0:black,setpts=PTS-STARTPTS"
        if d is not None:
            chain += f",tpad=stop_mode=clone:stop_duration={d},trim=duration={d},setpts=PTS-STARTPTS"
        label = f"v{i}"
        chain += f"[{label}]"
        parts.append(chain)
        labels.append(f"[{label}]")

    # Add blanks if needed
    blanks = total - len(meta_list)
    # Choose blank duration
    if d is not None:
        d_blank = d
    else:
        # If no duration specified, use min duration among inputs (approx) so -shortest is okay
        d_blank = max(1.0, min((m.get("duration") or 0.0) for m in meta_list) or 1.0)
    for j in range(blanks):
        label = f"b{j}"
        parts.append(f"color=c=black:s={cell_w}x{cell_h}:r={out_fps}:d={d_blank}[{label}]")
        labels.append(f"[{label}]")

    # xstack layout
    layout_elems = []
    for idx in range(total):
        r = idx // cols
        c = idx % cols
        x = c * cell_w
        y = r * cell_h
        layout_elems.append(f"{x}_{y}")
    shortest = 0 if d is not None else 1
    layout = "|".join(layout_elems)
    inputs_concat = "".join(labels)
    xstack_out = "st"
    # Use xstack without 'fill' for broader ffmpeg compatibility; inputs are uniform size already
    parts.append(f"{inputs_concat}xstack=inputs={total}:layout={layout}:shortest={shortest}[{xstack_out}]")

    out_w = cell_w * cols
    out_h = cell_h * rows
    # Optional cinema-zoom (zoom-out) until mid-duration
    final_src = f"[{xstack_out}]"
    if cinema_zoom and duration and duration > 0:
        # Focus on the given cell index
        fi = max(0, min(rows * cols - 1, focus_index))
        r0 = fi // cols
        c0 = fi % cols
        cx = c0 * cell_w + cell_w / 2.0
        cy = r0 * cell_h + cell_h / 2.0
        # Use zoompan for robust per-frame zoom instead of dynamic crop expressions
        z0 = max(rows, cols)  # start fully on one cell or tighter (no other cells visible)
        mid_frames = max(1, int(round(out_fps * duration / 2.0)))
        # Smooth ease-out from z0 to 1 by mid-duration using smoothstep: 3u^2 - 2u^3
        # u = on/mid_frames
        ease = f"(3*pow(on/{mid_frames}\\,2) - 2*pow(on/{mid_frames}\\,3))"
        zoom_expr = f"if(lte(on\\,{mid_frames})\\,{z0} - ({z0}-1)*{ease}\\,1)"
        x_expr = f"max(0\\,min({cx}-iw/zoom/2\\, iw-iw/zoom))"
        y_expr = f"max(0\\,min({cy}-ih/zoom/2\\, ih-ih/zoom))"
        parts.append(f"{final_src}zoompan=z={zoom_expr}:x={x_expr}:y={y_expr}:d=1:fps={out_fps}:s={out_w}x{out_h}[cz]")
        # Normalize PTS and enforce exact duration (avoid extra fps here to preserve timing)
        parts.append(f"[cz]setpts=PTS-STARTPTS,trim=duration={duration}[czt]")
        final_src = "[czt]"

    # Optional downscale
    if max_width and out_w > max_width:
        scale_h = even(int(round(out_h * (max_width / out_w))))
        parts.append(f"{final_src}scale={even(max_width)}:{scale_h}[vout]")
        out_w, out_h = even(max_width), scale_h
        final_label = "[vout]"
    else:
        final_label = final_src

    filter_complex = ";".join(parts)
    return filter_complex, (out_w, out_h), final_label


def run_ffmpeg_grid(paths: List[str], rows: int, cols: int, target_h: int, out_fps: int, duration: Optional[float], max_width: int, out_path: str, cinema_zoom: bool, focus_index: int) -> int:
    # Probe all inputs
    meta_list: List[Dict[str, Any]] = []
    good_paths: List[str] = []
    for p in paths:
        m = ffprobe_video(p)
        if m is None:
            print(f"[WARN] Skipping (probe failed): {p}")
            continue
        meta_list.append(m)
        good_paths.append(p)
    if not meta_list:
        print("No valid videos to process", file=sys.stderr)
        return 4

    # Build filter graph
    filt, (out_w, out_h), final_label = make_filter_complex(meta_list, rows, cols, target_h, out_fps, duration, max_width, cinema_zoom, focus_index)

    # Build command
    cmd: List[str] = ["ffmpeg", "-y", "-hide_banner", "-stats", "-loglevel", "warning"]
    for p in good_paths:
        cmd += ["-i", p]
    cmd += [
    "-filter_complex", filt,
    "-map", final_label,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart",
        out_path,
    ]
    print("Running:")
    print(" ", " ".join(shlex.quote(x) for x in cmd))
    # Stream output (progress)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            # ffmpeg prints progress lines; forward them
            print(line.rstrip())
    finally:
        proc.wait()
    return proc.returncode


# ---------- Collage ----------

def make_collage_video(*args, **kwargs):
    # Backward compat placeholder; not used in ffmpeg path
    pass


# ---------- Main ----------

def _main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    all_videos = find_videos(args.inputs, recursive=args.recursive)
    # Filter: only consider videos whose filename contains the word "smartbric" (case-insensitive)
    all_videos = [p for p in all_videos if "bric" in os.path.basename(p).lower()]
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
        fps=args.fps,
        shuffle_pool=args.shuffle,
        max_width=args.max_width,
        cinema_zoom=args.cinema_zoom,
        focus_index=args.focus_index,
    )

    # Probe-first selection, skipping unreadables
    prepared_paths: List[str] = []
    tried: set[str] = set()

    def try_probe(path: str) -> bool:
        tried.add(path)
        m = ffprobe_video(path)
        if m is None:
            print(f"[WARN] Skipping {path}: probe failed")
            return False
        return True

    for p in chosen:
        if try_probe(p):
            prepared_paths.append(p)

    if len(prepared_paths) < cfg.num:
        remaining = [p for p in all_videos if p not in tried]
        if cfg.seed is not None:
            random.seed(cfg.seed + 1)
        random.shuffle(remaining)
        for p in remaining:
            if len(prepared_paths) >= cfg.num:
                break
            if try_probe(p):
                prepared_paths.append(p)

    if not prepared_paths:
        print("No clips loaded", file=sys.stderr)
        return 4

    rows, cols = compute_grid(len(prepared_paths), cfg.cols)
    # Validate cinema-zoom requirements
    if cfg.cinema_zoom and (cfg.duration is None or cfg.duration <= 0):
        print("--cinema-zoom requires --duration > 0 (transition completes at mid-duration)", file=sys.stderr)
        return 2
    # Determine output FPS: min across inputs (fallback 24) if not provided
    if cfg.fps is not None:
        out_fps = int(cfg.fps)
    else:
        fps_vals: List[float] = []
        for p in prepared_paths:
            info = ffprobe_video(p)
            if info and info.get("fps") and info["fps"] > 0:
                fps_vals.append(float(info["fps"]))
        out_fps = int(max(1, int(min(fps_vals) if fps_vals else 24)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rc = run_ffmpeg_grid(prepared_paths, rows, cols, cfg.resize_height, out_fps, cfg.duration, cfg.max_width, args.out, cfg.cinema_zoom, cfg.focus_index)
    if rc != 0:
        print(f"ffmpeg failed with code {rc}", file=sys.stderr)
        return rc
    print(f"Wrote collage to {args.out} ({rows}x{cols} grid, fps={out_fps})")
    return 0


def main() -> int:
    return _main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
