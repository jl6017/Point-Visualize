"""Combine center-cropped PNG images from a folder into an n x n grid.

Usage:
    python image_combine.py --in-folder path/to/pngs --out combined.png --grid 4 --crop 0.7

Behavior:
- Reads all .png files in the input folder, sorted by numeric sequence when possible.
- Center-crops each image to the given fraction (0.7 -> keep center 70% x 70%).
- Resizes cropped tiles to a consistent tile size (based on the first image's crop size).
- Pastes up to grid*grid images into an n x n canvas, filling empty cells with black.
- Saves the combined image to --out (default: combined.png).

"""
from pathlib import Path
import argparse
import re
from PIL import Image
import sys
import time


def _natural_key(p: Path):
    """Return a key for sorting that tries to use the first integer in filename, else the name."""
    s = p.stem
    m = re.search(r"(\d+)", s)
    if m:
        return (0, int(m.group(1)), s)
    return (1, s)


def center_crop(img: Image.Image, fraction: float) -> Image.Image:
    if not (0 < fraction <= 1.0):
        raise ValueError("crop fraction must be in (0, 1]")
    w, h = img.size
    cw = int(round(w * fraction))
    ch = int(round(h * fraction))
    left = (w - cw) // 2
    top = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def combine_images(folder: Path, out_path: Path, n: int = 4, m: int = 4, crop: float = 0.7):
    pngs = sorted(folder.glob("*.png"), key=_natural_key)
    if not pngs:
        raise FileNotFoundError(f"No .png files found in: {folder}")

    max_tiles = n * m
    selected = pngs[:max_tiles]

    # Open first image to get tile size after crop
    first = Image.open(selected[0]).convert("RGBA")
    first_cropped = center_crop(first, crop)
    tile_w, tile_h = first_cropped.size

    # Prepare canvas (n rows x m cols)
    canvas_w = tile_w * m
    canvas_h = tile_h * n
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))

    # Paste first and remaining
    for idx, p in enumerate(selected):
        try:
            img = Image.open(p).convert("RGBA")
        except Exception as e:
            print(f"[warn] Failed to open {p}: {e}", file=sys.stderr)
            continue
        cropped = center_crop(img, crop)
        if cropped.size != (tile_w, tile_h):
            cropped = cropped.resize((tile_w, tile_h), resample=Image.LANCZOS)

        row = idx // m
        col = idx % m
        canvas.paste(cropped, (col * tile_w, row * tile_h))

    # If fewer images than grid*grid, remaining cells remain black

    # Save as RGBA or convert to RGB if user prefers
    # Save as PNG to preserve transparency if present
    canvas.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser("Combine center-cropped PNGs into an n x n grid")
    parser.add_argument("--in-folder", "-i", type=str, required=True, help="Folder containing .png files")
    parser.add_argument("--out", "-o", type=str, default="combined.png", help="Output combined image path")
    parser.add_argument("--grid", "-g", nargs=2, type=int, default=[4, 4], metavar=("N", "M"), help="Grid size as two ints: N M (rows x cols). Example: -g 4 5")
    parser.add_argument("--crop", "-c", type=float, default=0.8, help="Center crop fraction (0-1], e.g., 0.7 keeps center 70%%)")

    args = parser.parse_args()
    folder = Path(args.in_folder)
    out = Path(args.out)
    # append timestamp to output filename
    ts = time.strftime("%Y%m%d_%H%M%S")
    if out.exists() and out.is_dir():
        out = out / f"combined_{ts}.png"
    else:
        # place timestamp before suffix, keep extension if present
        if out.suffix:
            out = out.with_name(f"{out.stem}_{ts}{out.suffix}")
        else:
            out = out.with_name(f"{out.name}_{ts}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    if not folder.exists() or not folder.is_dir():
        parser.error(f"--in-folder must be an existing directory: {folder}")

    if args.grid[0] <= 0 or args.grid[1] <= 0:
        parser.error("--grid values must be >= 1")

    try:
        path = combine_images(folder, out, n=int(args.grid[0]), m=int(args.grid[1]), crop=args.crop)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Saved combined image: {path}")


if __name__ == "__main__":
    main()
