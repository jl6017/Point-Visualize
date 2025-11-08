"""Combine tiled PNGs into a single large image.

This module provides `combine_pngs(folder, out_path, pad=0, fill=(0,0,0))` which:
1. Lists all *.png files in `folder`.
2. Parses row and column indices from filenames like `r00_c00.png` (case-insensitive).
3. Places each tile into a grid and writes `out_path`.

If tiles have varying sizes the function uses the maximum tile width/height and positions each
tile at (col * tile_w, row * tile_h). Missing tiles are filled with the `fill` color.

Example usage:
	python combine_png.py --folder tiles_dir --out combined.png

"""

from pathlib import Path
import re
from PIL import Image
from typing import Tuple, Optional
import argparse
import sys
import time
import os


COORD_REGEXES = [
	re.compile(r"r(?P<r>\d+)[^0-9]*c(?P<c>\d+)", re.IGNORECASE),
	re.compile(r"row[_-]?(?P<r>\d+)[^0-9]*col[_-]?(?P<c>\d+)", re.IGNORECASE),
	re.compile(r"(?P<r>\d+)[^0-9]+(?P<c>\d+)")  # fallback: first two numbers
]


def parse_row_col(filename: str) -> Optional[Tuple[int, int]]:
	"""Parse row and column indices from a filename.

	Returns (row, col) as integers, or None if parsing fails.
	"""
	name = Path(filename).stem
	for rx in COORD_REGEXES:
		m = rx.search(name)
		if m:
			try:
				r = int(m.group('r'))
				c = int(m.group('c'))
				return r, c
			except Exception:
				# fallback for unnamed groups in the last regex
				groups = [g for g in m.groups() if g is not None]
				if len(groups) >= 2:
					return int(groups[0]), int(groups[1])
	return None


def combine_pngs(folder: str, out_path: str, pad: int = 0, fill=(0, 0, 0), scale: float = 1.0) -> str:
	"""Combine PNG tiles in `folder` into `out_path`.

	Args:
		folder: directory containing PNG tiles.
		out_path: output file path for combined PNG.
		pad: number of pixels padding between tiles (applies as spacing).
		fill: background color (tuple like (R,G,B) or (R,G,B,A)).

	Returns:
		The path to the written combined image.
	"""
	folder_p = Path(folder)
	if not folder_p.exists() or not folder_p.is_dir():
		raise ValueError(f"Folder not found: {folder}")

	pngs = sorted(folder_p.glob('*.png'))
	if not pngs:
		raise ValueError(f"No PNG files found in folder: {folder}")

	tiles = {}
	max_r = max_c = -1
	max_w = max_h = 0

	# load images and parse coordinates
	for p in pngs:
		rc = parse_row_col(p.name)
		if rc is None:
			# skip files that don't parse
			continue
		r, c = rc
		im = Image.open(p).convert('RGBA')
		w, h = im.size
		max_w = max(max_w, w)
		max_h = max(max_h, h)
		tiles[(r, c)] = im
		max_r = max(max_r, r)
		max_c = max(max_c, c)

	if not tiles:
		raise ValueError('No PNG tiles matched the expected filename pattern (e.g. r00_c00.png)')

	rows = max_r + 1
	cols = max_c + 1

	out_w = cols * max_w + (cols - 1) * pad
	out_h = rows * max_h + (rows - 1) * pad

	# determine mode from fill
	if len(fill) == 4:
		mode = 'RGBA'
	else:
		mode = 'RGB'

	out_img = Image.new(mode, (out_w, out_h), color=fill)

	for (r, c), im in tiles.items():
		x = c * (max_w + pad)
		y = r * (max_h + pad)
		# if tile smaller than cell, paste centered (top-left alignment could be used instead)
		w, h = im.size
		paste_x = x
		paste_y = y
		out_img.paste(im, (paste_x, paste_y), im)

	out_path_p = Path(out_path)
	# If scaling requested, resize the final image (scale < 1 for downscale, >1 to upscale)
	if scale is None:
		scale = 1.0
	if scale != 1.0:
		if scale <= 0:
			raise ValueError('Scale must be > 0')
		new_w = max(1, int(round(out_w * scale)))
		new_h = max(1, int(round(out_h * scale)))
		# Use high-quality downsampling filter
		out_img = out_img.resize((new_w, new_h), resample=Image.LANCZOS)

	out_img.save(out_path_p)
	return str(out_path_p)


def _cli(argv=None):
	ts = time.strftime("%Y%m%d_%H%M%S")
	p = argparse.ArgumentParser(description='Combine tiled PNG images into one large PNG.')
	p.add_argument('--folder', '-f', required=True, help='Folder containing PNG tiles')
	p.add_argument('--out', '-o', default=os.path.join('out', f'combined_{ts}.png'), help='Output combined PNG file')
	p.add_argument('--pad', type=int, default=0, help='Pixel padding between tiles')
	p.add_argument('--fill', default=None, help='Background fill color as R,G,B or R,G,B,A (e.g. 0,0,0)')
	p.add_argument('--scale', type=float, default=1.0, help='Scale factor for the final image (e.g. 0.25 for 1/4 resolution)')
	args = p.parse_args(argv)

	fill = (0, 0, 0)
	if args.fill:
		parts = [int(x) for x in args.fill.split(',')]
		fill = tuple(parts)

	try:
		out = combine_pngs(args.folder, args.out, pad=args.pad, fill=fill, scale=args.scale)
		print(f'Wrote combined image to: {out}')
	except Exception as e:
		print('Error:', e)
		sys.exit(2)


if __name__ == '__main__':
	_cli()

