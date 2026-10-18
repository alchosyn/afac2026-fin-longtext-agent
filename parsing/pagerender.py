# -*- coding: utf-8 -*-
"""Render PDF pages to PNG once, for the layout models that take page images
(PaddleOCR PP-StructureV3, dots.ocr). Pure pymupdf, so every virtual environment can use it.

    render_pages(pdf_path, out_dir, dpi=200, pages=None) -> [(page_no, png_path, width_px, height_px)]

page_no is 1-based; pages=None renders every page. Existing PNGs are reused (size read from
the file header), so several runners can share one rendering."""
import struct
from pathlib import Path

import pymupdf


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def render_pages(pdf_path, out_dir, dpi=200, pages=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    wanted = list(range(1, doc.page_count + 1)) if pages is None else sorted(set(int(p) for p in pages))
    out = []
    zoom = dpi / 72.0
    for pno in wanted:
        if pno < 1 or pno > doc.page_count:
            continue
        png = out_dir / f"p{pno:04d}.png"
        if not png.exists():
            pix = doc[pno - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            pix.save(str(png))
            w, h = pix.width, pix.height
        else:
            w, h = png_size(png)
        out.append((pno, png, w, h))
    return out
