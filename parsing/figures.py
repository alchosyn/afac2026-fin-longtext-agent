# -*- coding: utf-8 -*-
"""Figure images of the structure layer.

When the structure of a document is built (parsing.structure, or script/render_figures.py
for a structure built earlier), every captioned figure is cut out of the PDF page at
FIGURE_DPI with a small pad and saved as processed_data/<doc_id>/figures/<local id>.png;
the figure's structure record carries the path (field `image`, relative to the repository
root), the figure chunk copies the pointer, and a retrieved figure chunk hands the file to
the multimodal judge (agent.figures). Nothing is generated from the image at build time:
the file is the crop and nothing else.

    from parsing.figures import render_figure, save_figure
    png = render_figure(page, bbox)                      # bytes, or None for a degenerate box
    rel = save_figure(page, bbox, "pack2_text01", "f01") # "processed_data/pack2_text01/figures/f01.png"
"""
from pathlib import Path

import pymupdf

from .common import PROC

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DPI = 150      # the judge read 9 of 10 hand-checked charts at this resolution (2026-09-24)
PAD_PT = 4            # points of page around the figure box, so edge labels are not clipped
MIN_SIDE = 4          # points: a thinner box is a parsing artefact, not a figure


def figure_rect(page, bbox, pad=PAD_PT):
    """The clip rectangle in page points of a box given in page fractions (the MinerU
    builds) or in page points (the legacy builds: integers, pymupdf4llm's coordinates)."""
    W, H = page.rect.width, page.rect.height
    x0, y0, x1, y1 = bbox
    if max(bbox) > 1.5:                                   # points already
        return pymupdf.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad) & page.rect
    return pymupdf.Rect(x0 * W - pad, y0 * H - pad, x1 * W + pad, y1 * H + pad) & page.rect


def render_figure(page, bbox, dpi=FIGURE_DPI, pad=PAD_PT):
    """PNG bytes of the box cut from the page; None when the box is degenerate."""
    rect = figure_rect(page, bbox, pad)
    if rect.is_empty or rect.width < MIN_SIDE or rect.height < MIN_SIDE:
        return None
    return page.get_pixmap(dpi=dpi, clip=rect).tobytes("png")


def figure_path(doc_id, local_id):
    return PROC / doc_id / "figures" / f"{local_id}.png"


def save_figure(page, bbox, doc_id, local_id, dpi=FIGURE_DPI, pad=PAD_PT, force=False):
    """Write the figure's PNG unless it exists (force=True rewrites). Returns the path
    relative to the repository root, or None when the box is degenerate."""
    path = figure_path(doc_id, local_id)
    if not path.exists() or force:
        png = render_figure(page, bbox, dpi, pad)
        if png is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
    return str(path.relative_to(ROOT)).replace("\\", "/")
