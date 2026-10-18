# -*- coding: utf-8 -*-
"""Export the native PDF data MinerU reads (docvortex over pypdfium2) to
processed_data/<doc_id>/native.json. MinerU builds this per page while parsing (it fills the
layout boxes with these characters, classifies the text layer as usable or not, reads ruling
lines for tables) but its exports (markdown, middle JSON, zip) keep only block boxes and text.

Run with the MinerU venv's python, which has docvortex (the project python does not):
    <mineru_venv>/Scripts/python.exe script/export_native.py fc_text02 [doc_id ...]

native.json (coordinates in PDF points, origin at the top-left of the page as docvortex gives them):
  doc_id, source_pdf, versions {mineru, docvortex, pypdfium2}, classify ("txt": text layer usable,
  "ocr": MinerU would OCR the document), fonts: [[name, size, weight, flags], ...],
  pages: [{page (1-based), size [w, h], rotation, n_chars,
           lines: [{bbox, rotation, spans: [{text, bbox, font (index into fonts), mode (text render
                   mode of all its chars, or a list per char when they differ; 3 = invisible,
                   the OCR layer of a scan), chars: [[x0, y0, x1, y1] per character of text],
                   sup / sub / url only when set}]}],
           drawing_lines: [[x0, y0, x1, y1, width, "h" | "v"]]   visible horizontal / vertical lines,
           paths: [[x0, y0, x1, y1, segments, fill 0/1, stroke 0/1, rgba | null, form_depth]],
           images: [[x0, y0, x1, y1, fingerprint]], forms, signatures: [[x0, y0, x1, y1]],
           links: [{target, bboxes}]}]
A span's text equals the concatenation of its characters (checked while writing: mismatches are
counted in the summary line and the span keeps no char boxes).
"""
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from docvortex.document.pdf import PDFDocument  # noqa: E402


def r2(v):
    return round(float(v), 2)


def box(b):
    b = getattr(b, "bbox", b)
    return [r2(x) for x in b]


def export(doc_id, pdf_path):
    t0 = time.time()
    doc = PDFDocument(str(pdf_path))
    fonts, font_idx = [], {}
    stats = {"pages": 0, "chars": 0, "spans": 0, "span_char_mismatch": 0, "paths": 0, "drawing_lines": 0, "images": 0}

    def font_of(f):
        key = (f.get("name"), r2(f.get("size") or 0), f.get("weight"), f.get("flags"))
        if key not in font_idx:
            font_idx[key] = len(fonts)
            fonts.append(list(key))
        return font_idx[key]

    pages = []
    for i in range(doc.page_count):
        lines = []
        for line in doc.get_page_lines(i):
            spans = []
            for s in line["spans"]:
                chars = s.get("chars") or []
                modes = [c.get("text_render_mode") for c in chars]
                sp = {"text": s["text"], "bbox": box(s["bbox"]), "font": font_of(s.get("font") or {}),
                      "mode": modes[0] if len(set(modes)) <= 1 and modes else (modes or None)}
                if s["text"] == "".join(c["char"] for c in chars):
                    sp["chars"] = [box(c["bbox"]) for c in chars]
                else:
                    stats["span_char_mismatch"] += 1
                for k_out, k_in in (("sup", "superscript"), ("sub", "subscript"), ("url", "url")):
                    if s.get(k_in):
                        sp[k_out] = s[k_in]
                spans.append(sp)
                stats["spans"] += 1
                stats["chars"] += len(chars)
            lines.append({"bbox": box(line["bbox"]), "rotation": r2(line.get("rotation") or 0), "spans": spans})
        vg = doc.get_page_vector_geometry(i)
        dl = [box(l.bbox) + [r2(l.width), l.orientation[0]] for l in vg.drawing_lines]
        paths = [box(p.bbox) + [p.segment_count, int(p.fill_visible), int(p.stroke_visible),
                                list(p.fill_rgba) if p.fill_rgba else None, p.form_depth] for p in vg.path_infos]
        images = [box(im.bbox) + [im.fingerprint] for im in doc.get_page_image_infos(i)]
        pages.append({"page": i + 1, "size": [r2(x) for x in doc.page_size(i)], "rotation": doc.page_rotation(i),
                      "n_chars": doc.page_char_count(i), "lines": lines, "drawing_lines": dl, "paths": paths,
                      "images": images, "forms": [box(b) for b in doc.get_page_form_bboxes(i)],
                      "signatures": [box(b) for b in doc.get_page_signature_bboxes(i)],
                      "links": [{"target": a.target, "bboxes": [box(b) for b in a.bboxes]} for a in doc.get_page_link_annotations(i)]})
        stats["pages"] += 1
        stats["paths"] += len(paths)
        stats["drawing_lines"] += len(dl)
        stats["images"] += len(images)
    out = {"doc_id": doc_id, "source_pdf": str(pdf_path.relative_to(ROOT)).replace("\\", "/"),
           "versions": {k: version(k) for k in ("mineru", "docvortex", "pypdfium2")},
           "classify": doc.classify(), "fonts": fonts, "pages": pages}
    path = PROC / doc_id / "native.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    doc.close()
    mb = path.stat().st_size / 1e6
    print(f"{doc_id}: {stats} classify={out['classify']} fonts={len(fonts)} -> {path} ({mb:.1f} MB, {time.time() - t0:.0f}s)", flush=True)


def main(argv):
    cards = {json.loads(l)["doc_id"]: json.loads(l) for l in (PROC / "docs.jsonl").open(encoding="utf-8")}
    for doc_id in argv:
        export(doc_id, ROOT / cards[doc_id]["source_pdf"])


if __name__ == "__main__":
    main(sys.argv[1:])
