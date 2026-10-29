# -*- coding: utf-8 -*-
"""The MinerU-based document pipeline, one module per stage:

    mineru.json (untouched)
        -> parsing.audit    quality gate 1: report only (coverage, numbering, mislabels, table candidates)
        -> parsing.repair   block-level fixes in memory (headings the audit found among text blocks,
                           text the layout model left outside every block, tables MinerU left as text ...)
        -> parsing.structure heading tree, paragraphs, tables, figures -> elements.jsonl (the structure layer)
        -> parsing.chunk    retrieval units over elements.jsonl: clauses as they are, paragraphs packed,
                           long tables split with their header repeated -> chunks.jsonl
    parsing.tables  text-layer projection, MinerU-grid reconciliation, candidate tables, interleaved
                    cells re-read from the text layer (shared)
    parsing.native  native.json, page kinds (scan / digital), RapidOCR, scan-table number check
    parsing.figures every figure cut out of the PDF at build time -> processed_data/<doc>/figures/<id>.png,
                    the path recorded on the figure record and copied into its chunk; the text inside a
                    figure (text layer, else cached OCR) -> labels, numbers, and the first line as caption
                    when the hand review calls it a title (script/render_figures.py does the same for
                    structures built earlier)
    parsing.rerun   MinerU run again on a cropped region (an image captioned 表N that is a table)
    parsing.outline the heading list (bookmarks, else the printed TOC) -> docs.json
    parsing.common  MinerU block helpers, numbering styles, document cards

Command-line entry points live in script/: audit_doc.py (gate 1) and build_elements_mineru.py
(repair + chunk). The earlier pymupdf4llm route and the regulatory / txt builders are kept
under experiments/archive/legacy_build.
"""
