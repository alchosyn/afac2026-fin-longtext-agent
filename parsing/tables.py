# -*- coding: utf-8 -*-
"""Table structure for the MinerU-based build.

MinerU's table_body is an HTML grid whose rows and merged cells are reliable but whose
column boundaries are guessed per table (narrow columns get merged into a neighbour,
decorative bands become empty columns) and whose cell text is model-generated (wrapped
lines interleaved, spaces inside numbers, dropped dashes). This module keeps MinerU's
rows, recomputes the columns from the PDF text layer with a projection profile, and
takes every cell's text from the text layer.

    proj = project(page, bbox_pt)                 # columns + visual lines from the text layer
    rows, status, meta = reconcile(html, proj)    # MinerU rows x projection columns
    status: agree | fixed_merge | fixed_phantom | conflict | empty

The projection: every word of the region is a rectangle; the words are grouped into
visual lines by their vertical centre, then the horizontal axis is scanned for bands
that no (or almost no) line covers -- those bands are the column separators.
"""
import html as htmlmod
import re
from collections import Counter

import pymupdf

ROW_TOL = 3.0        # pt: words whose vertical centres differ by less belong to one visual line
MIN_GAP = 6.0        # pt: a white band must be at least this wide to separate two columns
LEAK = 0.05          # a band may be crossed by at most this share of the lines
EDGE = 3             # pt: bands touching the region edge are not separators

_PUNCT = re.compile(r"[^\w一-鿿%.]")


def norm(s):
    return _PUNCT.sub("", htmlmod.unescape(str(s or ""))).lower()


# ----------------------------------------------------------------------------- MinerU HTML
def html_grid(h):
    """Rows of the HTML table as a matrix with colspan/rowspan expanded; a covered cell
    is None, a real cell is its text. Returns (matrix, n_cols)."""
    rows = re.findall(r"<tr>(.*?)</tr>", h, flags=re.S)
    occupied, matrix = {}, []
    for ri, r in enumerate(rows):
        col, cells = 0, []
        for attr, val in re.findall(r"<td([^>]*)>(.*?)</td>", r, flags=re.S):
            while (ri, col) in occupied:
                col += 1
            cs = int((re.search(r'colspan="(\d+)"', attr) or [None, 1])[1])
            rs = int((re.search(r'rowspan="(\d+)"', attr) or [None, 1])[1])
            text = htmlmod.unescape(re.sub(r"<[^>]+>", "", val)).strip()
            for dr in range(rs):
                for dc in range(cs):
                    occupied[(ri + dr, col + dc)] = text if (dr == 0 and dc == 0) else None
            col += cs
        matrix.append(None)
    n_cols = max((c for _, c in occupied), default=-1) + 1
    out = []
    for ri in range(len(rows)):
        row = [occupied.get((ri, c), "") for c in range(n_cols)]
        out.append(["" if v is None else v for v in row])
    return out, n_cols


def header_rows_of_html(h, max_rows=3):
    """How many leading rows of MinerU's HTML grid are header: the first row, plus the rows a
    header cell spans with rowspan, plus the row of sub-columns under a header cell with
    colspan > 1, plus the row after a title row (one cell spanning the whole width, "公司及其
    子公司对外担保情况"). Decided from the spans, not from the cell text (the text of a code
    table's data rows looks exactly like a header). At most max_rows."""
    rows = re.findall(r"<tr>(.*?)</tr>", h, flags=re.S)
    if len(rows) < 2:
        return 1
    spans = []                                                  # per row: [(colspan, rowspan)]
    for r in rows[:max_rows + 1]:
        cells = re.findall(r"<td([^>]*)>", r)
        spans.append([(int((re.search(r'colspan="(\d+)"', a) or [None, 1])[1]),
                       int((re.search(r'rowspan="(\d+)"', a) or [None, 1])[1])) for a in cells])
    width = max((sum(cs for cs, _ in row) for row in spans), default=1)
    n = 1
    changed = True
    while changed and n < min(max_rows, len(rows) - 1):
        changed = False
        reach = max((ri + rs for ri, row in enumerate(spans[:n]) for _, rs in row), default=n)
        if reach > n:                                           # a header cell spans down into the next row
            n, changed = min(reach, max_rows), True
            continue
        last = spans[n - 1]
        if len(last) == 1 and last[0][0] >= width and width > 1:   # a title row: the real header is below it
            n, changed = n + 1, True
            continue
        if n == 1 and any(cs > 1 for cs, _ in last):           # merged header cells: the sub-columns are the next row
            n, changed = n + 1, True
    return n


def drop_empty_columns(matrix):
    """Remove columns that are empty in every row (MinerU's phantom columns). Returns (matrix, dropped)."""
    if not matrix:
        return matrix, 0
    n = max(len(r) for r in matrix)
    keep = [c for c in range(n) if any(c < len(r) and r[c].strip() for r in matrix)]
    return [[r[c] if c < len(r) else "" for c in keep] for r in matrix], n - len(keep)


# ----------------------------------------------------------------------------- text layer projection
def group_lines(words, tol=ROW_TOL):
    words = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if lines and abs(yc - lines[-1][0]) <= tol:
            lines[-1][1].append(w)
        else:
            lines.append([yc, [w]])
    return [sorted(ws, key=lambda w: w[0]) for _, ws in lines]


def separators(lines, x0, x1, min_gap=MIN_GAP, leak=LEAK):
    """x positions of the column separators: midpoints of the vertical white bands."""
    L, R = int(x0), int(x1) + 1
    cover = [0] * (R - L + 1)
    for ws in lines:
        for w in ws:
            for x in range(max(0, int(w[0]) - L), min(len(cover), int(w[2]) - L + 1)):
                cover[x] += 1
    n = max(1, len(lines))
    gaps, start = [], None
    for i, c in enumerate(cover):
        empty = c <= leak * n
        if empty and start is None:
            start = i
        if (not empty or i == len(cover) - 1) and start is not None:
            if i - start >= min_gap and start > EDGE and i < len(cover) - EDGE:
                gaps.append((start + L, i + L))
            start = None
    return [(a + b) / 2 for a, b in gaps], gaps


def project(page, bbox_pt):
    """Columns and visual lines of a table region from the text layer.
    Returns {"lines": [[cell text per column] ...], "seps": [...], "ranges": [(x0, x1) per column], "n_words": int}."""
    x0, y0, x1, y1 = bbox_pt
    words = page.get_text("words", clip=pymupdf.Rect(x0, y0, x1, y1))
    if not words:
        return {"lines": [], "seps": [], "ranges": [], "n_words": 0}
    lines = group_lines(words)
    seps, _ = separators(lines, x0, x1)
    n_cols = len(seps) + 1

    def col_of(w):
        xc = (w[0] + w[2]) / 2
        return sum(1 for s in seps if xc > s)

    out, cols_words = [], [[] for _ in range(n_cols)]
    for ws in lines:
        cells = [[] for _ in range(n_cols)]
        for w in ws:
            c = col_of(w)
            cells[c].append(w[4])
            cols_words[c].append(w)
        out.append([join_words(c) for c in cells])
    ranges = [(min(w[0] for w in c), max(w[2] for w in c)) if c else (None, None) for c in cols_words]
    return {"lines": out, "seps": seps, "ranges": ranges, "n_words": len(words)}


def join_words(words):
    """Words of one cell on one line: no space between CJK characters, one space otherwise."""
    out = ""
    for w in words:
        if not out:
            out = w
        elif re.search(r"[一-鿿]$", out) or re.match(r"[一-鿿]", w):
            out += w
        else:
            out += " " + w
    return out


# ----------------------------------------------------------------------------- reconciliation
def _chars(s):
    return Counter(norm(s))


def align_lines_to_rows(mineru_rows, lines):
    """Group the visual lines into MinerU's rows by character multisets, in order: a line
    belongs to the current row while its characters fit into what that row still owes.
    Returns a list of line-index groups, one per MinerU row (possibly empty)."""
    groups, li = [], 0
    for row in mineru_rows:
        owed = _chars("".join(row))
        group = []
        while li < len(lines):
            need = _chars("".join(lines[li]))
            if not need:                                   # a blank line: attach to the current row
                group.append(li); li += 1; continue
            if all(owed[c] >= k for c, k in need.items()):
                owed -= need
                group.append(li); li += 1
                if sum(owed.values()) == 0:
                    break
            else:
                break
        groups.append(group)
    if li < len(lines):                                    # leftovers: attach to the last row
        groups[-1] = groups[-1] + list(range(li, len(lines))) if groups else [list(range(li, len(lines)))]
    return groups


def merge_evidence(mineru_rows, lines):
    """True when some MinerU cell equals two or more adjacent projection cells joined --
    the signature of a narrow column merged into its neighbour."""
    cells = {norm(c) for r in mineru_rows for c in r if norm(c)}
    for line in lines:
        vals = [norm(c) for c in line]
        for i in range(len(vals)):
            for j in range(i + 2, min(i + 4, len(vals)) + 1):
                if all(vals[i:j]) and "".join(vals[i:j]) in cells:
                    return True
    return False


def rebuild(mineru_rows, proj):
    """MinerU's rows x the projection's columns, every cell's text from the text layer."""
    groups = align_lines_to_rows(mineru_rows, proj["lines"])
    n_cols = len(proj["seps"]) + 1
    rows = []
    for g in groups:
        cells = ["" for _ in range(n_cols)]
        for li in g:
            for c, t in enumerate(proj["lines"][li]):
                if t:
                    cells[c] = join_words([cells[c], t]) if cells[c] else t
        rows.append(cells)
    return rows


MIN_UNSCRAMBLE = 4   # characters: a shorter cell cannot be read as interleaved lines


def _stack_readings(words, target):
    """Readings of a cell from stacked text-layer words: the words are grouped into visual
    lines and into column bands (words that overlap horizontally), and every window of
    consecutive lines of one band whose characters are exactly the cell's gives one reading,
    the lines read top to bottom. Returns the distinct readings."""
    lines = group_lines(words)
    parent = list(range(len(words)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if min(words[i][2], words[j][2]) - max(words[i][0], words[j][0]) > 0:
                parent[find(i)] = find(j)
    bands = {}
    for i, w in enumerate(words):
        bands.setdefault(find(i), set()).add(id(w))
    readings = {}                                           # normalised reading -> (reading, is one text-layer word)
    for members in bands.values():
        band_lines = [[w for w in line if id(w) in members] for line in lines]
        band_lines = [l for l in band_lines if l]
        for s in range(len(band_lines)):
            have = Counter()
            for e in range(s, len(band_lines)):
                have += Counter(norm("".join(w[4] for w in band_lines[e])))
                if any(have[c] > target[c] for c in have):
                    break
                if have == target:
                    text = join_words([join_words([w[4] for w in l]) for l in band_lines[s:e + 1]])   # lines joined like words: no space inside CJK
                    single = e == s and len(band_lines[s]) == 1
                    prev = readings.get(norm(text))
                    if prev is None or (single and not prev[1]):
                        readings[norm(text)] = (text, single)
                    break
    return readings


def unscramble_cells(rows, page, bbox_pt):
    """Cells whose wrapped lines MinerU's native table path read side by side ("正常存续" ->
    "正续常存", "认缴24,750 / 实缴24,750" -> "实认缴缴2244,,775500": ~1,000 cells in 175 tables,
    2026-09-30) re-read from the text layer, for the conflict tables the projection could not
    rebuild. A cell is replaced only when exactly one reading has its characters: the
    text-layer words whose characters all belong to the cell, stacked in one column band over
    consecutive lines, read top to bottom; readings that differ only in spacing are one, the
    same word elsewhere in the table gives the same reading, and a reading that is a single
    text-layer word wins over stacked ones (fr_catl_2024 p67 "间排接放": the word "间接排放" beats
    the two-line stacks "间接 / 排放" and "排放 / 间接" of neighbouring rows). Two stacks reading
    differently stay ambiguous (p67 "宁德区厂": "宁德厂 / 区" and the previous row's "区 / 宁德厂"),
    as does a cell the text layer holds as scattered single characters (p83 "2年-5年"): no
    tie-break by the cell's first character, which would rewrite that correct cell. A cell whose
    reading has the same character order is left alone. Returns (rows, cells checked, cells
    restored, cells ambiguous)."""
    x0, y0, x1, y1 = bbox_pt
    words = [w for w in page.get_text("words", clip=pymupdf.Rect(x0, y0, x1, y1)) if norm(w[4])]
    if not words:
        return rows, 0, 0, 0
    wchars = [Counter(norm(w[4])) for w in words]
    checked = restored = ambiguous = 0
    out = []
    for row in rows:
        new_row = []
        for cell in row:
            seq = norm(cell)
            if len(seq) < MIN_UNSCRAMBLE:
                new_row.append(cell)
                continue
            target = Counter(seq)
            cand = [words[i] for i, wc in enumerate(wchars) if all(target[c] >= k for c, k in wc.items())]
            if not cand:
                new_row.append(cell)
                continue
            checked += 1
            readings = _stack_readings(cand, target)
            single = [v[0] for v in readings.values() if v[1]]
            pick = single[0] if len(single) == 1 else next(iter(readings.values()))[0] if len(readings) == 1 else None
            if pick is not None:
                if norm(pick) != seq:
                    new_row.append(pick)
                    restored += 1
                    continue
            elif len(readings) > 1:
                ambiguous += 1
            new_row.append(cell)
        out.append(new_row)
    return out, checked, restored, ambiguous


def reconcile(html, proj):
    """Decide the table structure. Returns (rows, status, meta)."""
    matrix, m_cols = html_grid(html)
    matrix, dropped = drop_empty_columns(matrix)
    m_eff = max((len(r) for r in matrix), default=0)
    p_cols = len(proj["seps"]) + 1 if proj["lines"] else 0
    meta = {"mineru_cols": m_cols, "mineru_cols_effective": m_eff, "projection_cols": p_cols,
            "mineru_rows": len(matrix), "projection_lines": len(proj["lines"]), "phantom_dropped": dropped}
    if not matrix:
        return [], "empty", meta
    if not proj["lines"]:
        return matrix, "no_text_layer", meta
    if p_cols == m_eff:
        return rebuild(matrix, proj), ("fixed_phantom" if dropped else "agree"), meta
    if p_cols > m_eff and merge_evidence(matrix, proj["lines"]):
        return rebuild(matrix, proj), "fixed_merge", meta
    return matrix, "conflict", meta


# ----------------------------------------------------------------------------- output forms
def to_markdown(rows):
    if not rows:
        return ""
    n = max(len(r) for r in rows)
    rows = [[c.replace("|", "／").replace("\n", " ") for c in r] + [""] * (n - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def is_layout_table(rows, max_cols=3, min_avg_chars=25, long_cell=60):
    """A grid used to lay out prose rather than data. Three signatures:
      * a single column
      * at most two columns with hardly any numeric cells (two-column lists)
      * at most three columns where one column holds long text (number | title | clause text)"""
    if not rows:
        return False
    n = max(len(r) for r in rows)
    cells = [c for r in rows for c in r if c.strip()]
    if not cells:
        return False
    if n == 1:
        return True
    numeric = sum(1 for c in cells if re.fullmatch(r"[\d.,%（）()\-－—/\s]+", c))
    if n <= 2 and numeric / len(cells) < 0.1:
        return True
    if n <= max_cols:
        for col in range(n):
            texts = [r[col] for r in rows if col < len(r) and r[col].strip()]
            if texts and sum(len(t) for t in texts) / len(texts) >= long_cell:
                return True
        avg = sum(len(c) for c in cells) / len(cells)
        if n <= 2 and avg >= min_avg_chars:
            return True
    return False


def rows_to_paragraphs(rows):
    """Cells of a layout table read row by row, one paragraph per row."""
    return [" ".join(c for c in r if c.strip()) for r in rows if any(c.strip() for c in r)]


def same_columns(ranges_a, ranges_b, min_overlap=0.5):
    """Two tables share the physical column layout when the text extents of every column overlap."""
    if len(ranges_a) != len(ranges_b) or not ranges_a:
        return False
    for (a0, a1), (b0, b1) in zip(ranges_a, ranges_b):
        if None in (a0, b0):
            continue
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        if inter / max(1e-6, min(a1 - a0, b1 - b0)) < min_overlap:
            return False
    return True


# ----------------------------------------------------------------------------- tables MinerU left as text
# A borderless table the layout model did not detect comes out as text blocks: either one
# block per cell (numbers scattered over the page) or one block per column (the cells of
# a column glued into a string). Runs of punctuation-free text blocks are the candidates, the
# text-layer projection rebuilds the grid, and validate_grid decides whether it is a table at all.
SENT_PUNCT = re.compile(r"[。，；：;:]")
TRIGGER = re.compile(r"(如下|列示如下|情况如下|见下表|详见下表|如表\s*[\d一二三四五六七八九十]+|见表\s*[\d一二三四五六七八九十]+)\s*[:：]?\s*$")
UNIT_LINE = re.compile(r"^\s*(单位|币种|金额单位)\s*[:：]")
NUM_CELL = re.compile(r"^[(（]?[-−+]?\d[\d,，]*(\.\d+)?%?[)）]?$")
DATE_UNITS = {"年", "月", "日"}


def cell_like(text):
    """A text block that could be table cells: not empty and without sentence punctuation."""
    return bool(text.strip()) and not SENT_PUNCT.search(text)


def group_rows(lines):
    """Visual lines -> table rows. The anchor column is the leftmost column with text in
    at least two lines (the row labels). A line whose anchor cell is empty continues the
    previous row (wrapped cells, stacked header lines), except when that row holds nothing
    but its anchor cell (a label line such as "2024 年" printed above the header)."""
    if not lines:
        return []
    n = max(len(l) for l in lines)
    lines = [list(l) + [""] * (n - len(l)) for l in lines]
    anchor = next((c for c in range(n) if sum(1 for l in lines if l[c].strip()) >= 2), 0)
    rows = []
    for l in lines:
        if l[anchor].strip() or not rows:
            rows.append(l)
            continue
        prev = rows[-1]
        if not any(prev[c].strip() for c in range(n) if c != anchor):
            rows.append(l)                                  # a bare label row: start a new row
            continue
        rows[-1] = [join_words([a, b]) if a and b else (a or b) for a, b in zip(prev, l)]
    return rows


def strip_label_rows(rows):
    """Leading rows with a single non-empty cell ("本集团", "2024 年") are labels, not data.
    Returns (rows, label text)."""
    labels = []
    while rows and sum(1 for c in rows[0] if c.strip()) == 1 and len(rows) > 1:
        labels.append(next(c for c in rows[0] if c.strip()))
        rows = rows[1:]
    return rows, " ".join(labels)


GLYPH_CELL = re.compile(r"^[\s•●■◆▪✿❖☑☐□√✓✔×-]+$")     # bullets, checkboxes, Wingdings private-use glyphs


def validate_grid(rows, allow_text_pairs=False, underline=False):
    """Is the rebuilt grid a table? At least 2 x 2, no cell that is only a bullet or a checkbox,
    and one of: a column that is mostly numbers, three or more columns, an underlined header,
    or (when a heading / trigger sentence introduces it) a two-column list of short labels and
    their values. A glyph-only cell is a bulleted list or a checkbox line laid out in columns,
    never a table (fr_catl_2024 p144 "• 其他应收款组合1", fr_cscec_2025 p36 / p38 infographic
    bullets, pack2_text03 p1 "❖ ..." summary lines): 2026-09-30 it marked 5 of the 7 false
    recovered tables and none of the 21 true ones."""
    if not rows:
        return False, "empty"
    n_cols = max(len(r) for r in rows)
    n_rows = len(rows)
    if n_cols < 2 or n_rows < 2:
        return False, f"shape {n_rows}x{n_cols}"
    if any(GLYPH_CELL.match(c) for r in rows for c in r if c.strip()):
        return False, "glyph_cells"
    numeric = False
    for c in range(n_cols):
        cells = [r[c] for r in rows if c < len(r) and r[c].strip()]
        if cells and sum(1 for x in cells if NUM_CELL.match(x.replace(" ", ""))) / len(cells) >= 0.5:
            numeric = True
    filled = sum(1 for r in rows if sum(1 for c in r if c.strip()) >= 2)
    if filled < 2:
        return False, "rows_with_one_cell"
    head = [c.strip() for c in rows[0] if c.strip()]
    if head and sum(1 for c in head if c in DATE_UNITS) >= len(head) / 2:
        return False, "date_line"                            # "年 月 日" over "2026 4 21": a signature date, not a table
    if numeric:
        return True, f"{n_rows}x{n_cols} numeric"
    if n_cols >= 3:
        return True, f"{n_rows}x{n_cols} wide"
    if underline:
        return True, f"{n_rows}x{n_cols} underline"
    if allow_text_pairs and n_cols == 2 and n_rows >= 3:
        left = [len(r[0]) for r in rows if r[0].strip()]
        right = [len(r[1]) for r in rows if len(r) > 1 and r[1].strip()]
        # a label | description list: short labels, descriptions clearly longer (a TOC in two
        # columns or a signature block has short cells on both sides)
        if left and right and max(left) <= 20 and sum(right) / len(right) >= 12 and sum(right) / len(right) > sum(left) / len(left):
            return True, f"{n_rows}x{n_cols} text_pairs"
    return False, f"{n_rows}x{n_cols} text_only"


REGION_TOL = 0.005   # page fraction: a text block this close to the run's vertical extent lies inside the region


def find_table_candidates(content, page, W, H, text_of):
    """Candidate borderless tables among a page's blocks (MinerU blocks without the
    header / footer / page_number / index ones, in reading order).

    One signal proposes runs of positions in `content`: two or more consecutive text blocks
    without sentence punctuation (a trigger sentence "...如下:" / "见下表" followed by text
    blocks was a second signal until 2026-10-01; measured 2026-09-30 it recovered no table
    the run signal missed and produced the fr_catl_2024 p144 false one, so it is now only a
    licence for two-column text pairs, like a heading before the run). Every text block whose
    box lies inside the run's vertical extent is a member too: its words are in the projection
    anyway, and a member outside the run survived as a duplicate paragraph beside the table
    (fr_byd_2024 p203 row label, fr_byd_2025 p192, fr_cscec_2025 p12). Each run is projected
    over the full text width, its rows are grouped, and validate_grid decides. Overlapping
    runs: the larger is tried first and a smaller one is skipped once a larger one containing
    it passed.

    Returns a list of dicts in evaluation order:
        run (sorted positions), trig (always False, kept for the audit), bbox (page fractions),
        rows, label, ok, why, underline
    Nothing is modified; callers replace blocks (the builder) or report (the audit)."""
    if len(content) < 2:
        return []
    xs = [b["bbox"][0] for b in content]
    xe = [b["bbox"][2] for b in content]
    x0, x1 = min(xs), max(xe)

    candidates = []
    i = 0
    while i < len(content):                            # punctuation-free runs
        b = content[i]
        if b["type"] != "text" or not cell_like(text_of[id(b)]):
            i += 1
            continue
        j = i
        while j < len(content) and content[j]["type"] == "text" and (cell_like(text_of[id(content[j])]) or UNIT_LINE.match(text_of[id(content[j])])):
            j += 1
        run = [k for k in range(i, j) if not UNIT_LINE.match(text_of[id(content[k])])]
        if len(run) >= 2:
            candidates.append(set(run))
        i = j

    out, accepted, accepted_y = [], set(), []
    for run in sorted(candidates, key=lambda c: (-len(c), min(c))):
        if run & accepted:
            continue
        members = [content[k] for k in sorted(run)]
        y0 = min(m["bbox"][1] for m in members)
        y1 = max(m["bbox"][3] for m in members)
        if any(min(y1, b1) - max(y0, b0) >= 0.5 * max(1e-6, y1 - y0) for b0, b1 in accepted_y):
            continue                                        # the same region already passed as another run
        inside = {k for k, b in enumerate(content) if b["type"] == "text"
                  and b["bbox"][1] >= y0 - REGION_TOL and b["bbox"][3] <= y1 + REGION_TOL}
        run = run | inside
        members = [content[k] for k in sorted(run)]
        proj = project(page, (x0 * W, y0 * H - 2, x1 * W, y1 * H + 2))
        rows, label = strip_label_rows(group_rows(proj["lines"]))
        underline = any("underline" in (s.get("styles") or []) for m in members for s in m.get("content", []) if isinstance(s, dict))
        first = sorted(run)[0]
        prev = content[first - 1] if first > 0 else None
        before = prev is not None and (prev["type"] in ("doc_title", "paragraph_title")
                                       or (prev["type"] == "text" and len(text_of[id(prev)]) <= 120 and bool(TRIGGER.search(text_of[id(prev)]))))
        ok, why = validate_grid(rows, allow_text_pairs=before, underline=underline)
        if ok:
            accepted |= run
            accepted_y.append((y0, y1))
        out.append({"run": sorted(run), "trig": False, "bbox": [x0, y0, x1, y1], "rows": rows, "label": label,
                    "ok": ok, "why": why, "underline": underline})
    return out


def rows_to_html(rows):
    """A plain <table> for the rebuilt grid, in the shape MinerU's table_body uses."""
    n = max(len(r) for r in rows)
    out = ["<table>"]
    for r in rows:
        r = list(r) + [""] * (n - len(r))
        out.append("<tr>" + "".join(f"<td>{htmlmod.escape(c)}</td>" for c in r) + "</tr>")
    out.append("</table>")
    return "".join(out)
