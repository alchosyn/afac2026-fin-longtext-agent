# -*- coding: utf-8 -*-
"""Numbers in a statement and how to grep them in the documents.

  distinct : a number rare enough to locate its sentence directly
             (3384, 1139.39, 8894.3, 2786, 1.56, 91.1 ...)
  filter   : years and percentages -- too common to search for, only used to
             narrow down when a distinct number hits too many places
  ignore   : small integers such as "20 年", "3 月"
"""
import re

NUM_RE = re.compile(r"\d[\d,，]*(?:\.\d+)?%?")


def classify(raw):
    value = raw.replace(",", "").replace("，", "")
    if value.endswith("%"):
        return "filter", value[:-1]
    if "." in value:
        digits = value.replace(".", "")
        return ("distinct" if len(digits) >= 3 else "filter"), value
    n = int(value)
    if 1900 <= n <= 2099:
        return "filter", value
    if n < 100:
        return "ignore", value
    return "distinct", value


def extract_numbers(text):
    """[(raw, kind, value)] in order of appearance, duplicates removed."""
    out, seen = [], set()
    for m in NUM_RE.finditer(text):
        raw = m.group(0)
        kind, value = classify(raw)
        if value in seen:
            continue
        seen.add(value)
        out.append((raw, kind, value))
    return out


def number_pattern(value):
    """Regex matching the number as printed in a report: optional thousands
    separators, not part of a longer number (2786 must not match 12786 or 2786.5)."""
    if "." in value:
        ip, fp = value.split(".", 1)
    else:
        ip, fp = value, ""
    groups = []
    while len(ip) > 3:
        groups.insert(0, ip[-3:])
        ip = ip[:-3]
    groups.insert(0, ip)
    pat = r"[,，\s]?".join(re.escape(g) for g in groups)
    if fp:
        pat += r"\." + re.escape(fp)
    else:
        pat += r"(?!\.\d)"
    return r"(?<![\d.])" + pat + r"(?!\d)"


def table_rows_with(text, pattern):
    """For a table unit keep the caption, the header row, the separator and only
    the data rows that match the pattern (plus the source line)."""
    rows = text.split("\n")
    start = next((i for i, r in enumerate(rows) if r.startswith("|")), None)
    if start is None or len(rows) < start + 3:
        return text
    rx = re.compile(pattern)
    keep = rows[:start + 2]                       # caption lines, header, separator
    for r in rows[start + 2:]:
        if r.startswith("|"):
            if rx.search(r):
                keep.append(r)
        else:
            keep.append(r)                        # source / note line
    return "\n".join(keep)
