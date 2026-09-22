"""Reproducible, approximate English lexical counts from two-column PDFs.

Counts are layout counts, not TeXcount prose counts: captions, algorithm text,
subheadings and alphabetic table labels are included. References and keywords
are excluded. Formula symbols/numerals are mostly excluded by the token rule.
Review extracted files for column-order and section-boundary errors.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent
TOKEN = re.compile(r"(?<![A-Za-z])[A-Za-z]{2,}(?:[-'][A-Za-z]+)*|\b[aAI]\b")
HEADING = re.compile(r"^([IVX]+)\.\s+([A-Z][A-Z ,&:()\-]+)$")
# Manually checked funding/affiliation footnotes in the archived extraction.
EXCLUDED_LINES = {"tang2023": range(42, 65), "jiang2024": range(39, 45),
                  "zhao2024": range(39, 42), "zhao_noma2024": range(42, 45)}


def count(text):
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(r"(?<=[a-z])-\s*\n\s*(?=[a-z])", "", text)
    return len(TOKEN.findall(text))


def extract(path):
    doc = fitz.open(path)
    lines = []
    for i, page in enumerate(doc):
        top = 25
        if i == 0:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    if "Abstract" in "".join(s["text"] for s in line["spans"]):
                        top = line["bbox"][1] - 1
                        break
        mid = page.rect.width / 2
        for left, right in ((40, mid), (mid, page.rect.width - 30)):
            text = page.get_text("text", clip=fitz.Rect(left, top, right, page.rect.height - 25), sort=False)
            for line in text.splitlines():
                line = unicodedata.normalize("NFKC", line).strip()
                if line and not re.match(r"^(?:979-|978-|arXiv:)", line):
                    lines.append((i + 1, line))
    return doc, lines


def analyze(path):
    doc, lines = extract(path)
    buckets = {}
    starts = {}
    active = None
    for i, (page, line) in enumerate(lines):
        if i + 1 in EXCLUDED_LINES.get(path.stem, ()):
            continue
        if line.startswith("Abstract"):
            active = "Abstract"
            starts[active] = {"page": page, "line": i + 1}
            buckets[active] = [re.sub(r"^Abstract\s*[—–-]?\s*", "", line)]
        elif line.startswith("Index Terms"):
            active = None
        elif line in {"REFERENCES", "ACKNOWLEDGMENT", "ACKNOWLEDGMENTS"}:
            break
        elif HEADING.fullmatch(line):
            active = line
            starts[active] = {"page": page, "line": i + 1}
            buckets[active] = []
        elif active:
            buckets[active].append(line)
    out = ROOT / "extracted" / f"{path.stem}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(f"{n:04d} P{page} {line}" for n, (page, line) in enumerate(lines, 1)) + "\n")
    return {"file": str(path), "pages": len(doc), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sections": [{"section": name, "words_approx": count("\n".join(text)), **starts[name]}
                         for name, text in buckets.items()]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="*")
    parser.add_argument("--output", default="section_counts.json")
    args = parser.parse_args()
    paths = [Path(p) for p in args.pdfs] if args.pdfs else sorted((ROOT / "pdfs").glob("*.pdf"))
    results = [analyze(p) for p in paths]
    (ROOT / args.output).write_text(json.dumps(results, indent=2) + "\n")
    for item in results:
        print(Path(item["file"]).name, "pages=", item["pages"])
        for section in item["sections"]:
            print(f"  {section['words_approx']:4d}  p{section['page']}  {section['section']}")
