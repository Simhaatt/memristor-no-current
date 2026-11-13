"""Inspect an external memristor dataset for no-current sequence suitability.

This is a lightweight data-hunt helper. Point it at a manually downloaded
dataset folder; it extracts zip files into a local `_extracted` folder and
summarizes likely tabular V/I/time files.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".csv", ".txt", ".tsv", ".dat"}
TABLE_SUFFIXES = TEXT_SUFFIXES | {".xlsx", ".xls"}
ARCHIVE_SUFFIXES = {".zip"}
VOLTAGE_HINTS = ("voltage", "volt", " v", "v_", "bias")
CURRENT_HINTS = ("current", "curr", " i", "i_", "amp")
TIME_HINTS = ("time", "timestamp", "sample", "index", "step")


def extract_archives(root: Path) -> list[Path]:
    extracted: list[Path] = []
    for archive_path in root.rglob("*"):
        if archive_path.is_file() and archive_path.suffix.lower() in ARCHIVE_SUFFIXES:
            target = archive_path.with_name(archive_path.stem + "_extracted")
            target.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(target)
                extracted.append(target)
            except zipfile.BadZipFile:
                pass
    return extracted


def sniff_delimiter(path: Path) -> str:
    sample = path.read_text(errors="ignore", encoding="utf-8")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
        return dialect.delimiter
    except csv.Error:
        if "\t" in sample:
            return "\t"
        if ";" in sample:
            return ";"
        return ","


def inspect_text_table(path: Path, max_rows: int = 2000) -> dict[str, Any]:
    delimiter = sniff_delimiter(path)
    rows: list[list[str]] = []
    with path.open("r", errors="ignore", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            if row:
                rows.append(row)
    if not rows:
        return {"path": str(path), "kind": "text", "error": "empty"}

    first = [cell.strip() for cell in rows[0]]
    header_like = any(any(ch.isalpha() for ch in cell) for cell in first)
    data_rows = rows[1:] if header_like else rows
    col_count = max(len(row) for row in rows)
    lower_header = " ".join(first).lower()

    numeric_counts = [0] * col_count
    for row in data_rows:
        for idx, cell in enumerate(row[:col_count]):
            value = cell.strip().replace(",", ".")
            try:
                float(value)
                numeric_counts[idx] += 1
            except ValueError:
                pass

    def has_hint(hints: tuple[str, ...]) -> bool:
        return any(hint in lower_header for hint in hints)

    likely_voltage = has_hint(VOLTAGE_HINTS)
    likely_current = has_hint(CURRENT_HINTS)
    likely_time = has_hint(TIME_HINTS)
    enough_numeric_cols = sum(count > max(3, len(data_rows) // 2) for count in numeric_counts) >= 2

    return {
        "path": str(path),
        "kind": "text",
        "suffix": path.suffix.lower(),
        "delimiter": "\\t" if delimiter == "\t" else delimiter,
        "sample_rows": len(rows),
        "columns": col_count,
        "header_like": header_like,
        "header": first[:12],
        "numeric_counts": numeric_counts[:12],
        "likely_voltage_header": likely_voltage,
        "likely_current_header": likely_current,
        "likely_time_header": likely_time,
        "enough_numeric_cols_for_vi": enough_numeric_cols,
        "candidate_ordered_vi": bool(enough_numeric_cols and (likely_voltage or likely_current or col_count in {2, 3, 4})),
    }


def inspect_excel(path: Path) -> dict[str, Any]:
    try:
        import pandas as pd
    except Exception as exc:
        return {"path": str(path), "kind": "excel", "error": f"pandas unavailable: {exc}"}
    try:
        xl = pd.ExcelFile(path)
        sheets = []
        for sheet in xl.sheet_names[:20]:
            frame = xl.parse(sheet, nrows=20)
            cols = [str(col) for col in frame.columns]
            lower = " ".join(cols).lower()
            sheets.append(
                {
                    "sheet": sheet,
                    "columns": cols[:12],
                    "n_columns": int(frame.shape[1]),
                    "likely_voltage_header": any(hint in lower for hint in VOLTAGE_HINTS),
                    "likely_current_header": any(hint in lower for hint in CURRENT_HINTS),
                    "likely_time_header": any(hint in lower for hint in TIME_HINTS),
                }
            )
        return {"path": str(path), "kind": "excel", "sheets": sheets}
    except Exception as exc:
        return {"path": str(path), "kind": "excel", "error": str(exc)}


def inspect_root(root: Path) -> dict[str, Any]:
    extracted = extract_archives(root)
    files = [path for path in root.rglob("*") if path.is_file()]
    table_files = [path for path in files if path.suffix.lower() in TABLE_SUFFIXES]
    archive_files = [path for path in files if path.suffix.lower() in ARCHIVE_SUFFIXES]

    inspections: list[dict[str, Any]] = []
    for path in table_files[:200]:
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            try:
                inspections.append(inspect_text_table(path))
            except Exception as exc:
                inspections.append({"path": str(path), "kind": "text", "error": str(exc)})
        elif suffix in {".xlsx", ".xls"}:
            inspections.append(inspect_excel(path))

    candidates = [
        row
        for row in inspections
        if row.get("candidate_ordered_vi")
        or (row.get("kind") == "excel" and any(sheet.get("likely_voltage_header") or sheet.get("likely_current_header") for sheet in row.get("sheets", [])))
    ]
    return {
        "root": str(root),
        "archives": [str(path) for path in archive_files],
        "extracted_dirs": [str(path) for path in extracted],
        "file_count": len(files),
        "table_file_count": len(table_files),
        "inspected_table_count": len(inspections),
        "candidate_ordered_vi_count": len(candidates),
        "candidate_ordered_vi_files": [row["path"] for row in candidates[:50]],
        "inspections": inspections,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# External Memristor Dataset Schema Inspection",
        "",
        f"Root: `{report['root']}`",
        "",
        "| Check | Value |",
        "|---|---:|",
        f"| files | {report['file_count']} |",
        f"| archives | {len(report['archives'])} |",
        f"| table files | {report['table_file_count']} |",
        f"| inspected tables | {report['inspected_table_count']} |",
        f"| candidate ordered V/I files | {report['candidate_ordered_vi_count']} |",
        "",
        "## Candidate Files",
        "",
    ]
    if report["candidate_ordered_vi_files"]:
        for candidate in report["candidate_ordered_vi_files"]:
            lines.append(f"- `{candidate}`")
    else:
        lines.append("No likely ordered V/I table files found by this heuristic.")
    lines.extend(["", "## Inspected Tables", ""])
    for row in report["inspections"][:80]:
        lines.append(f"### `{row['path']}`")
        if "error" in row:
            lines.append(f"- error: `{row['error']}`")
        elif row["kind"] == "text":
            lines.append(f"- columns: `{row['columns']}`")
            lines.append(f"- header: `{row.get('header', [])}`")
            lines.append(f"- candidate ordered V/I: `{row.get('candidate_ordered_vi')}`")
        elif row["kind"] == "excel":
            for sheet in row.get("sheets", [])[:8]:
                lines.append(f"- sheet `{sheet['sheet']}` columns: `{sheet['columns']}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_root(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.out, report)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "json": str(json_path), "candidates": report["candidate_ordered_vi_count"]}, indent=2))


if __name__ == "__main__":
    main()
