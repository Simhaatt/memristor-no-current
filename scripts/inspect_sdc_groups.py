"""Inspect SDC raw state-data grouping and dopant labels.

This is the Step 0 decision check. It reports whether the currently available
SDC source appears to contain explicit dopant-family labels such as W/Sn/Cr/C.
The local public SDC clone usually contains voltage/day/device-size groupings
(`data_04`, `data_10`, `150m150m`, etc.), not dopant names.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DOPANT_PATTERNS = {
    "W": re.compile(r"(^|[^a-z0-9])(w|tungsten)([^a-z0-9]|$)", re.IGNORECASE),
    "Sn": re.compile(r"(^|[^a-z0-9])(sn|tin)([^a-z0-9]|$)", re.IGNORECASE),
    "Cr": re.compile(r"(^|[^a-z0-9])(cr|chromium)([^a-z0-9]|$)", re.IGNORECASE),
    "C": re.compile(r"(^|[^a-z0-9])(c|carbon)([^a-z0-9]|$)", re.IGNORECASE),
}


def csv_files(root: Path) -> list[Path]:
    state_root = root / "data" / "state_data"
    if not state_root.exists():
        state_root = root
    return sorted(path for path in state_root.rglob("*.csv") if path.is_file())


def path_parts_after_state(path: Path) -> list[str]:
    parts = list(path.parts)
    if "state_data" in parts:
        idx = parts.index("state_data")
        return parts[idx + 1 :]
    return parts


def inspect(root: Path) -> dict[str, object]:
    files = csv_files(root)
    token_counts: Counter[str] = Counter()
    level_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dopant_hits: dict[str, list[str]] = {name: [] for name in DOPANT_PATTERNS}
    for path in files:
        rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
        lowered = rel.lower()
        for dopant, pattern in DOPANT_PATTERNS.items():
            if pattern.search(lowered):
                dopant_hits[dopant].append(rel)
        parts = path_parts_after_state(path)
        for i, part in enumerate(parts[:-1]):
            level_counts[f"level_{i}"][part] += 1
            token_counts[part] += 1
        token_counts[path.stem] += 1

    explicit_dopants = {name: hits for name, hits in dopant_hits.items() if hits}
    likely_grouping = "unknown"
    if explicit_dopants:
        likely_grouping = "explicit_dopant_labels_detected"
    elif any("data_04" in counter or "data_10" in counter for counter in level_counts.values()):
        likely_grouping = "voltage_or_day_groups_only_no_explicit_dopants"

    return {
        "root": str(root),
        "state_csv_files": len(files),
        "likely_grouping": likely_grouping,
        "explicit_dopant_hit_counts": {name: len(hits) for name, hits in dopant_hits.items()},
        "explicit_dopant_examples": {name: hits[:10] for name, hits in explicit_dopants.items()},
        "directory_level_counts": {
            level: dict(counter.most_common(30)) for level, counter in sorted(level_counts.items())
        },
        "top_path_tokens": dict(token_counts.most_common(50)),
        "decision": (
            "Use current SDC source for dopant split."
            if explicit_dopants
            else "Current SDC source does not expose W/Sn/Cr/C dopant labels; use a four-dopant release for Step 5."
        ),
    }


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# SDC Group/Dopant Inspection",
        "",
        f"Root: `{payload['root']}`",
        "",
        f"State CSV files: {payload['state_csv_files']}",
        "",
        f"Likely grouping: `{payload['likely_grouping']}`",
        "",
        f"Decision: {payload['decision']}",
        "",
        "## Explicit Dopant Hit Counts",
        "",
        "| Dopant | Hits |",
        "|---|---:|",
    ]
    for name, count in payload["explicit_dopant_hit_counts"].items():  # type: ignore[index,union-attr]
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## Directory Level Counts", ""])
    for level, counts in payload["directory_level_counts"].items():  # type: ignore[index,union-attr]
        lines.append(f"### {level}")
        lines.append("")
        lines.append("| Token | Files |")
        lines.append("|---|---:|")
        for token, count in counts.items():
            lines.append(f"| `{token}` | {count} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/sdc/memristor-state-characterisation"))
    parser.add_argument("--out", type=Path, default=Path("results/sdc_group_dopant_inspection.json"))
    args = parser.parse_args()
    payload = inspect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.out.with_suffix(".md"), payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
