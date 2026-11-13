"""Package results and trained model checkpoints into a downloadable zip."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def add_tree(archive: zipfile.ZipFile, root: Path, arc_prefix: str) -> list[str]:
    added = []
    if not root.exists():
        return added
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_file():
            arcname = (Path(arc_prefix) / path.relative_to(root)).as_posix()
            archive.write(path, arcname)
            added.append(arcname)
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dist/memristorkan_results_models.zip")
    parser.add_argument("--include-code", action="store_true")
    parser.add_argument("--include-processed-metadata", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "note": (
            "Contains available results and model checkpoints. Older runs before "
            "--save-models was added have results only, not trained weights."
        ),
        "files": [],
    }
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        files = []
        files.extend(add_tree(archive, Path("results"), "results"))
        files.extend(add_tree(archive, Path("checkpoints"), "checkpoints"))
        if args.include_code:
            files.extend(add_tree(archive, Path("src"), "src"))
            files.extend(add_tree(archive, Path("scripts"), "scripts"))
            files.extend(add_tree(archive, Path("configs"), "configs"))
            for filename in ["README.md", "KAGGLE_README.md", "requirements_kaggle.txt", "pyproject.toml"]:
                path = Path(filename)
                if path.exists():
                    archive.write(path, filename)
                    files.append(filename)
        if args.include_processed_metadata:
            processed_paths = [
                Path("data/processed/sdc_state_metadata.json"),
                Path("data/processed/sdc_state_splits.npz"),
            ]
            for pattern in [
                "sdc4dopant_holdout_*.npz",
                "sdc4dopant_holdout_*_metadata.json",
                "sdc4dopant_pooled.npz",
                "sdc4dopant_pooled_metadata.json",
            ]:
                processed_paths.extend(sorted(Path("data/processed").glob(pattern)))
            seen_processed = set()
            for path in processed_paths:
                if path in seen_processed:
                    continue
                seen_processed.add(path)
                if path.exists():
                    arcname = str(Path("data/processed") / path.name)
                    arcname = Path("data/processed", path.name).as_posix()
                    archive.write(path, arcname)
                    files.append(arcname)
        manifest["files"] = files
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))

    print(json.dumps({"out": str(out), "bytes": out.stat().st_size, "files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
