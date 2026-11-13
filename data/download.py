"""Download original datasets at the archived source revisions."""
import argparse
import hashlib
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    sources = json.loads((ROOT / "data/SOURCES.json").read_text())
    if args.list or not args.all:
        print(json.dumps(sources, indent=2))
        return
    for source in sources:
        target = ROOT / source["directory"]
        if "commit" in source:
            if target.exists():
                print(f"Already present, not modified: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", source["url"], str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", source["commit"]], check=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
            archive_path = target / "data_sets.zip"
            if not archive_path.exists():
                try:
                    urllib.request.urlretrieve(source["download"], archive_path)
                except Exception as error:
                    raise SystemExit(f"Download data_sets.zip manually from {source['url']} to {archive_path}: {error}")
            if hashlib.sha256(archive_path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("D0132 archive differs from the archived study version")
            with zipfile.ZipFile(archive_path) as archive:
                destination = (target / "data_sets_extracted").resolve()
                for name in archive.namelist():
                    if not (destination / name).resolve().is_relative_to(destination):
                        raise ValueError(f"unsafe archive entry: {name}")
                archive.extractall(destination)


if __name__ == "__main__":
    main()
