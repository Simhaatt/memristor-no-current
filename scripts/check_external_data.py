"""Check external dataset landing pages without downloading restricted data."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen


DATASETS = {
    "sdc_github": "https://github.com/sdpenguin/memristor-state-characterisation",
    "diffusive_memristor_lboro": "https://repository.lboro.ac.uk/articles/dataset/Current-voltage_measurements_on_diffusive_memristors/30197974",
    "zenodo_qualitative_fallback": "https://zenodo.org/records/18619371",
}


def check_url(url: str) -> dict[str, object]:
    request = Request(url, method="GET", headers={"User-Agent": "memristorkan-access-check/0.1"})
    try:
        with urlopen(request, timeout=20) as response:
            return {"ok": True, "status": response.status, "final_url": response.geturl()}
    except URLError as exc:
        return {"ok": False, "error": str(exc)}


def main() -> None:
    print(json.dumps({name: check_url(url) for name, url in DATASETS.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
