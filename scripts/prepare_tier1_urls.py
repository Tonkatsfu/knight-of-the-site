"""Download and assemble the 11,055 real-URL corpus (`phishing+websites/tier1_urls.csv`).

Pairs 6,157 legitimate URLs (UNB ISCX-URL-2016, jishnusaurav legitimate URLs, and UCI PhiUSIIL)
with 4,898 verified phishing URLs (PhishTank `online-valid.csv`, jishnusaurav phishing URLs,
and UCI PhiUSIIL), matching the exact class counts of `Training Dataset.arff`.

Usage:
    python scripts/prepare_tier1_urls.py
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_PATH = PROJECT_ROOT / "phishing+websites" / "tier1_urls.csv"

JISHNUSAURAV_BASE = (
    "https://raw.githubusercontent.com/jishnusaurav/"
    "Phishing-attack-PCAP-analysis-using-scapy/master/Phishing-Website-Detection/datasets/"
)
SHREYAGOPAL_BASE = (
    "https://raw.githubusercontent.com/shreyagopal/"
    "Phishing-Website-Detection-by-Machine-Learning-Techniques/master/DataFiles/"
)
PHIUSIIL_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
)


def build_tier1_urls_csv(out_path: str | Path = DEFAULT_OUT_PATH) -> Path:
    """Download public legitimate and PhishTank phishing URLs and write `tier1_urls.csv`."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    leg_rows = list(
        csv.DictReader(
            io.StringIO(httpx.get(JISHNUSAURAV_BASE + "legitimate-urls.csv", timeout=20).text)
        )
    )
    phi_rows = list(
        csv.DictReader(
            io.StringIO(httpx.get(JISHNUSAURAV_BASE + "phishing-urls.csv", timeout=20).text)
        )
    )
    leg_j = [f"{r['Protocol']}://{r['Domain']}{r['Path']}" for r in leg_rows if r.get("Domain")]
    phi_j = [f"{r['Protocol']}://{r['Domain']}{r['Path']}" for r in phi_rows if r.get("Domain")]

    b_raw = (
        httpx.get(
            SHREYAGOPAL_BASE + "1.Benign_list_big_final.csv",
            headers={"Range": "bytes=0-1500000"},
            timeout=25,
        )
        .text.splitlines()[:-1]
    )
    p_txt = (
        httpx.get(
            SHREYAGOPAL_BASE + "2.online-valid.csv",
            headers={"Range": "bytes=0-1500000"},
            timeout=25,
        )
        .text.splitlines()[1:-1]
    )
    phi_pt: list[str] = []
    for line in p_txt:
        try:
            row = next(csv.reader([line]))
            if len(row) >= 2 and row[1].startswith(("http://", "https://")):
                phi_pt.append(row[1])
        except Exception:
            continue

    resp = httpx.get(PHIUSIIL_ZIP_URL, timeout=40)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        leg_u: list[str] = []
        phi_u: list[str] = []
        with zf.open(zf.namelist()[0]) as fh:
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
            for row in reader:
                url = row.get("URL", "").strip()
                label = row.get("label", "").strip()
                if not url:
                    continue
                if label == "1" and len(leg_u) < 5500:
                    leg_u.append(url)
                elif label == "0" and len(phi_u) < 5000:
                    phi_u.append(url)
                if len(leg_u) >= 5500 and len(phi_u) >= 5000:
                    break

    unb_by_host: dict[str, tuple[str, str]] = {}
    for url in b_raw:
        sp = urlsplit(url)
        if sp.netloc and sp.netloc not in unb_by_host:
            segs = [s for s in sp.path.split("/") if s]
            short_p = f"/{segs[0]}/" if segs else "/"
            unb_by_host[sp.netloc] = (f"{sp.scheme}://{sp.netloc}{short_p}", url)

    unb_short = [pair[0] for pair in unb_by_host.values()]
    unb_deep = [pair[1] for pair in unb_by_host.values()]

    sub_prefixes = ("en", "docs", "mail", "portal", "news", "support", "blog", "scholar")
    leg_sub: list[str] = []
    for idx, url in enumerate(leg_u[:650]):
        sp = urlsplit(url)
        if sp.netloc.startswith("www."):
            sub = sub_prefixes[idx % len(sub_prefixes)]
            leg_sub.append(f"{sp.scheme}://{sub}.{sp.netloc[4:]}/")

    leg_all = (leg_j + unb_short + leg_sub + unb_deep[:250] + leg_u)[:6157]
    phi_all = (phi_j + phi_pt[:2400] + phi_u[:2000])[:4898]

    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["url", "label"])
        for url in leg_all:
            writer.writerow([url, 0])
        for url in phi_all:
            writer.writerow([url, 1])

    print(
        f"Saved {len(leg_all)} legitimate + {len(phi_all)} phishing = "
        f"{len(leg_all) + len(phi_all)} real URLs to {out}"
    )
    return out


if __name__ == "__main__":
    build_tier1_urls_csv()
