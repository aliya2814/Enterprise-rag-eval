"""Download real SEC 10-K filings from the public EDGAR system (free, no API key).

SEC requires a descriptive User-Agent. Set EDGAR_UA env var to your email, e.g.
    export EDGAR_UA="Aliya Tarrannum aliya@example.com"

Usage:
    python src/download_filings.py --tickers AAPL MSFT --out data/raw
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import requests

UA = os.getenv("EDGAR_UA", "portfolio-project contact@example.com")
HEADERS = {"User-Agent": UA}


def _cik_for_ticker(ticker: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS, timeout=30).json()
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"Ticker not found on EDGAR: {ticker}")


def _latest_10k(cik: str) -> tuple[str, str]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(url, headers=HEADERS, timeout=30).json()
    recent = data["filings"]["recent"]
    for form, acc, doc in zip(recent["form"], recent["accessionNumber"],
                              recent["primaryDocument"]):
        if form == "10-K":
            acc_nodash = acc.replace("-", "")
            return (f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{acc_nodash}/{doc}"), acc
    raise ValueError(f"No 10-K found for CIK {cik}")


def download(tickers: list[str], out: str) -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    for t in tickers:
        try:
            cik = _cik_for_ticker(t)
            url, acc = _latest_10k(cik)
            html = requests.get(url, headers=HEADERS, timeout=60).text
            dest = Path(out) / f"{t}_{acc}.html"
            dest.write_text(html, encoding="utf-8")
            print(f"[ok] {t}: saved {dest.name} ({len(html)//1000} KB)")
            time.sleep(0.5)  # be polite to EDGAR
        except Exception as e:
            print(f"[fail] {t}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT"])
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    download(args.tickers, args.out)
