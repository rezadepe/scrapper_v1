"""Helper bersama untuk health check & benchmark."""
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

WIB = ZoneInfo("Asia/Jakarta")
AWALAN_KONTEN_GAGAL = ("Gagal mengambil halaman", "Konten tidak ditemukan", "Konten kosong",
                       "Selector tidak ditemukan")
PANJANG_KONTEN_MINIMAL = 200


def sekarang_wib():
    return datetime.now(WIB)


def rentang_tanggal(hari):
    """Rentang tanggal versi WIB (bukan jam server). hari=3 -> hari ini & 2 hari sebelumnya."""
    akhir = sekarang_wib().date()
    return akhir - timedelta(days=max(hari, 1) - 1), akhir


def konten_valid(teks, konten_invalid=()):
    if not teks or not isinstance(teks, str):
        return False
    if teks in konten_invalid or teks.startswith(AWALAN_KONTEN_GAGAL):
        return False
    return len(teks) >= PANJANG_KONTEN_MINIMAL


def pct(bagian, total):
    return round(100.0 * bagian / total, 1) if total else None


def info_lingkungan(label):
    commit = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        pass
    return {
        "label": label,
        "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tz_server": datetime.now().astimezone().tzname(),
        "commit_app": commit,
    }


def tulis_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def tambah_csv(path, baris, kolom):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    baru = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kolom, extrasaction="ignore")
        if baru:
            w.writeheader()
        for b in baris:
            w.writerow(b)
