"""
Putar ulang respons yang direkam (python -m qa.capture_fixtures) TANPA internet.
Gagal di sini = parser berubah perilakunya terhadap HTML/JSON yang sama (regresi kode).
Kalau health check gagal tapi test ini lulus = situsnya yang berubah/memblokir, bukan kode Anda.
"""
import json
import os
from datetime import date

import pytest

from qa.engine_loader import panggil_scraper
from qa.http_probe import HttpProbe

FOLDER = os.path.join(os.path.dirname(__file__), "fixtures")
SUMBER = sorted(d for d in os.listdir(FOLDER) if os.path.exists(os.path.join(FOLDER, d, "manifest.json"))) \
    if os.path.isdir(FOLDER) else []


def _muat(nama):
    with open(os.path.join(FOLDER, nama, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(FOLDER, nama, "tape.json"), encoding="utf-8") as f:
        tape = json.load(f)
    return manifest, tape


@pytest.mark.skipif(not SUMBER, reason="Belum ada fixture. Jalankan: python -m qa.capture_fixtures")
@pytest.mark.parametrize("folder", SUMBER)
def test_parser_stabil(engine, folder):
    m, tape = _muat(folder)
    with HttpProbe("replay", tape=tape) as probe:
        artikel = panggil_scraper(engine, m["sumber"], m["tema"], m["halaman"],
                                  date.fromisoformat(m["tanggal_mulai"]), date.fromisoformat(m["tanggal_akhir"]))
        konten = {link: engine.scrape_isi_artikel(link) for link in m["konten_ok"]}

    assert not probe.miss, f"Request tidak ada di fixture (URL berubah?): {probe.miss[:2]}"
    assert len(artikel) == m["jumlah_artikel"]
    assert sum(1 for a in artikel if a.get("tanggal")) == m["jumlah_tanggal_terbaca"]
    assert all(a.get("judul") and str(a.get("link", "")).startswith("http") for a in artikel)
    from qa.common import konten_valid
    for link, ok in m["konten_ok"].items():
        assert konten_valid(konten[link], engine.KONTEN_INVALID) == ok, f"Isi artikel berubah: {link}"
