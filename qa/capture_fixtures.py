"""
capture_fixtures.py -- rekam respons LIVE tiap sumber menjadi fixture untuk test offline.

Jalankan saat scraper sedang BERFUNGSI BAIK (cek dulu dengan health_check):
    python -m qa.capture_fixtures                    # semua sumber
    python -m qa.capture_fixtures --sumber Detik     # satu sumber (mis. setelah memperbaiki parser)

Hasil: tests/fixtures/<sumber>/tape.json (respons mentah) + manifest.json (ekspektasi).
Test `tests/test_replay_fixtures.py` lalu memutar ulang respons ini TANPA internet
dan memastikan parser masih menghasilkan jumlah artikel & isi yang sama.
Kalau situs berubah dan Anda memperbaiki parser, rekam ulang fixture sumber itu.
"""
import argparse
import os
from datetime import datetime

from qa.common import konten_valid, rentang_tanggal, tulis_json
from qa.engine_loader import load_engine, panggil_scraper, slug
from qa.http_probe import HttpProbe

FOLDER_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures")


def rekam_sumber(engine, nama, tema, hari, halaman, n_konten, folder):
    tgl_mulai, tgl_akhir = rentang_tanggal(hari)
    tape = {}
    with HttpProbe("record", tape=tape):
        artikel = panggil_scraper(engine, nama, tema, halaman, tgl_mulai, tgl_akhir) or []
        konten = {}
        for a in [a for a in artikel if not a.get("konten")][:n_konten]:
            konten[a["link"]] = konten_valid(engine.scrape_isi_artikel(a["link"]), engine.KONTEN_INVALID)

    manifest = {
        "sumber": nama, "tema": tema, "halaman": halaman,
        "tanggal_mulai": str(tgl_mulai), "tanggal_akhir": str(tgl_akhir),
        "direkam_pada": datetime.now().isoformat(timespec="seconds"),
        "jumlah_artikel": len(artikel),
        "jumlah_tanggal_terbaca": sum(1 for a in artikel if a.get("tanggal")),
        "konten_ok": konten,
    }
    d = os.path.join(folder, slug(nama))
    tulis_json(os.path.join(d, "tape.json"), tape)
    tulis_json(os.path.join(d, "manifest.json"), manifest)
    return manifest, len(tape)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", nargs="*")
    ap.add_argument("--tema", default="indonesia")
    ap.add_argument("--hari", type=int, default=2)
    ap.add_argument("--halaman", type=int, default=1)
    ap.add_argument("--konten", type=int, default=2, help="Jumlah halaman artikel yang ikut direkam")
    ap.add_argument("--folder", default=FOLDER_DEFAULT)
    args = ap.parse_args(argv)

    engine = load_engine()
    for nama in args.sumber or list(engine.SCRAPER_PER_SUMBER.keys()):
        try:
            m, n = rekam_sumber(engine, nama, args.tema, args.hari, args.halaman, args.konten, args.folder)
            peringatan = "  ⚠️ 0 artikel -- jangan dipakai sebagai fixture, cek sumbernya dulu" if not m["jumlah_artikel"] else ""
            print(f"✓ {nama}: {m['jumlah_artikel']} artikel, {n} respons direkam{peringatan}")
        except Exception as e:
            print(f"✗ {nama}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
