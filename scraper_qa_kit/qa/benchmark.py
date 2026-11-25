"""
benchmark.py -- ukur performa END-TO-END pencarian, persis alur app (cari_berita,
paralel per sumber + ambil isi + ringkasan + sentimen), tanpa cache Streamlit.

Contoh:
    python -m qa.benchmark                         # skenario "cepat"
    python -m qa.benchmark --skenario default_app  # setara setting default aplikasi
    python -m qa.benchmark --skenario cepat default_app seminggu --label laptop

Yang diukur per skenario:
    - total durasi, fase pencarian (paralel per sumber), fase ambil isi artikel
    - waktu selesai per sumber (siapa yang paling lambat)
    - jumlah artikel, artikel/menit, jumlah request & request gagal
"""
import argparse
import os
import re
import time

from qa.common import info_lingkungan, rentang_tanggal, sekarang_wib, tambah_csv, tulis_json
from qa.engine_loader import load_engine, slug
from qa.http_probe import HttpProbe

SKENARIO = {
    # nama: (hari, halaman, ambil_konten)
    "cepat": (1, 1, False),
    "default_app": (2, 3, True),   # app default: kemarin s/d hari ini, 3 halaman, isi lengkap ON
    "seminggu": (7, 3, True),
}
KOLOM_HISTORY = ["waktu_wib", "label", "skenario", "tema", "total_s", "fase_pencarian_s", "fase_konten_s",
                 "jumlah_artikel", "artikel_per_menit", "request_total", "request_gagal", "sumber_terlambat"]
_RE_SELESAI = re.compile(r"^(?:✅ (?P<ok>.+?): \d+ artikel ditemukan|\[ERROR\] (?P<err>.+?):)")


def jalankan(engine, nama, tema, label):
    hari, halaman, ambil_konten = SKENARIO[nama]
    tgl_mulai, tgl_akhir = rentang_tanggal(hari)
    sumber = list(engine.SCRAPER_PER_SUMBER.keys())
    pesan = []
    t0 = time.perf_counter()

    def cb(msg):
        pesan.append((time.perf_counter() - t0, msg))

    error = None
    hasil = []
    with HttpProbe("monitor") as probe:
        try:
            hasil = engine.cari_berita(tema, sumber, tgl_mulai, tgl_akhir, halaman, ambil_konten,
                                       match_persis=False, match_semua_kata=False, progress_callback=cb)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        total = time.perf_counter() - t0
        http = probe.ringkas()

    selesai_per_sumber = {}
    for t, m in pesan:
        g = _RE_SELESAI.match(m)
        if g:
            selesai_per_sumber[(g.group("ok") or g.group("err")).strip()] = round(t, 1)
    fase_cari = max(selesai_per_sumber.values()) if selesai_per_sumber else total
    awal_konten = next((t for t, m in pesan if m.startswith("Mengambil isi")), None)
    fase_konten = (total - awal_konten) if awal_konten is not None else 0.0
    terlambat = max(selesai_per_sumber, key=selesai_per_sumber.get) if selesai_per_sumber else None

    return {
        "skenario": nama, "hari": hari, "halaman": halaman, "ambil_konten": ambil_konten,
        "tanggal_mulai": str(tgl_mulai), "tanggal_akhir": str(tgl_akhir), "tema": tema,
        "error": error,
        "total_s": round(total, 1),
        "fase_pencarian_s": round(fase_cari, 1),
        "fase_konten_s": round(fase_konten, 1),
        "jumlah_artikel": len(hasil),
        "artikel_per_menit": round(len(hasil) / (total / 60), 1) if total > 0 else None,
        "selesai_per_sumber_s": dict(sorted(selesai_per_sumber.items(), key=lambda kv: kv[1])),
        "sumber_terlambat": terlambat,
        "artikel_per_sumber": _hitung_per_sumber(hasil, engine),
        "http": http,
        "pesan_error": [m for _, m in pesan if "ERROR" in m or "⚠️" in m][:15],
    }


def _hitung_per_sumber(hasil, engine):
    hitung = {}
    for a in hasil:
        s = engine._sumber_grup(a)
        hitung[s] = hitung.get(s, 0) + 1
    return hitung


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark performa pencarian end-to-end")
    ap.add_argument("--skenario", nargs="*", default=["cepat"], choices=list(SKENARIO))
    ap.add_argument("--tema", default="indonesia")
    ap.add_argument("--label", default=os.environ.get("QA_LABEL", "lokal"))
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)

    engine = load_engine()
    mulai = sekarang_wib()
    hasil = []
    for nama in args.skenario:
        print(f"→ Skenario {nama} {SKENARIO[nama]} ...", flush=True)
        r = jalankan(engine, nama, args.tema, args.label)
        print(f"  total {r['total_s']}s (cari {r['fase_pencarian_s']}s, isi {r['fase_konten_s']}s), "
              f"{r['jumlah_artikel']} artikel, terlambat: {r['sumber_terlambat']}", flush=True)
        hasil.append(r)

    laporan = {"jenis": "benchmark",
               "meta": {**info_lingkungan(args.label), "waktu_mulai_wib": mulai.strftime("%Y-%m-%d %H:%M")},
               "skenario": hasil}
    cap = mulai.strftime("%Y%m%d_%H%M")
    tulis_json(os.path.join(args.out, f"benchmark_{slug(args.label)}_{cap}.json"), laporan)
    tulis_json(os.path.join(args.out, f"latest_benchmark_{slug(args.label)}.json"), laporan)
    tambah_csv(os.path.join(args.out, "benchmark_history.csv"),
               [{**r, "waktu_wib": laporan["meta"]["waktu_mulai_wib"], "label": args.label,
                 "request_total": r["http"]["request_total"], "request_gagal": r["http"]["request_gagal"]}
                for r in hasil], KOLOM_HISTORY)
    print(f"\nLaporan tersimpan di {args.out}/latest_benchmark_{slug(args.label)}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
