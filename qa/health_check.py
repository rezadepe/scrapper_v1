"""
health_check.py -- cek kesehatan LIVE setiap scraper.

Contoh:
    python -m qa.health_check                       # semua sumber, keyword "indonesia", 3 hari
    python -m qa.health_check --label laptop        # beri label lingkungan
    python -m qa.health_check --sumber Detik Kompas --halaman 2 --sampel-konten 5

Setiap sumber dijalankan BERGANTIAN (bukan paralel) supaya durasi & request
bisa diatribusikan ke sumbernya masing-masing.

Output (folder reports/):
    health_<label>_<YYYYmmdd_HHMM>.json   detail lengkap
    latest_<label>.json                    salinan run terakhir
    history.csv                            1 baris per sumber per run (untuk tren)
    SUMMARY.md                             tabel ringkas yang mudah dibaca
"""
import argparse
import os
import time
import traceback
from datetime import datetime, timedelta

from qa.common import (WIB, info_lingkungan, konten_valid, pct, rentang_tanggal,
                       sekarang_wib, tambah_csv, tulis_json)
from qa.engine_loader import load_engine, panggil_scraper, slug
from qa.http_probe import HttpProbe

# Ambang klasifikasi -- silakan disesuaikan.
AMBANG = {
    "pct_tanggal_terbaca_min": 90,
    "pct_dalam_rentang_min": 90,
    "pct_konten_berhasil_min": 70,
}
KOLOM_HISTORY = ["waktu_wib", "label", "sumber", "status", "jumlah_artikel", "pct_tanggal_terbaca",
                 "pct_dalam_rentang", "pct_tanggal_masa_depan", "pct_konten_berhasil", "durasi_s",
                 "request_total", "request_gagal", "terdeteksi_blokir", "alasan"]


def evaluasi_tanggal(artikel, tgl_mulai, tgl_akhir):
    batas_bawah = datetime.combine(tgl_mulai, datetime.min.time())
    batas_atas = datetime.combine(tgl_akhir, datetime.max.time())
    batas_masa_depan = sekarang_wib().replace(tzinfo=None) + timedelta(hours=1)

    terbaca = [a["tanggal"] for a in artikel if isinstance(a.get("tanggal"), datetime)]
    naive = [t.replace(tzinfo=None) for t in terbaca]
    dalam = [t for t in naive if batas_bawah <= t <= batas_atas]
    masa_depan = [t for t in naive if t > batas_masa_depan]
    return {
        "pct_tanggal_terbaca": pct(len(terbaca), len(artikel)),
        "pct_dalam_rentang": pct(len(dalam), len(naive)),
        "pct_tanggal_masa_depan": pct(len(masa_depan), len(naive)),
        "tanggal_min": min(naive).isoformat() if naive else None,
        "tanggal_maks": max(naive).isoformat() if naive else None,
        # Contoh mentah vs hasil parse -> untuk cek manual zona waktu / format
        "contoh_tanggal": [
            {"tanggal_teks": a.get("tanggal_teks"),
             "tanggal_parse": a["tanggal"].isoformat() if isinstance(a.get("tanggal"), datetime) else None}
            for a in artikel[:3]
        ],
    }


def uji_konten(engine, artikel, n_sampel):
    hasil = {"diuji": 0, "berhasil": 0, "sumber_konten": {"dari_search": 0, "fetch_halaman": 0},
             "panjang_rata": None, "durasi_rata_s": None, "contoh_gagal": []}
    panjang, durasi = [], []
    for a in artikel[:n_sampel]:
        hasil["diuji"] += 1
        teks = a.get("konten")
        if teks:
            hasil["sumber_konten"]["dari_search"] += 1
        else:
            hasil["sumber_konten"]["fetch_halaman"] += 1
            t0 = time.perf_counter()
            try:
                teks = engine.scrape_isi_artikel(a["link"])
            except Exception as e:
                teks = f"EXCEPTION {type(e).__name__}: {e}"
            durasi.append(time.perf_counter() - t0)
        if konten_valid(teks, engine.KONTEN_INVALID):
            hasil["berhasil"] += 1
            panjang.append(len(teks))
        else:
            hasil["contoh_gagal"].append({"link": a["link"], "hasil": (teks or "")[:120]})
    hasil["pct_berhasil"] = pct(hasil["berhasil"], hasil["diuji"])
    hasil["panjang_rata"] = round(sum(panjang) / len(panjang)) if panjang else None
    hasil["durasi_rata_s"] = round(sum(durasi) / len(durasi), 2) if durasi else None
    return hasil


def klasifikasi(r):
    """Tentukan status sumber + alasannya."""
    alasan = []
    http = r["http"]
    if r.get("error"):
        return "error", [f"Exception: {r['error']}"]
    if r["jumlah_artikel"] == 0:
        if http["request_total"] == 0:
            return "error", ["Tidak ada request yang terkirim"]
        if http["terdeteksi_blokir"]:
            return "diblokir", [f"{http['terdeteksi_blokir']} respons terlihat diblokir (403/429/503/Cloudflare)"]
        if http["request_gagal"] == http["request_total"]:
            return "http_error", [f"Semua request gagal: {http['status_http']}"]
        return "kosong_mencurigakan", ["Request berhasil tapi 0 artikel -- kemungkinan selector/format respons berubah"]

    if http["terdeteksi_blokir"]:
        alasan.append(f"{http['terdeteksi_blokir']} request terlihat diblokir")
    t = r["tanggal"]
    if t["pct_tanggal_terbaca"] is not None and t["pct_tanggal_terbaca"] < AMBANG["pct_tanggal_terbaca_min"]:
        alasan.append(f"Tanggal terbaca hanya {t['pct_tanggal_terbaca']}%")
    if t["pct_dalam_rentang"] is not None and t["pct_dalam_rentang"] < AMBANG["pct_dalam_rentang_min"]:
        alasan.append(f"Hanya {t['pct_dalam_rentang']}% tanggal dalam rentang")
    if t["pct_tanggal_masa_depan"]:
        alasan.append(f"{t['pct_tanggal_masa_depan']}% tanggal di masa depan (salah parse / zona waktu)")
    k = r["konten"]
    if k["diuji"] and (k["pct_berhasil"] or 0) < AMBANG["pct_konten_berhasil_min"]:
        alasan.append(f"Isi artikel berhasil diambil hanya {k['pct_berhasil']}%")
    return ("menurun" if alasan else "ok"), alasan


def cek_sumber(engine, probe, nama, tema, halaman, tgl_mulai, tgl_akhir, n_sampel):
    probe.set_label(nama)
    log = []
    hasil = {"sumber": nama, "error": None}
    t0 = time.perf_counter()
    artikel = []
    try:
        artikel = panggil_scraper(engine, nama, tema, halaman, tgl_mulai, tgl_akhir,
                                  progress_callback=log.append) or []
    except Exception as e:
        hasil["error"] = f"{type(e).__name__}: {e}"
        hasil["traceback"] = traceback.format_exc()[-1500:]
    hasil["durasi_scrape_s"] = round(time.perf_counter() - t0, 2)

    hasil["jumlah_artikel"] = len(artikel)
    hasil["link_unik"] = len({a.get("link") for a in artikel})
    hasil["pct_judul_link_lengkap"] = pct(
        sum(1 for a in artikel if a.get("judul") and str(a.get("link", "")).startswith("http")), len(artikel))
    hasil["pct_ada_ringkasan_redaksi"] = pct(sum(1 for a in artikel if a.get("ringkasan_redaksi")), len(artikel))
    hasil["tanggal"] = evaluasi_tanggal(artikel, tgl_mulai, tgl_akhir)

    t1 = time.perf_counter()
    hasil["konten"] = uji_konten(engine, artikel, n_sampel)
    hasil["durasi_konten_s"] = round(time.perf_counter() - t1, 2)
    hasil["durasi_s"] = round(hasil["durasi_scrape_s"] + hasil["durasi_konten_s"], 2)

    hasil["http"] = probe.ringkas(nama)
    hasil["log_scraper"] = log[-10:]
    hasil["status"], hasil["alasan"] = klasifikasi(hasil)
    return hasil


IKON = {"ok": "✅", "menurun": "🟡", "kosong_mencurigakan": "🟠", "diblokir": "⛔", "http_error": "🔴", "error": "💥"}


def tulis_summary(path, laporan):
    m = laporan["meta"]
    baris = [
        f"# Laporan Health Check Scraper",
        "",
        f"- Waktu: {m['waktu_mulai_wib']} WIB (durasi {m['durasi_total_s']} detik)",
        f"- Lingkungan: `{m['label']}` | GitHub Actions: {m['github_actions']} | commit app: `{m['commit_app']}`",
        f"- Parameter: tema `{m['tema']}`, {m['tanggal_mulai']} s/d {m['tanggal_akhir']}, "
        f"{m['halaman']} halaman, sampel isi {m['sampel_konten']} artikel/sumber",
        "",
        "| Sumber | Status | Artikel | Tgl terbaca | Dlm rentang | Isi OK | Req (gagal) | Durasi | Catatan |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in laporan["sumber"]:
        baris.append(
            f"| {r['sumber']} | {IKON.get(r['status'], '')} {r['status']} | {r['jumlah_artikel']} | "
            f"{r['tanggal']['pct_tanggal_terbaca'] if r['tanggal']['pct_tanggal_terbaca'] is not None else '-'}% | "
            f"{r['tanggal']['pct_dalam_rentang'] if r['tanggal']['pct_dalam_rentang'] is not None else '-'}% | "
            f"{r['konten']['pct_berhasil'] if r['konten']['pct_berhasil'] is not None else '-'}% | "
            f"{r['http']['request_total']} ({r['http']['request_gagal']}) | {r['durasi_s']}s | "
            f"{'; '.join(r['alasan'])} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(baris) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Health check live scraper berita")
    ap.add_argument("--tema", default="indonesia")
    ap.add_argument("--hari", type=int, default=3, help="Rentang hari ke belakang (WIB)")
    ap.add_argument("--halaman", type=int, default=1)
    ap.add_argument("--sumber", nargs="*", help="Subset sumber (default semua)")
    ap.add_argument("--sampel-konten", type=int, default=3, help="Artikel per sumber yang diuji isinya")
    ap.add_argument("--label", default=os.environ.get("QA_LABEL", "lokal"))
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)

    engine = load_engine()
    sumber = args.sumber or list(engine.SCRAPER_PER_SUMBER.keys())
    tgl_mulai, tgl_akhir = rentang_tanggal(args.hari)
    mulai = sekarang_wib()
    t0 = time.perf_counter()

    hasil_semua = []
    with HttpProbe("monitor") as probe:
        for nama in sumber:
            print(f"→ {nama} ...", flush=True)
            r = cek_sumber(engine, probe, nama, args.tema, args.halaman, tgl_mulai, tgl_akhir, args.sampel_konten)
            print(f"  {IKON.get(r['status'], '')} {r['status']}: {r['jumlah_artikel']} artikel, "
                  f"{r['durasi_s']}s {'; '.join(r['alasan'])}", flush=True)
            hasil_semua.append(r)

    laporan = {
        "jenis": "health_check",
        "meta": {**info_lingkungan(args.label),
                 "waktu_mulai_wib": mulai.strftime("%Y-%m-%d %H:%M"),
                 "durasi_total_s": round(time.perf_counter() - t0, 1),
                 "tema": args.tema, "tanggal_mulai": str(tgl_mulai), "tanggal_akhir": str(tgl_akhir),
                 "halaman": args.halaman, "sampel_konten": args.sampel_konten, "ambang": AMBANG},
        "sumber": hasil_semua,
    }
    cap = mulai.strftime("%Y%m%d_%H%M")
    tulis_json(os.path.join(args.out, f"health_{slug(args.label)}_{cap}.json"), laporan)
    tulis_json(os.path.join(args.out, f"latest_{slug(args.label)}.json"), laporan)
    tambah_csv(os.path.join(args.out, "history.csv"), [
        {"waktu_wib": laporan["meta"]["waktu_mulai_wib"], "label": args.label, "sumber": r["sumber"],
         "status": r["status"], "jumlah_artikel": r["jumlah_artikel"],
         "pct_tanggal_terbaca": r["tanggal"]["pct_tanggal_terbaca"],
         "pct_dalam_rentang": r["tanggal"]["pct_dalam_rentang"],
         "pct_tanggal_masa_depan": r["tanggal"]["pct_tanggal_masa_depan"],
         "pct_konten_berhasil": r["konten"]["pct_berhasil"], "durasi_s": r["durasi_s"],
         "request_total": r["http"]["request_total"], "request_gagal": r["http"]["request_gagal"],
         "terdeteksi_blokir": r["http"]["terdeteksi_blokir"], "alasan": "; ".join(r["alasan"])}
        for r in hasil_semua
    ], KOLOM_HISTORY)
    tulis_summary(os.path.join(args.out, "SUMMARY.md"), laporan)
    print(f"\nLaporan tersimpan di {args.out}/ (SUMMARY.md, latest_{slug(args.label)}.json, history.csv)")

    # exit code 0 selalu: health check adalah laporan, bukan gerbang deploy
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
