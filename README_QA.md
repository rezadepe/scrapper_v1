# Paket QA Scraper Berita

Paket ini menguji **reliability** dan **performa** scraper di `app.py` tanpa mengubah `app.py` sama sekali.
Engine dimuat langsung dari `app.py` (bagian di atas blok `# UI STREAMLIT`), jadi setiap perubahan
scraper otomatis ikut teruji.

## Isi

| File | Fungsi | Butuh internet? |
|---|---|---|
| `tests/` (pytest) | Unit test parser tanggal, zona waktu, encoding keyword, konten invalid, Kumparan, replay fixture | Tidak |
| `qa/health_check.py` | Cek live tiap sumber: status, jumlah artikel, kualitas tanggal, keberhasilan ambil isi, error HTTP/blokir | Ya |
| `qa/benchmark.py` | Ukur durasi end-to-end pencarian (fase cari vs ambil isi, sumber paling lambat) | Ya |
| `qa/capture_fixtures.py` | Rekam respons live jadi fixture untuk test offline | Ya |
| `.github/workflows/scraper-qa.yml` | Menjalankan semuanya otomatis setiap hari & menerbitkan laporan | — |

## Pemasangan

Salin isi paket ini ke root repo (sejajar dengan `app.py`), lalu:

```bash
pip install -r requirements.txt -r qa/requirements-qa.txt
```

`pytest` sengaja tidak dimasukkan ke `requirements.txt` supaya tidak ikut ter-install di Streamlit Cloud.

## Menjalankan secara lokal

```bash
python -m pytest                                   # unit test offline (±3 detik)
python -m qa.health_check --label laptop           # cek live 9 sumber (±2-5 menit)
python -m qa.benchmark --skenario cepat default_app --label laptop
python -m qa.capture_fixtures                      # rekam fixture (saat scraper sedang sehat)
```

Opsi berguna untuk health check: `--sumber Detik Kompas`, `--tema prabowo`, `--hari 7`,
`--halaman 2`, `--sampel-konten 5`.

Laporan tersimpan di folder `reports/`:
`SUMMARY.md` (tabel ringkas), `latest_<label>.json` (detail), `history.csv` (tren antar-run),
`latest_benchmark_<label>.json`, `benchmark_history.csv`.

## Membaca hasil

**Status per sumber (health check):**

| Status | Arti | Langkah |
|---|---|---|
| ✅ ok | Semua metrik di atas ambang | — |
| 🟡 menurun | Ada artikel, tapi tanggal/isi/blokir bermasalah | Lihat kolom Catatan |
| 🟠 kosong_mencurigakan | Request sukses tapi 0 artikel | Hampir pasti selector/format respons berubah |
| ⛔ diblokir | Respons 403/429/503 atau halaman Cloudflare | Bandingkan hasil laptop vs GitHub Actions |
| 🔴 http_error | Semua request gagal (timeout, koneksi, 4xx/5xx) | Cek URL endpoint / situs down |
| 💥 error | Scraper melempar exception | Lihat `traceback` di JSON |

Ambang bisa diubah di `AMBANG` dalam `qa/health_check.py`.

**Cara membedakan "kode rusak" vs "situs berubah/memblokir":**

- Test replay fixture **lulus** tapi health check **gagal** → situsnya berubah atau memblokir, bukan kode Anda.
- Test replay fixture **gagal** → perubahan kode Anda merusak parser.
- Health check gagal di GitHub Actions tapi lulus di laptop → IP server luar negeri diblokir
  (kemungkinan besar Streamlit Cloud juga terkena).

**Unit test yang saat ini gagal itu disengaja.** Test tersebut mendokumentasikan bug yang sudah
ditemukan (singkatan bulan, waktu relatif, zona waktu Detik, encoding keyword Detik/Kompas, pesan
error Tempo dianalisis sentimen, pencocokan potongan kata di Kumparan). Saat bug diperbaiki, test
akan lulus. Jangan ubah test agar lulus; ubah kodenya.

## Otomatis dengan GitHub Actions

Workflow `scraper-qa.yml` berjalan:
- setiap hari 06:00 WIB: unit test + health check
- setiap Senin 08:00 WIB: + benchmark `default_app`
- manual kapan saja: tab **Actions → Scraper QA → Run workflow** (bisa memilih skenario benchmark)

Laporan diterbitkan ke branch terpisah **`qa-reports`** (folder `reports/`), bukan ke `main`,
sehingga **tidak memicu redeploy** aplikasi di Streamlit Cloud dan tidak mengotori riwayat kode.

Pengaturan yang perlu dicek sekali di GitHub:
**Settings → Actions → General → Workflow permissions → Read and write permissions.**

Catatan: GitHub dapat menonaktifkan jadwal workflow pada repo yang lama tidak ada aktivitas.
Jika jadwal berhenti, aktifkan lagi dari tab Actions.

## Catatan penting tentang fixture

Fixture (`tests/fixtures/*/tape.json`) berisi salinan mentah halaman berita. Jika repo Anda
**publik**, jangan commit fixture ke repo (konten berita berhak cipta). Tambahkan
`tests/fixtures/` ke `.gitignore` dan simpan fixture hanya di laptop. Test replay akan otomatis
di-skip di GitHub Actions jika fixture tidak ada.

## Menambah sumber baru

1. Tambahkan fungsi scraper dan daftarkan di `SCRAPER_PER_SUMBER` (dan `SUMBER_DENGAN_FILTER_TANGGAL_NATIVE` jika mendukung filter tanggal).
2. `python -m qa.health_check --sumber "Nama Sumber"` sampai statusnya ✅.
3. `python -m qa.capture_fixtures --sumber "Nama Sumber"`.
4. `python -m pytest` — sumber baru otomatis masuk test encoding keyword & replay.
