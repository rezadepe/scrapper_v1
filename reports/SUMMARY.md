# Laporan Health Check Scraper

- Waktu: 2026-11-26 11:44 WIB (durasi 59.6 detik)
- Lingkungan: `github-actions` | GitHub Actions: True | commit app: `daeb81c`
- Parameter: tema `indonesia`, 2026-11-24 s/d 2026-11-26, 1 halaman, sampel isi 3 artikel/sumber

| Sumber | Status | Artikel | Tgl terbaca | Dlm rentang | Isi OK | Req (gagal) | Durasi | Catatan |
|---|---|---|---|---|---|---|---|---|
| CNBC Indonesia | ✅ ok | 10 | 100.0% | 100.0% | 100.0% | 1 (0) | 1.42s |  |
| CNN Indonesia | ✅ ok | 10 | 100.0% | 100.0% | 100.0% | 1 (0) | 2.55s |  |
| Detik | ✅ ok | 12 | 100.0% | 100.0% | 100.0% | 4 (0) | 19.48s |  |
| Kompas | 🟡 menurun | 20 | 100.0% | 0.0% | 66.7% | 4 (0) | 8.08s | Hanya 0.0% tanggal dalam rentang; Isi artikel berhasil diambil hanya 66.7% |
| Republika | 🟡 menurun | 22 | 100.0% | 50.0% | 100.0% | 4 (0) | 10.33s | Hanya 50.0% tanggal dalam rentang |
| Liputan6 | 🟠 kosong_mencurigakan | 0 | -% | -% | -% | 1 (0) | 2.46s | Request berhasil tapi 0 artikel -- kemungkinan selector/format respons berubah |
| Kumparan | 🟡 menurun | 119 | 100.0% | 31.1% | 100.0% | 13 (0) | 8.35s | Hanya 31.1% tanggal dalam rentang |
| Tempo.co | ⛔ diblokir | 0 | -% | -% | -% | 2 (2) | 1.92s | 2 respons terlihat diblokir (403/429/503/Cloudflare) |
| The Conversation | 🟡 menurun | 3 | 100.0% | 100.0% | 0.0% | 4 (0) | 5.02s | Isi artikel berhasil diambil hanya 0.0% |
