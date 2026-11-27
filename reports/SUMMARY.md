# Laporan Health Check Scraper

- Waktu: 2026-11-27 11:51 WIB (durasi 74.1 detik)
- Lingkungan: `github-actions` | GitHub Actions: True | commit app: `daeb81c`
- Parameter: tema `indonesia`, 2026-11-25 s/d 2026-11-27, 1 halaman, sampel isi 3 artikel/sumber

| Sumber | Status | Artikel | Tgl terbaca | Dlm rentang | Isi OK | Req (gagal) | Durasi | Catatan |
|---|---|---|---|---|---|---|---|---|
| CNBC Indonesia | ✅ ok | 10 | 100.0% | 100.0% | 100.0% | 1 (0) | 1.65s |  |
| CNN Indonesia | ✅ ok | 10 | 100.0% | 100.0% | 100.0% | 1 (0) | 1.73s |  |
| Detik | 🟡 menurun | 12 | 100.0% | 100.0% | 66.7% | 4 (0) | 33.08s | Isi artikel berhasil diambil hanya 66.7% |
| Kompas | 🟡 menurun | 20 | 100.0% | 20.0% | 66.7% | 4 (0) | 9.26s | Hanya 20.0% tanggal dalam rentang; Isi artikel berhasil diambil hanya 66.7% |
| Republika | 🟡 menurun | 21 | 100.0% | 66.7% | 100.0% | 4 (0) | 9.01s | Hanya 66.7% tanggal dalam rentang |
| Liputan6 | 🟠 kosong_mencurigakan | 0 | -% | -% | -% | 1 (0) | 1.98s | Request berhasil tapi 0 artikel -- kemungkinan selector/format respons berubah |
| Kumparan | 🟡 menurun | 120 | 100.0% | 35.0% | 100.0% | 13 (0) | 10.16s | Hanya 35.0% tanggal dalam rentang |
| Tempo.co | ⛔ diblokir | 0 | -% | -% | -% | 2 (2) | 2.03s | 2 respons terlihat diblokir (403/429/503/Cloudflare) |
| The Conversation | 🟡 menurun | 3 | 100.0% | 100.0% | 0.0% | 4 (0) | 5.15s | Isi artikel berhasil diambil hanya 0.0% |
