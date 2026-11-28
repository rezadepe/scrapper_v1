# Laporan Health Check Scraper

- Waktu: 2026-11-28 11:35 WIB (durasi 64.7 detik)
- Lingkungan: `github-actions` | GitHub Actions: True | commit app: `daeb81c`
- Parameter: tema `indonesia`, 2026-11-26 s/d 2026-11-28, 1 halaman, sampel isi 3 artikel/sumber

| Sumber | Status | Artikel | Tgl terbaca | Dlm rentang | Isi OK | Req (gagal) | Durasi | Catatan |
|---|---|---|---|---|---|---|---|---|
| CNBC Indonesia | ✅ ok | 10 | 100.0% | 100.0% | 100.0% | 1 (0) | 1.33s |  |
| CNN Indonesia | ✅ ok | 10 | 100.0% | 100.0% | 100.0% | 1 (0) | 1.79s |  |
| Detik | ✅ ok | 12 | 100.0% | 100.0% | 100.0% | 4 (0) | 29.77s |  |
| Kompas | 🟠 kosong_mencurigakan | 0 | -% | -% | -% | 1 (0) | 1.51s | Request berhasil tapi 0 artikel -- kemungkinan selector/format respons berubah |
| Republika | ✅ ok | 22 | 100.0% | 100.0% | 100.0% | 4 (0) | 10.54s |  |
| Liputan6 | 🟠 kosong_mencurigakan | 0 | -% | -% | -% | 1 (0) | 2.2s | Request berhasil tapi 0 artikel -- kemungkinan selector/format respons berubah |
| Kumparan | 🟡 menurun | 117 | 100.0% | 35.0% | 100.0% | 13 (0) | 10.32s | Hanya 35.0% tanggal dalam rentang |
| Tempo.co | ⛔ diblokir | 0 | -% | -% | -% | 2 (2) | 2.58s | 2 respons terlihat diblokir (403/429/503/Cloudflare) |
| The Conversation | 🟡 menurun | 5 | 100.0% | 100.0% | 0.0% | 4 (0) | 4.67s | Isi artikel berhasil diambil hanya 0.0% |
