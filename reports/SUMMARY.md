# Laporan Health Check Scraper

- Waktu: 2026-11-25 14:45 WIB (durasi 51.7 detik)
- Lingkungan: `github-actions` | GitHub Actions: True | commit app: `daeb81c`
- Parameter: tema `indonesia`, 2026-11-23 s/d 2026-11-25, 1 halaman, sampel isi 3 artikel/sumber

| Sumber | Status | Artikel | Tgl terbaca | Dlm rentang | Isi OK | Req (gagal) | Durasi | Catatan |
|---|---|---|---|---|---|---|---|---|
| CNBC Indonesia | ✅ ok | 10 | 100.0% | 100.0% | 100.0% | 1 (0) | 1.57s |  |
| CNN Indonesia | ✅ ok | 10 | 100.0% | 100.0% | 100.0% | 1 (0) | 1.92s |  |
| Detik | ✅ ok | 12 | 100.0% | 100.0% | 100.0% | 4 (0) | 9.51s |  |
| Kompas | 🟡 menurun | 20 | 100.0% | 10.0% | 66.7% | 4 (0) | 8.43s | Hanya 10.0% tanggal dalam rentang; Isi artikel berhasil diambil hanya 66.7% |
| Republika | 🟡 menurun | 22 | 100.0% | 77.3% | 100.0% | 4 (0) | 9.14s | Hanya 77.3% tanggal dalam rentang; 4.5% tanggal di masa depan (salah parse / zona waktu) |
| Liputan6 | 🟠 kosong_mencurigakan | 0 | -% | -% | -% | 1 (0) | 2.28s | Request berhasil tapi 0 artikel -- kemungkinan selector/format respons berubah |
| Kumparan | 🟡 menurun | 129 | 100.0% | 34.9% | 100.0% | 13 (0) | 9.68s | Hanya 34.9% tanggal dalam rentang |
| Tempo.co | ⛔ diblokir | 0 | -% | -% | -% | 2 (2) | 2.7s | 2 respons terlihat diblokir (403/429/503/Cloudflare) |
| The Conversation | 🟡 menurun | 4 | 100.0% | 100.0% | 0.0% | 4 (0) | 6.43s | Isi artikel berhasil diambil hanya 0.0% |
