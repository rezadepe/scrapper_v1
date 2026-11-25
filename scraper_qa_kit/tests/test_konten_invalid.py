"""Pesan error tidak boleh diperlakukan sebagai isi artikel (diringkas / dianalisis sentimen)."""
import pytest

from qa.http_probe import HttpProbe, _url_lengkap, kunci_request

URL_TEMPO = "https://www.tempo.co/politik/contoh-artikel-123"


def _tape(url, status, teks):
    return {kunci_request("GET", _url_lengkap("GET", url, None), ""): {"status": status, "text": teks}}


@pytest.mark.parametrize("status, html", [
    (403, ""),                                        # diblokir Cloudflare
    (200, "<html><body><p>tanpa wrapper</p></body></html>"),  # struktur halaman berubah
])
def test_error_tempo_dikenali_sebagai_konten_invalid(engine, status, html):
    with HttpProbe("replay", tape=_tape(URL_TEMPO, status, html)):
        hasil = engine.scrape_isi_artikel(URL_TEMPO)
    assert hasil in engine.KONTEN_INVALID, f"{hasil!r} tidak ada di KONTEN_INVALID"


def test_halaman_gagal_generik_dikenali(engine, monkeypatch):
    monkeypatch.setattr(engine, "ambil_halaman", lambda *a, **k: None)
    assert engine.scrape_isi_artikel("https://news.detik.com/berita/d-1/x") in engine.KONTEN_INVALID


def test_pesan_error_tidak_dianalisis_sentimen(engine, monkeypatch):
    pesan_error = "Gagal mengambil halaman (HTTP 403)"
    artikel = {"sumber": "Uji", "judul": "Pemerintah umumkan program baru",
               "link": "https://contoh.id/1", "tanggal": None, "tanggal_teks": None}
    monkeypatch.setitem(engine.SCRAPER_PER_SUMBER, "Uji", lambda *a, **k: [dict(artikel)])
    monkeypatch.setattr(engine, "scrape_isi_artikel", lambda *a, **k: pesan_error)

    teks_dianalisis = []
    asli = engine.analisis_sentimen
    monkeypatch.setattr(engine, "analisis_sentimen", lambda t: (teks_dianalisis.append(t), asli(t))[1])

    hasil = engine.cari_berita("program", ["Uji"], None, None, 1, ambil_konten=True)
    assert hasil, "artikel uji hilang"
    assert pesan_error not in teks_dianalisis, "Sentimen dihitung dari pesan error, bukan dari berita"
    assert pesan_error not in (hasil[0].get("ringkasan") or "")
