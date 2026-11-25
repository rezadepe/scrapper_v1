"""Tanggal harus dalam WIB, apa pun zona waktu server (Streamlit Cloud = UTC)."""
import calendar
from datetime import date, datetime

TS_25NOV_0030_WIB = calendar.timegm(datetime(2026, 11, 24, 17, 30).timetuple())  # = 25 Nov 00:30 WIB

HTML_DETIK = f"""
<article class="list-content__item">
  <h3 class="media__title"><a href="https://news.detik.com/berita/d-1/contoh">Judul contoh</a></h3>
  <h2 class="media__subtitle">detikNews</h2>
  <div class="media__date"><span d-time="{TS_25NOV_0030_WIB}" title="Rabu, 25 Nov 2026 00:30 WIB"></span></div>
</article>"""


def test_detik_tanggal_wib_di_server_utc(engine, monkeypatch, server_utc):
    monkeypatch.setattr(engine, "ambil_halaman", lambda *a, **k: HTML_DETIK)
    hasil = engine.scrape_detik("contoh", max_halaman=1)
    assert hasil[0]["tanggal"] == datetime(2026, 11, 25, 0, 30)


def test_detik_artikel_dini_hari_tidak_terbuang(engine, monkeypatch, server_utc):
    monkeypatch.setattr(engine, "ambil_halaman", lambda *a, **k: HTML_DETIK)
    hasil = engine.cari_berita("contoh", ["Detik"], date(2026, 11, 25), date(2026, 11, 25), 1, ambil_konten=False)
    assert len(hasil) == 1, "Artikel 25 Nov 00:30 WIB terbuang dari filter tanggal 25 Nov"
