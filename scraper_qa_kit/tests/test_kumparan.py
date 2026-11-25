"""Pencocokan keyword Kumparan harus per kata, bukan potongan kata."""
XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
 <url><loc>https://kumparan.com/a/1</loc><news:news><news:publication_date>2026-11-25T08:00:00+07:00</news:publication_date>
  <news:title>Pakar psikologi soroti tekanan kerja</news:title><news:keywords>kesehatan mental</news:keywords></news:news></url>
 <url><loc>https://kumparan.com/a/2</loc><news:news><news:publication_date>2026-11-25T09:00:00+07:00</news:publication_date>
  <news:title>PSI umumkan calon kepala daerah</news:title><news:keywords>politik, pilkada</news:keywords></news:news></url>
</urlset>"""


def test_kumparan_tidak_cocok_potongan_kata(engine, monkeypatch):
    monkeypatch.setattr(engine, "KUMPARAN_SITEMAP_CHANNELS", ["https://kumparan.com/sitemap_uji.xml"])
    monkeypatch.setattr(engine, "ambil_halaman", lambda *a, **k: XML)
    hasil = engine.scrape_kumparan("PSI")
    assert [a["link"] for a in hasil] == ["https://kumparan.com/a/2"]


def test_kumparan_tanggal_wib(engine, monkeypatch):
    monkeypatch.setattr(engine, "KUMPARAN_SITEMAP_CHANNELS", ["https://kumparan.com/sitemap_uji.xml"])
    monkeypatch.setattr(engine, "ambil_halaman", lambda *a, **k: XML)
    hasil = {a["link"]: a["tanggal"] for a in engine.scrape_kumparan("PSI")}
    assert hasil["https://kumparan.com/a/2"].hour == 9
