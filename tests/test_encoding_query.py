"""Keyword harus dikirim utuh ke setiap situs, termasuk karakter khusus (&, #, +, spasi)."""
import json
from datetime import date
from urllib.parse import parse_qs, urlsplit

import pytest

from qa.engine_loader import load_engine, panggil_scraper
from qa.http_probe import HttpProbe

KEYWORD = "BI & OJK #rupiah +5%"
SUMBER_DENGAN_QUERY = [s for s in load_engine().SCRAPER_PER_SUMBER if s != "Kumparan"]  # Kumparan pakai sitemap


def _keyword_terkirim(miss):
    for m in miss:
        if m["body"]:
            try:
                if KEYWORD in json.loads(m["body"]).values():
                    return True
            except (ValueError, AttributeError):
                pass
        for nilai in parse_qs(urlsplit(m["url"]).query).values():
            if KEYWORD in nilai:
                return True
    return False


@pytest.mark.parametrize("sumber", SUMBER_DENGAN_QUERY)
def test_keyword_dikirim_utuh(engine, sumber):
    with HttpProbe("replay", tape={}) as probe:  # tape kosong: semua request dicatat, tidak ke internet
        panggil_scraper(engine, sumber, KEYWORD, 1, date(2026, 11, 1), date(2026, 11, 2))
    assert probe.miss, "scraper tidak mengirim request apa pun"
    assert _keyword_terkirim(probe.miss), f"Keyword rusak. Request: {[m['url'] for m in probe.miss][:2]}"
