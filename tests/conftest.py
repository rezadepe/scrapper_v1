import os
import time

import pytest

from qa.engine_loader import load_engine


@pytest.fixture(scope="session")
def engine():
    return load_engine()


@pytest.fixture
def server_utc():
    """Simulasikan server Streamlit Cloud yang jamnya UTC."""
    lama = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    yield
    if lama is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = lama
    time.tzset()


@pytest.fixture
def tangkap_url(engine, monkeypatch):
    """Ganti ambil_halaman: catat URL yang diminta, kembalikan None (tanpa internet)."""
    daftar = []

    def palsu(url, timeout=10, progress_callback=None):
        daftar.append(url)
        return None

    monkeypatch.setattr(engine, "ambil_halaman", palsu)
    return daftar
