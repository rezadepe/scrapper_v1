"""
engine_loader.py
Memuat fungsi-fungsi engine dari app.py TANPA menjalankan UI Streamlit.

Cara kerja:
- app.py dipotong tepat sebelum blok "# UI STREAMLIT" (semua fungsi scraper,
  parser, sentimen, dll. ada di atas blok itu).
- Modul `streamlit` diganti sementara dengan versi palsu (no-op), supaya
  st.set_page_config, st.secrets, @st.cache_data, dll. tidak error dan tidak
  butuh server Streamlit.
- Hasilnya adalah modul Python biasa: engine.scrape_detik(...), engine.parse_tanggal(...),
  engine.cari_berita(...), dst. Fungsi bisa di-monkeypatch untuk testing.

Path app.py bisa diganti lewat environment variable QA_APP_PATH.
"""
import os
import re
import sys
import types

_ENGINE = None
_PENANDA_UI = re.compile(r"^#\s*UI STREAMLIT", re.MULTILINE)


class _Apapun:
    """Objek yang menerima atribut/panggilan apa saja dan tidak melakukan apa-apa."""
    def __getattr__(self, nama):
        return _Apapun()

    def __call__(self, *args, **kwargs):
        return _Apapun()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False


class _SessionState(dict):
    def __getattr__(self, nama):
        try:
            return self[nama]
        except KeyError as e:
            raise AttributeError(nama) from e

    def __setattr__(self, nama, nilai):
        self[nama] = nilai


def _dekorator_cache(*args, **kwargs):
    # Mendukung @st.cache_data maupun @st.cache_data(ttl=..., ...)
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda fungsi: fungsi


def _buat_streamlit_palsu():
    st = types.ModuleType("streamlit")
    st.session_state = _SessionState()
    st.secrets = {}
    st.cache_data = _dekorator_cache
    st.cache_resource = _dekorator_cache
    st.__getattr__ = lambda nama: _Apapun()  # PEP 562: atribut lain jadi no-op
    return st


def path_app_default():
    return os.environ.get(
        "QA_APP_PATH",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"),
    )


def load_engine(app_path=None, paksa_ulang=False):
    """Return modul engine (singleton). paksa_ulang=True untuk memuat ulang."""
    global _ENGINE
    if _ENGINE is not None and not paksa_ulang:
        return _ENGINE

    app_path = os.path.abspath(app_path or path_app_default())
    with open(app_path, encoding="utf-8") as f:
        sumber = f.read()

    cocok = _PENANDA_UI.search(sumber)
    if not cocok:
        raise RuntimeError(
            "Penanda '# UI STREAMLIT' tidak ditemukan di app.py. "
            "Loader butuh penanda ini untuk memisahkan engine dari UI."
        )
    # Mundur ke baris '# ====' tepat sebelum penanda supaya potongannya rapi.
    potongan = sumber[: cocok.start()]
    potongan = potongan[: potongan.rfind("\n# ===") + 1] if "\n# ===" in potongan else potongan

    app_dir = os.path.dirname(app_path)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)  # supaya `from event_tracker import log_event` jalan

    st_asli = sys.modules.get("streamlit")
    sys.modules["streamlit"] = _buat_streamlit_palsu()
    try:
        engine = types.ModuleType("scraper_engine_dari_app")
        engine.__file__ = app_path
        # compile dengan nama file app.py -> nomor baris traceback sesuai app.py asli
        exec(compile(potongan, app_path, "exec"), engine.__dict__)
    finally:
        if st_asli is not None:
            sys.modules["streamlit"] = st_asli
        else:
            sys.modules.pop("streamlit", None)

    _ENGINE = engine
    return engine


def daftar_sumber(engine=None):
    engine = engine or load_engine()
    return list(engine.SCRAPER_PER_SUMBER.keys())


def panggil_scraper(engine, nama_sumber, tema, max_halaman, tanggal_mulai, tanggal_akhir, progress_callback=None):
    """Panggil scraper satu sumber dengan aturan argumen yang sama seperti app.py."""
    fungsi = engine.SCRAPER_PER_SUMBER[nama_sumber]
    if nama_sumber in engine.SUMBER_DENGAN_FILTER_TANGGAL_NATIVE:
        return fungsi(
            tema, max_halaman=max_halaman,
            tanggal_mulai=tanggal_mulai, tanggal_akhir=tanggal_akhir,
            progress_callback=progress_callback,
        )
    return fungsi(tema, max_halaman=max_halaman, progress_callback=progress_callback)


def slug(nama_sumber):
    return re.sub(r"[^a-z0-9]+", "_", nama_sumber.lower()).strip("_")
