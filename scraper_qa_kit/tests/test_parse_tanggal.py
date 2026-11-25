"""Parser tanggal: harus benar atau None -- TIDAK BOLEH menebak tanggal yang salah."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest


@pytest.mark.parametrize("teks, harapan", [
    ("Senin, 12 Oktober 2026 14:30 WIB", datetime(2026, 10, 12, 14, 30)),
    ("Minggu, 1 Maret 2026", datetime(2026, 3, 1)),
    ("Kompas.com - 07/11/2026, 10:15 WIB", datetime(2026, 11, 7, 10, 15)),
])
def test_format_lengkap(engine, teks, harapan):
    assert engine.parse_tanggal(teks) == harapan


@pytest.mark.parametrize("teks, harapan", [
    ("12 Okt 2026", datetime(2026, 10, 12)),
    ("3 Des 2026, 08:00", datetime(2026, 12, 3, 8, 0)),
    ("17 Agu 2026", datetime(2026, 8, 17)),
    ("5 Mei 2026 19:00 WIB", datetime(2026, 5, 5, 19, 0)),
])
def test_singkatan_bulan_indonesia(engine, teks, harapan):
    """Banyak situs memakai singkatan bulan. Hasil salah = artikel masuk tanggal keliru."""
    assert engine.parse_tanggal(teks) == harapan


@pytest.mark.parametrize("teks", ["2 jam yang lalu", "30 menit yang lalu", "1 hari yang lalu"])
def test_waktu_relatif_tidak_menebak_salah(engine, teks):
    """Boleh None (ditandai 'cek manual'), atau mendekati waktu sekarang (WIB). Tidak boleh tanggal acak."""
    hasil = engine.parse_tanggal(teks)
    sekarang = datetime.now(ZoneInfo("Asia/Jakarta")).replace(tzinfo=None)
    assert hasil is None or abs(hasil - sekarang) <= timedelta(days=1, hours=1), f"{teks!r} -> {hasil}"


@pytest.mark.parametrize("teks", ["", None, "No date", "Tanggal tidak tersedia"])
def test_teks_tanpa_tanggal_jadi_none(engine, teks):
    assert engine.parse_tanggal(teks) is None
