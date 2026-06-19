import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from datetime import datetime
import time
import random
import csv
import json
import re
import io

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(page_title="Scraper Berita Indonesia", page_icon="📰", layout="wide")

# =========================================================
# SESSION & UTIL (sama seperti versi notebook)
# =========================================================
@st.cache_resource
def buat_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    return session

SESSION = buat_session()

def ambil_halaman(url, timeout=10):
    try:
        time.sleep(random.uniform(0.8, 1.6))
        resp = SESSION.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
        return None
    except requests.RequestException:
        return None

def parse_tanggal(teks_tanggal):
    if not teks_tanggal or teks_tanggal == "No date":
        return None
    teks = teks_tanggal.strip()
    teks = re.sub(r"\b(WIB|WITA|WIT)\b", "", teks).strip()
    bulan_id = {
        "Januari": "January", "Februari": "February", "Maret": "March", "April": "April",
        "Mei": "May", "Juni": "June", "Juli": "July", "Agustus": "August",
        "September": "September", "Oktober": "October", "November": "November", "Desember": "December"
    }
    for id_, en in bulan_id.items():
        teks = teks.replace(id_, en)
    try:
        return dateparser.parse(teks, fuzzy=True, dayfirst=True)
    except (ValueError, OverflowError):
        return None

# =========================================================
# SCRAPER PER SUMBER
# =========================================================
def scrape_cnbcindonesia(tema, max_halaman=3):
    hasil = []
    for p in range(1, max_halaman + 1):
        url = f"https://www.cnbcindonesia.com/search?query={tema}&p={p}"
        html = ambil_halaman(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        artikel_list = soup.find_all("article")
        if not artikel_list:
            break
        for art in artikel_list:
            judul_tag = art.find("h2")
            link_tag = art.find("a", href=True)
            tanggal_tag = art.find("div", class_="text-cm") or art.find("span", class_="date")
            judul = judul_tag.get_text(strip=True) if judul_tag else None
            link = link_tag["href"] if link_tag else None
            tanggal_teks = tanggal_tag.get_text(strip=True) if tanggal_tag else None
            if judul and link:
                hasil.append({
                    "sumber": "CNBC Indonesia", "judul": judul, "link": link,
                    "tanggal_teks": tanggal_teks, "tanggal": parse_tanggal(tanggal_teks)
                })
    return hasil

def scrape_detik(tema, max_halaman=3, tanggal_mulai=None, tanggal_akhir=None):
    """
    Scraper Detik versi perbaikan berdasarkan struktur HTML asli halaman pencarian.
    tanggal_mulai / tanggal_akhir: objek date/datetime Python (opsional).
    Detik mendukung filter tanggal langsung lewat parameter fromdatex/todatex (format DD/MM/YYYY),
    jadi kita serahkan filtering tanggal ke server Detik -- lebih akurat & lebih cepat.
    """
    hasil = []

    fromdatex = tanggal_mulai.strftime("%d/%m/%Y") if tanggal_mulai else None
    todatex = tanggal_akhir.strftime("%d/%m/%Y") if tanggal_akhir else None

    for p in range(1, max_halaman + 1):
        url = f"https://www.detik.com/search/searchall?query={tema}&result_type=latest&page={p}"
        if fromdatex and todatex:
            from urllib.parse import quote
            url += f"&fromdatex={quote(fromdatex, safe='')}&todatex={quote(todatex, safe='')}"

        html = ambil_halaman(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        artikel_list = soup.select("article.list-content__item")
        if not artikel_list:
            break

        for art in artikel_list:
            judul_tag = art.select_one("h3.media__title a")
            sumber_tag = art.select_one("h2.media__subtitle")
            desc_tag = art.select_one("div.media__desc")
            tanggal_span = art.select_one("div.media__date span")

            if not judul_tag:
                continue

            judul = judul_tag.get_text(strip=True)
            link = judul_tag.get("href")

            tanggal_obj = None
            tanggal_teks = None
            if tanggal_span:
                # d-time berisi unix timestamp -- jauh lebih akurat daripada teks "2 jam yang lalu"
                unix_ts = tanggal_span.get("d-time")
                tanggal_teks = tanggal_span.get("title") or tanggal_span.get_text(strip=True)
                if unix_ts:
                    try:
                        tanggal_obj = datetime.fromtimestamp(int(unix_ts))
                    except (ValueError, OverflowError):
                        tanggal_obj = parse_tanggal(tanggal_teks)
                else:
                    tanggal_obj = parse_tanggal(tanggal_teks)

            if judul and link:
                hasil.append({
                    "sumber": f"Detik ({sumber_tag.get_text(strip=True) if sumber_tag else '-'})",
                    "judul": judul,
                    "link": link,
                    "ringkasan": desc_tag.get_text(strip=True) if desc_tag else None,
                    "tanggal_teks": tanggal_teks,
                    "tanggal": tanggal_obj
                })
    return hasil

def scrape_kompas(tema, max_halaman=3):
    hasil = []
    for p in range(1, max_halaman + 1):
        url = f"https://search.kompas.com/search?q={tema}&page={p}"
        html = ambil_halaman(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        artikel_list = soup.select("div.articleItem") or soup.select("a.article-link")
        if not artikel_list:
            break
        for art in artikel_list:
            judul_tag = art.find("h2", class_="articleTitle") or art.find("h2")
            link_tag = art if art.name == "a" else art.find("a", href=True)
            tanggal_tag = art.find("div", class_="articlePost-date") or art.find("div", class_="date")
            judul = judul_tag.get_text(strip=True) if judul_tag else (art.get_text(strip=True) if art.name == "a" else None)
            link = link_tag["href"] if link_tag and link_tag.has_attr("href") else None
            tanggal_teks = tanggal_tag.get_text(strip=True) if tanggal_tag else None
            if judul and link:
                hasil.append({
                    "sumber": "Kompas", "judul": judul, "link": link,
                    "tanggal_teks": tanggal_teks, "tanggal": parse_tanggal(tanggal_teks)
                })
    return hasil

SCRAPER_PER_SUMBER = {"CNBC Indonesia": scrape_cnbcindonesia, "Detik": scrape_detik, "Kompas": scrape_kompas}

# =========================================================
# AMBIL ISI ARTIKEL
# =========================================================
SELECTOR_KONTEN = {
    "cnbcindonesia.com": ("div", {"class_": "detail-text"}),
    "detik.com": ("div", {"class_": "detail__body-text"}),
    "kompas.com": ("div", {"class_": "read__content"}),
}

def scrape_isi_artikel(url):
    html = ambil_halaman(url)
    if not html:
        return "Gagal mengambil halaman"
    soup = BeautifulSoup(html, "html.parser")
    domain_cocok = next((d for d in SELECTOR_KONTEN if d in url), None)
    if domain_cocok:
        tag, attrs = SELECTOR_KONTEN[domain_cocok]
        kontainer = soup.find(tag, **attrs)
        if kontainer:
            for unwanted in kontainer.find_all(["script", "style", "div"], class_=re.compile("baca-juga|ads|related", re.I)):
                unwanted.decompose()
            paragraf = kontainer.find_all("p")
            isi = " ".join(p.get_text(strip=True) for p in paragraf)
            return isi if isi.strip() else "Konten kosong"
    return "Selector tidak ditemukan untuk domain ini"

# =========================================================
# FUNGSI UTAMA PENCARIAN
# =========================================================
def cari_berita(tema, sumber_dipilih, tanggal_mulai, tanggal_akhir, max_halaman, ambil_konten, progress_callback=None):
    semua_hasil = []
    for nama_sumber in sumber_dipilih:
        fungsi = SCRAPER_PER_SUMBER[nama_sumber]
        if progress_callback:
            progress_callback(f"Mengambil dari {nama_sumber}...")
        try:
            if nama_sumber == "Detik":
                # Detik mendukung filter tanggal langsung di request (lebih akurat)
                semua_hasil.extend(fungsi(
                    tema, max_halaman=max_halaman,
                    tanggal_mulai=tanggal_mulai, tanggal_akhir=tanggal_akhir
                ))
            else:
                semua_hasil.extend(fungsi(tema, max_halaman=max_halaman))
        except Exception as e:
            if progress_callback:
                progress_callback(f"[ERROR] {nama_sumber}: {e}")

    sudah_dilihat = set()
    hasil_unik = []
    for item in semua_hasil:
        if item["link"] not in sudah_dilihat:
            sudah_dilihat.add(item["link"])
            hasil_unik.append(item)

    dt_mulai = datetime.combine(tanggal_mulai, datetime.min.time()) if tanggal_mulai else None
    dt_akhir = datetime.combine(tanggal_akhir, datetime.min.time()) if tanggal_akhir else None

    hasil_final = []
    for item in hasil_unik:
        tgl = item["tanggal"]
        if dt_mulai and dt_akhir:
            if tgl is None:
                item["catatan"] = "Tanggal tidak terbaca, cek manual"
                hasil_final.append(item)
            elif dt_mulai <= tgl <= dt_akhir:
                hasil_final.append(item)
        else:
            hasil_final.append(item)

    if ambil_konten:
        total = len(hasil_final)
        for i, item in enumerate(hasil_final, 1):
            if progress_callback:
                progress_callback(f"Mengambil isi [{i}/{total}]: {item['judul'][:60]}...")
            item["konten"] = scrape_isi_artikel(item["link"])

    return hasil_final

# =========================================================
# UI STREAMLIT
# =========================================================
st.title("📰 Scraper Berita Indonesia")
st.caption("Cari berita dari CNBC Indonesia, Detik, dan Kompas berdasarkan tema dan rentang tanggal — gratis, tanpa API berbayar.")

with st.sidebar:
    st.header("⚙️ Pengaturan Pencarian")
    tema = st.text_input("Tema / kata kunci berita", placeholder="contoh: kaesang")
    sumber_dipilih = st.multiselect(
        "Sumber berita",
        options=list(SCRAPER_PER_SUMBER.keys()),
        default=list(SCRAPER_PER_SUMBER.keys())
    )
    pakai_filter_tanggal = st.checkbox("Filter berdasarkan tanggal", value=False)
    tanggal_mulai, tanggal_akhir = None, None
    if pakai_filter_tanggal:
        col1, col2 = st.columns(2)
        with col1:
            tanggal_mulai = st.date_input("Dari tanggal")
        with col2:
            tanggal_akhir = st.date_input("Sampai tanggal")
    max_halaman = st.slider("Jumlah halaman per sumber", min_value=1, max_value=10, value=3)
    ambil_konten = st.checkbox("Ambil isi lengkap artikel (lebih lambat)", value=True)
    tombol_cari = st.button("🔍 Cari Berita", type="primary", use_container_width=True)

if "hasil" not in st.session_state:
    st.session_state.hasil = []

if tombol_cari:
    if not tema.strip():
        st.warning("Masukkan tema berita dulu ya.")
    elif not sumber_dipilih:
        st.warning("Pilih minimal satu sumber berita.")
    else:
        status_box = st.empty()
        progress_log = []

        def progress_callback(msg):
            progress_log.append(msg)
            status_box.info("\n\n".join(progress_log[-5:]))  # tampilkan 5 log terakhir

        with st.spinner("Sedang mengambil data..."):
            hasil = cari_berita(
                tema=tema,
                sumber_dipilih=sumber_dipilih,
                tanggal_mulai=tanggal_mulai,
                tanggal_akhir=tanggal_akhir,
                max_halaman=max_halaman,
                ambil_konten=ambil_konten,
                progress_callback=progress_callback,
            )
        status_box.empty()
        st.session_state.hasil = hasil
        st.success(f"Selesai! Ditemukan {len(hasil)} artikel unik.")

# =========================================================
# TAMPILKAN HASIL
# =========================================================
hasil = st.session_state.hasil

if hasil:
    st.subheader(f"Hasil: {len(hasil)} artikel")

    # Filter tampilan per sumber
    sumber_unik = sorted(set(h["sumber"] for h in hasil))
    filter_tampil = st.multiselect("Tampilkan sumber:", sumber_unik, default=sumber_unik)
    hasil_tampil = [h for h in hasil if h["sumber"] in filter_tampil]

    for item in hasil_tampil:
        tgl_str = item["tanggal"].strftime("%d %B %Y") if item.get("tanggal") else (item.get("tanggal_teks") or "Tanggal tidak diketahui")
        with st.expander(f"📌 {item['judul']}  —  {item['sumber']} ({tgl_str})"):
            st.markdown(f"**Link:** [{item['link']}]({item['link']})")
            if item.get("catatan"):
                st.warning(item["catatan"])
            if "konten" in item:
                st.write(item["konten"])

    # ---------------- EKSPOR ----------------
    st.subheader("⬇️ Unduh Hasil")

    def buat_csv(data):
        output = io.StringIO()
        kolom = ["sumber", "judul", "link", "tanggal_teks", "tanggal", "konten", "catatan"]
        writer = csv.DictWriter(output, fieldnames=kolom, extrasaction="ignore")
        writer.writeheader()
        for item in data:
            baris = dict(item)
            if isinstance(baris.get("tanggal"), datetime):
                baris["tanggal"] = baris["tanggal"].strftime("%Y-%m-%d")
            writer.writerow(baris)
        return output.getvalue()

    def buat_json(data):
        salinan_semua = []
        for item in data:
            salinan = dict(item)
            if isinstance(salinan.get("tanggal"), datetime):
                salinan["tanggal"] = salinan["tanggal"].strftime("%Y-%m-%d")
            salinan_semua.append(salinan)
        return json.dumps(salinan_semua, ensure_ascii=False, indent=2)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Unduh sebagai CSV",
            data=buat_csv(hasil_tampil),
            file_name=f"berita_{tema or 'hasil'}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Unduh sebagai JSON",
            data=buat_json(hasil_tampil),
            file_name=f"berita_{tema or 'hasil'}.json",
            mime="application/json",
            use_container_width=True,
        )
else:
    st.info("Masukkan tema di sidebar lalu klik **Cari Berita** untuk mulai.")
