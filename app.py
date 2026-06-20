import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote
import time
import random
import csv
import json
import re
import io

import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(page_title="Scraper Berita Indonesia", page_icon="📰", layout="wide")

try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None

try:
    AI_SUMMARY_PASSWORD = st.secrets.get("AI_SUMMARY_PASSWORD")
except Exception:
    AI_SUMMARY_PASSWORD = None

GEMINI_MODEL = "gemini-flash-latest"
PESAN_HUBUNGI_DEVELOPER = (
    "Password salah. Fitur ringkasan AI ini terbatas — "
    "hubungi developer aplikasi untuk mendapatkan akses."
)

# =========================================================
# SESSION & UTIL (sama seperti versi sebelumnya)
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

def bersihkan_isi_html(isi_html):
    """
    Konversi HTML isi artikel jadi teks polos. Pakai separator spasi di get_text()
    SEBELUM strip, supaya kata yang dipisah tag inline (mis. <span> berwarna untuk
    keyword/tag) tidak nempel jadi satu kata (mis. 'DPPPSIAhmad' -> 'DPP PSI Ahmad').
    Juga buang tabel "baca juga"/"pilihan redaksi", iframe chart, dan script.
    """
    if not isi_html:
        return ""
    soup = BeautifulSoup(isi_html, "html.parser")
    for unwanted in soup.find_all(["script", "style", "table", "iframe"]):
        unwanted.decompose()
    paragraf = soup.find_all("p")
    sumber_teks = paragraf if paragraf else [soup]
    teks = " ".join(tag.get_text(" ", strip=True) for tag in sumber_teks)
    return re.sub(r"\s+", " ", teks).strip()

# =========================================================
# RINGKASAN EKSTRAKTIF -- LOKAL, GRATIS, OTOMATIS UNTUK SEMUA ARTIKEL
# Pakai TextRank: cosine similarity antar kalimat (setelah stopword removal
# Bahasa Indonesia via Sastrawi) -> graph -> PageRank -> ambil kalimat top-N.
# Tanpa API, tanpa rate limit, jalan lokal di server Streamlit.
# =========================================================
_stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

KONTEN_INVALID = {
    "Gagal mengambil halaman", "Konten kosong",
    "Selector tidak ditemukan untuk domain ini"
}

def _split_kalimat(teks):
    teks = re.sub(r"\s+", " ", teks).strip()
    # lookahead juga mengizinkan tanda kutip pembuka sebelum huruf besar/angka,
    # supaya kalimat kutipan narasumber (mis. '"...," kata Menteri...') terdeteksi
    # sebagai kalimat baru, bukan nempel ke kalimat sebelumnya.
    kalimat = re.split(r"(?<=[.!?])\s+(?=[\"'\u201c\u201d\u2018\u2019]?[A-Z0-9])", teks)
    return [k.strip() for k in kalimat if len(k.strip()) > 20]

def ringkas_ekstraktif(teks, jumlah_kalimat=3):
    if not teks or len(teks.strip()) < 200:
        return teks.strip() if teks else ""

    kalimat_list = _split_kalimat(teks)
    if len(kalimat_list) <= jumlah_kalimat:
        return " ".join(kalimat_list)

    kalimat_bersih = [_stopword_remover.remove(k.lower()) for k in kalimat_list]
    kalimat_bersih = [k if k.strip() else "kata" for k in kalimat_bersih]

    try:
        vektor = CountVectorizer().fit_transform(kalimat_bersih)
        sim_matrix = cosine_similarity(vektor)
        graph = nx.from_numpy_array(sim_matrix)
        skor = nx.pagerank(graph, max_iter=200)
    except Exception:
        return " ".join(kalimat_list[:jumlah_kalimat])

    ranking = sorted(((skor[i], i) for i in range(len(kalimat_list))), reverse=True)
    top_index = sorted(idx for _, idx in ranking[:jumlah_kalimat])
    return " ".join(kalimat_list[i] for i in top_index)

# =========================================================
# RINGKASAN AI (GEMINI) -- ON-DEMAND, 1 ARTIKEL PER KLIK (FITUR TERBATAS)
# =========================================================
def _panggil_gemini(prompt, api_key, timeout=30):
    """Helper generik buat panggil Gemini API. Dipakai oleh ringkas_gemini dan
    bandingkan_sudut_pandang_gemini biar gak duplikasi logic request/response."""
    if not api_key:
        return None, "API key Gemini belum di-set di Streamlit secrets."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        return None, "Request ke Gemini timeout, coba lagi."
    except requests.exceptions.RequestException as e:
        return None, f"Gagal menghubungi Gemini: {e}"

    if resp.status_code == 429:
        return None, "Rate limit Gemini tercapai, coba lagi beberapa saat lagi."
    if resp.status_code != 200:
        return None, f"Gemini API error (status {resp.status_code})."

    try:
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None, "Gemini tidak mengembalikan hasil (kemungkinan konten difilter safety policy)."
        parts = candidates[0].get("content", {}).get("parts", [])
        hasil_teks = "".join(p.get("text", "") for p in parts).strip()
        if not hasil_teks:
            return None, "Gemini mengembalikan respons kosong."
        return hasil_teks, None
    except (ValueError, KeyError, IndexError):
        return None, "Gagal membaca respons dari Gemini."

def ringkas_gemini(teks, api_key, max_chars=15000):
    if not teks:
        return None, "Tidak ada konten untuk diringkas."
    teks_potong = teks[:max_chars]
    prompt = (
        "Ringkas berita Bahasa Indonesia berikut ini dalam 3-4 kalimat yang jelas dan natural. "
        "Jangan tambahkan opini atau informasi di luar teks yang diberikan. "
        "Langsung tulis ringkasannya saja, tanpa kalimat pembuka seperti 'Berikut ringkasannya:'.\n\n"
        f"Teks berita:\n{teks_potong}"
    )
    return _panggil_gemini(prompt, api_key)

def bandingkan_sudut_pandang_gemini(artikel_list, api_key, max_chars_per_artikel=6000, max_artikel=4):
    """
    Minta Gemini membandingkan framing/sudut pandang dari beberapa artikel yang
    membahas kejadian yang sama tapi ditulis oleh sumber berita berbeda.
    """
    if not artikel_list or len(artikel_list) < 2:
        return None, "Minimal perlu 2 artikel untuk dibandingkan."

    artikel_dipakai = artikel_list[:max_artikel]
    bagian = []
    for art in artikel_dipakai:
        konten = (art.get("konten") or art.get("ringkasan_redaksi") or art.get("judul") or "")
        konten = konten[:max_chars_per_artikel]
        bagian.append(f"=== Sumber: {art.get('sumber')} ===\nJudul: {art.get('judul')}\nIsi: {konten}")
    gabungan = "\n\n".join(bagian)

    prompt = (
        "Berikut beberapa artikel berita Bahasa Indonesia dari sumber media berbeda yang "
        "membahas kejadian/topik yang sama. Bandingkan bagaimana masing-masing sumber "
        "membingkai beritanya: apa yang ditekankan, sudut pandang yang diambil, fakta yang "
        "ditonjolkan atau dilewatkan, dan perbedaan nada penulisan antar sumber. "
        "Jangan menilai mana yang benar/salah atau bias secara politis -- cukup deskripsikan "
        "perbedaan framing-nya secara objektif dalam beberapa paragraf singkat.\n\n"
        f"{gabungan}"
    )
    return _panggil_gemini(prompt, api_key, timeout=45)

# =========================================================
# ANALISIS HASIL: deteksi liputan sama lintas sumber, word cloud, statistik
# =========================================================
def deteksi_klaster_berita(hasil, threshold=0.30):
    """
    Kelompokkan artikel yang kemungkinan membahas kejadian/topik yang sama,
    berdasarkan kemiripan judul (TF-IDF + cosine similarity -> connected
    components via networkx). Threshold 0.30 dikalibrasi manual: pasangan
    judul yang jelas-jelas satu kejadian biasanya skor >=0.33, sementara yang
    cuma "satu rumpun topik besar" ada di kisaran 0.15-0.26.
    Hanya kembalikan klaster dengan >=2 artikel dari sumber BERBEDA.
    """
    if len(hasil) < 2:
        return []

    judul_list = [item.get("judul", "") for item in hasil]
    try:
        vektor = TfidfVectorizer().fit_transform(judul_list)
        sim_matrix = cosine_similarity(vektor)
    except ValueError:
        return []

    graph = nx.Graph()
    graph.add_nodes_from(range(len(hasil)))
    for i in range(len(hasil)):
        for j in range(i + 1, len(hasil)):
            if sim_matrix[i][j] >= threshold:
                graph.add_edge(i, j)

    klaster_valid = []
    for komponen in nx.connected_components(graph):
        if len(komponen) < 2:
            continue
        indeks = sorted(komponen)
        sumber_set = {hasil[i]["sumber"] for i in indeks}
        if len(sumber_set) >= 2:
            klaster_valid.append(indeks)

    klaster_valid.sort(key=len, reverse=True)
    return klaster_valid

_STOPWORDS_TAMBAHAN = {
    "kata", "ujar", "kami", "akan", "juga", "dengan", "untuk", "dari", "yang",
    "ini", "itu", "ada", "ke", "di", "dan", "para", "saat", "tersebut", "dalam"
}

def buat_wordcloud(hasil_tampil):
    teks_gabungan = " ".join(
        f"{item.get('judul', '')} {item.get('ringkasan_redaksi') or item.get('ringkasan') or ''}"
        for item in hasil_tampil
    ).strip()
    if not teks_gabungan:
        return None

    stopwords_id = set(StopWordRemoverFactory().get_stop_words()) | _STOPWORDS_TAMBAHAN
    try:
        wc = WordCloud(
            width=900, height=350, background_color="white",
            stopwords=stopwords_id, collocations=False
        ).generate(teks_gabungan)
        return wc
    except ValueError:
        return None

def buat_dataframe_statistik(hasil_tampil):
    """DataFrame ringan buat chart jumlah artikel per sumber & per hari."""
    baris = []
    for item in hasil_tampil:
        tgl = item.get("tanggal")
        baris.append({
            "sumber": item.get("sumber", "Lainnya"),
            "tanggal": tgl.date() if isinstance(tgl, datetime) else None,
        })
    return pd.DataFrame(baris)

# =========================================================
# SCRAPER PER SUMBER
# =========================================================
def scrape_cnbcindonesia(tema, max_halaman=3):
    """
    Search result CNBC di-load via JS/AJAX, jadi panggil langsung API JSON-nya.
    Bonus: response ini sudah include isi artikel lengkap (strisi),
    jadi gak perlu request tambahan ke halaman detail artikel.
    """
    hasil = []
    limit = 10
    for p in range(max_halaman):
        start = p * limit
        params = {
            "query": tema, "idtype": "1,4", "start": start,
            "limit": limit, "isrelevance": 1
        }
        url = f"https://www.cnbcindonesia.com/api/v2/search-result?{urlencode(params)}"
        raw = ambil_halaman(url)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            break
        if data.get("status") != "success":
            break
        artikel_list = data.get("data", [])
        if not artikel_list:
            break

        for art in artikel_list:
            judul = art.get("strjudul")
            link = art.get("url")
            tanggal_teks = art.get("dtnewsdate")
            tanggal_obj = None
            if tanggal_teks:
                try:
                    tanggal_obj = datetime.strptime(tanggal_teks, "%Y/%m/%d %H:%M:%S")
                except ValueError:
                    tanggal_obj = parse_tanggal(tanggal_teks)

            isi_teks = bersihkan_isi_html(art.get("strisi", ""))

            if judul and link:
                hasil.append({
                    "sumber": "CNBC Indonesia",
                    "judul": judul,
                    "link": link,
                    "ringkasan_redaksi": art.get("strringkasan"),
                    "tanggal_teks": tanggal_teks,
                    "tanggal": tanggal_obj,
                    "konten": isi_teks,
                })

        if len(artikel_list) < limit:
            break

    return hasil

def scrape_cnnindonesia(tema, max_halaman=3, tanggal_mulai=None, tanggal_akhir=None):
    """
    Sama seperti CNBC (satu grup CMS), search result CNN di-load via JS/AJAX.
    Endpoint asli ditemukan dari field 'payload' (base64) di response API:
    /api/v3/search?query=..&idtype=1&start=..&limit=..&fromdate=..&todate=..&isrelevance=1
    Format tanggal API: YYYY/MM/DD (beda dari format URL publik DD/MM/YYYY).
    Bonus: response sudah include isi artikel lengkap (strisi).
    """
    hasil = []
    limit = 10
    for p in range(max_halaman):
        start = p * limit
        params = {
            "query": tema, "idtype": "1", "start": start,
            "limit": limit, "isrelevance": 1
        }
        if tanggal_mulai and tanggal_akhir:
            params["fromdate"] = tanggal_mulai.strftime("%Y/%m/%d")
            params["todate"] = tanggal_akhir.strftime("%Y/%m/%d")

        url = f"https://www.cnnindonesia.com/api/v3/search?{urlencode(params)}"
        raw = ambil_halaman(url)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            break
        if data.get("status") != "success":
            break
        artikel_list = data.get("data", [])
        if not artikel_list:
            break

        for art in artikel_list:
            judul = art.get("strjudul")
            link = art.get("url")
            tanggal_teks = art.get("dtnewsdate")
            tanggal_obj = None
            if tanggal_teks:
                try:
                    tanggal_obj = datetime.strptime(tanggal_teks, "%Y/%m/%d %H:%M:%S")
                except ValueError:
                    tanggal_obj = parse_tanggal(tanggal_teks)

            # strisi sudah HTML lengkap -> bersihkan tabel "Pilihan Redaksi"/"Lihat Juga",
            # iframe, script, lalu ambil teks dari <p>
            isi_teks = bersihkan_isi_html(art.get("strisi", ""))

            if judul and link:
                hasil.append({
                    "sumber": "CNN Indonesia",
                    "judul": judul,
                    "link": link,
                    "ringkasan_redaksi": art.get("strringkasan"),
                    "tanggal_teks": tanggal_teks,
                    "tanggal": tanggal_obj,
                    "konten": isi_teks,
                })

        if len(artikel_list) < limit:
            break

    return hasil

def scrape_detik(tema, max_halaman=3, tanggal_mulai=None, tanggal_akhir=None):
    hasil = []
    fromdatex = tanggal_mulai.strftime("%d/%m/%Y") if tanggal_mulai else None
    todatex = tanggal_akhir.strftime("%d/%m/%Y") if tanggal_akhir else None

    for p in range(1, max_halaman + 1):
        url = f"https://www.detik.com/search/searchall?query={tema}&result_type=latest&page={p}"
        if fromdatex and todatex:
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
                    "ringkasan_redaksi": desc_tag.get_text(strip=True) if desc_tag else None,
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

def scrape_republika(tema, max_halaman=3, tanggal_mulai=None, tanggal_akhir=None):
    """
    Republika server-side rendered, jadi requests+BeautifulSoup biasa cukup.
    Mendukung filter tanggal langsung via parameter URL (startdate/enddate, format YYYY-MM-DD).
    NOTE: parameter pagination ("page") best-effort, belum sempat diverifikasi
    untuk hasil >1 halaman -- tolong dicek manual kalau ada anomali.
    """
    hasil = []
    base_url = "https://republika.co.id/search/v3/all"

    for p in range(1, max_halaman + 1):
        params = {"q": tema, "search_type": "latest"}
        if tanggal_mulai and tanggal_akhir:
            params["latest_date"] = "custom"
            params["startdate"] = tanggal_mulai.strftime("%Y-%m-%d")
            params["enddate"] = tanggal_akhir.strftime("%Y-%m-%d")
        if p > 1:
            params["page"] = p

        url = f"{base_url}?{urlencode(params)}"
        html = ambil_halaman(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        artikel_list = soup.select("div.news-item")
        if not artikel_list:
            break

        for art in artikel_list:
            judul_tag = art.select_one("div.news-title a")
            desc_tag = art.select_one("div.news-description")
            tanggal_tag = art.select_one("div.news-source")

            if not judul_tag:
                continue

            judul = judul_tag.get_text(strip=True)
            link = judul_tag.get("href")
            tanggal_teks = tanggal_tag.get_text(strip=True) if tanggal_tag else None

            if judul and link:
                hasil.append({
                    "sumber": "Republika",
                    "judul": judul,
                    "link": link,
                    "ringkasan_redaksi": desc_tag.get_text(strip=True) if desc_tag else None,
                    "tanggal_teks": tanggal_teks,
                    "tanggal": parse_tanggal(tanggal_teks)
                })
    return hasil

def scrape_liputan6(tema, max_halaman=3, tanggal_mulai=None, tanggal_akhir=None):
    """
    Liputan6 server-side rendered, requests+BeautifulSoup cukup.
    Native date filter via from_date/to_date (format DD/MM/YYYY).
    NOTE: parameter pagination ("page") belum terverifikasi -- satu request
    saja sudah mengembalikan ~20-25 artikel (2 batch ter-render sekaligus di
    HTML, batch ke-2 disembunyikan via CSS class _hidden untuk infinite-scroll,
    tapi datanya tetap ada di HTML statis jadi tetap ke-scrape). Kalau page=N
    ternyata gak didukung, request berikutnya cuma akan dapat artikel duplikat
    yang otomatis ke-filter oleh dedup di cari_berita() -- aman, cuma boros 1 request.
    """
    hasil = []
    base_url = "https://www.liputan6.com/search"

    for p in range(1, max_halaman + 1):
        params = {"q": tema, "type": "all", "order": "latest"}
        if tanggal_mulai and tanggal_akhir:
            params["from_date"] = tanggal_mulai.strftime("%d/%m/%Y")
            params["to_date"] = tanggal_akhir.strftime("%d/%m/%Y")
        if p > 1:
            params["page"] = p

        url = f"{base_url}?{urlencode(params)}"
        html = ambil_halaman(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        artikel_list = soup.select("article.articles--iridescent-list--item")
        if not artikel_list:
            break

        for art in artikel_list:
            judul = art.get("data-title")
            link_tag = art.select_one("a[data-template-var='url']")
            link = link_tag.get("href") if link_tag else None

            time_tag = art.select_one("time[datetime]")
            tanggal_teks = time_tag.get_text(strip=True) if time_tag else None
            tanggal_obj = None
            if time_tag and time_tag.get("datetime"):
                try:
                    tanggal_obj = datetime.fromisoformat(time_tag["datetime"])
                except ValueError:
                    tanggal_obj = parse_tanggal(tanggal_teks)

            summary_tag = art.select_one("[data-template-var='summary']")
            ringkasan_redaksi = summary_tag.get_text(strip=True) if summary_tag else None

            if judul and link:
                hasil.append({
                    "sumber": "Liputan6",
                    "judul": judul,
                    "link": link,
                    "ringkasan_redaksi": ringkasan_redaksi,
                    "tanggal_teks": tanggal_teks,
                    "tanggal": tanggal_obj,
                })

        if len(artikel_list) < 15:
            break

    return hasil

SCRAPER_PER_SUMBER = {
    "CNBC Indonesia": scrape_cnbcindonesia,
    "CNN Indonesia": scrape_cnnindonesia,
    "Detik": scrape_detik,
    "Kompas": scrape_kompas,
    "Republika": scrape_republika,
    "Liputan6": scrape_liputan6,
}
SUMBER_DENGAN_FILTER_TANGGAL_NATIVE = {"Detik", "Republika", "CNN Indonesia", "Liputan6"}

# =========================================================
# AMBIL ISI ARTIKEL (untuk sumber yang gak include konten lengkap di search result)
# =========================================================
SELECTOR_KONTEN = {
    "cnbcindonesia.com": ("div", {"class_": "detail-text"}),  # fallback, biasanya udah ada dari API
    "detik.com": ("div", {"class_": "detail__body-text"}),
    "kompas.com": ("div", {"class_": "read__content"}),
    "republika.co.id": ("div", {"class_": "article-content"}),  # best-effort dari CSS halaman search, tolong dicek
    "liputan6.com": ("div", {"class_": "article-content-body"}),  # ditemukan dari JS brand-safety checker di halaman search
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
            isi = " ".join(p.get_text(" ", strip=True) for p in paragraf)
            isi = re.sub(r"\s+", " ", isi).strip()
            return isi if isi else "Konten kosong"
    return "Selector tidak ditemukan untuk domain ini"

# =========================================================
# FUNGSI UTAMA PENCARIAN
# =========================================================
def cari_berita(tema, sumber_dipilih, tanggal_mulai, tanggal_akhir, max_halaman, ambil_konten, match_persis=False, progress_callback=None):
    semua_hasil = []
    for nama_sumber in sumber_dipilih:
        fungsi = SCRAPER_PER_SUMBER[nama_sumber]
        if progress_callback:
            progress_callback(f"Mengambil dari {nama_sumber}...")
        try:
            if nama_sumber in SUMBER_DENGAN_FILTER_TANGGAL_NATIVE:
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

    # Filter frasa persis -- dilakukan SEBELUM ambil isi lengkap/ringkasan, supaya
    # artikel yang gak relevan langsung dibuang duluan (hemat resource scraping
    # & summarization). Dicek terhadap judul + ringkasan redaksi (kalau ada),
    # karena keduanya sudah tersedia dari hasil listing tanpa request tambahan.
    if match_persis and tema.strip():
        frasa = tema.strip().lower()
        hasil_unik = [
            item for item in hasil_unik
            if frasa in f"{item.get('judul', '')} {item.get('ringkasan_redaksi') or ''}".lower()
        ]

    dt_mulai = datetime.combine(tanggal_mulai, datetime.min.time()) if tanggal_mulai else None
    # PENTING: dt_akhir pakai max.time() (23:59:59), bukan min.time(), supaya artikel
    # yang terbit di "hari akhir" (termasuk hari ini) gak ke-exclude gara-gara dianggap
    # selesai jam 00:00:00. Ini bugfix dari versi sebelumnya.
    dt_akhir = datetime.combine(tanggal_akhir, datetime.max.time()) if tanggal_akhir else None

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
            if not item.get("konten"):
                if progress_callback:
                    progress_callback(f"Mengambil isi [{i}/{total}]: {item['judul'][:60]}...")
                item["konten"] = scrape_isi_artikel(item["link"])

            konten = item.get("konten", "")
            if konten and konten not in KONTEN_INVALID:
                if progress_callback:
                    progress_callback(f"Meringkas [{i}/{total}]: {item['judul'][:60]}...")
                item["ringkasan"] = ringkas_ekstraktif(konten)

    return hasil_final

# =========================================================
# UI STREAMLIT
# =========================================================
st.title("📰 Scraper Berita Indonesia")
st.caption("Cari berita dari CNBC Indonesia, CNN Indonesia, Detik, Kompas, Republika, dan Liputan6 berdasarkan tema dan rentang tanggal — gratis, tanpa API berbayar.")

with st.sidebar:
    st.header("⚙️ Pengaturan Pencarian")
    tema = st.text_input("Tema / kata kunci berita", placeholder="contoh: kaesang")
    match_persis = st.checkbox(
        "Cari frasa persis (exact match)",
        value=False,
        help="Kalau tema lebih dari 1 kata (mis. 'indonesia esport'), hanya tampilkan "
             "artikel yang judul/ringkasannya benar-benar mengandung frasa itu persis, "
             "bukan yang cuma mengandung salah satu katanya saja."
    )
    sumber_dipilih = st.multiselect(
        "Sumber berita",
        options=list(SCRAPER_PER_SUMBER.keys()),
        default=list(SCRAPER_PER_SUMBER.keys())
    )
    pakai_filter_tanggal = st.checkbox("Filter berdasarkan tanggal", value=False)
    if pakai_filter_tanggal:
        col1, col2 = st.columns(2)
        with col1:
            tanggal_mulai = st.date_input("Dari tanggal")
        with col2:
            tanggal_akhir = st.date_input("Sampai tanggal")
    else:
        tanggal_akhir = datetime.now().date()
        tanggal_mulai = tanggal_akhir - timedelta(days=1)
        st.caption(
            f"ℹ️ Tanpa filter manual, pencarian otomatis dibatasi ke "
            f"**{tanggal_mulai.strftime('%d %b %Y')} – {tanggal_akhir.strftime('%d %b %Y')}** "
            f"biar lebih cepat & hemat resource. Centang opsi di atas untuk ubah rentang."
        )
    max_halaman = st.slider("Jumlah halaman per sumber", min_value=1, max_value=10, value=3)
    ambil_konten = st.checkbox("Ambil isi lengkap artikel + ringkasan otomatis (lebih lambat)", value=True)
    tombol_cari = st.button("🔍 Cari Berita", type="primary", use_container_width=True)

    st.divider()
    st.subheader("✨ Ringkasan AI (Fitur Terbatas)")

    if "ai_unlocked" not in st.session_state:
        st.session_state.ai_unlocked = False

    if not GEMINI_API_KEY:
        st.caption("Fitur belum dikonfigurasi developer. Coba lagi nanti.")
    elif st.session_state.ai_unlocked:
        st.success("✅ Fitur ringkasan AI aktif untuk sesi ini.")
    else:
        with st.form("form_password_ai", clear_on_submit=True):
            password_input = st.text_input("Password akses", type="password")
            submit_password = st.form_submit_button("Aktifkan", use_container_width=True)
        if submit_password:
            if AI_SUMMARY_PASSWORD and password_input == AI_SUMMARY_PASSWORD:
                st.session_state.ai_unlocked = True
                st.rerun()
            else:
                st.error(PESAN_HUBUNGI_DEVELOPER)

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
            status_box.info("\n\n".join(progress_log[-5:]))

        with st.spinner("Sedang mengambil data..."):
            hasil = cari_berita(
                tema=tema,
                sumber_dipilih=sumber_dipilih,
                tanggal_mulai=tanggal_mulai,
                tanggal_akhir=tanggal_akhir,
                max_halaman=max_halaman,
                ambil_konten=ambil_konten,
                match_persis=match_persis,
                progress_callback=progress_callback,
            )
        status_box.empty()
        st.session_state.hasil = hasil
        st.session_state.klaster = deteksi_klaster_berita(hasil)
        st.success(f"Selesai! Ditemukan {len(hasil)} artikel unik.")

# =========================================================
# TAMPILKAN HASIL
# =========================================================
hasil = st.session_state.hasil

if hasil:
    st.subheader(f"Hasil: {len(hasil)} artikel")

    sumber_unik = sorted(set(h["sumber"] for h in hasil))
    filter_tampil = st.multiselect("Tampilkan sumber:", sumber_unik, default=sumber_unik)
    hasil_tampil = [h for h in hasil if h["sumber"] in filter_tampil]

    # ---------------- ANALISIS HASIL ----------------
    with st.expander("📊 Analisis Hasil (word cloud & statistik)"):
        wc = buat_wordcloud(hasil_tampil)
        if wc:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption("Belum cukup teks untuk membuat word cloud.")

        df_stat = buat_dataframe_statistik(hasil_tampil)
        if not df_stat.empty:
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Jumlah artikel per sumber")
                st.bar_chart(df_stat["sumber"].value_counts())
            with col_b:
                st.caption("Jumlah artikel per hari")
                per_hari = df_stat.dropna(subset=["tanggal"]).groupby("tanggal").size()
                if not per_hari.empty:
                    st.bar_chart(per_hari)
                else:
                    st.caption("Tanggal tidak terbaca untuk sebagian besar artikel.")

    # ---------------- LIPUTAN SAMA DARI BEBERAPA SUMBER ----------------
    klaster = st.session_state.get("klaster", [])
    klaster_tampil = [
        k for k in klaster
        if any(hasil[i] in hasil_tampil for i in k)
    ]
    if klaster_tampil:
        with st.expander(f"🔗 Liputan Sama dari Beberapa Sumber ({len(klaster_tampil)} topik)"):
            for idx_klaster, indeks_artikel in enumerate(klaster_tampil):
                artikel_klaster = [hasil[i] for i in indeks_artikel]
                sumber_terlibat = ", ".join(sorted({a["sumber"] for a in artikel_klaster}))
                st.markdown(f"**Topik {idx_klaster + 1}** — diliput oleh: {sumber_terlibat}")
                for a in artikel_klaster:
                    st.markdown(f"- [{a['judul']}]({a['link']}) — *{a['sumber']}*")

                if not GEMINI_API_KEY:
                    st.caption("💡 Fitur bandingkan sudut pandang AI belum dikonfigurasi developer.")
                elif not st.session_state.get("ai_unlocked", False):
                    st.caption("🔒 Fitur ini terkunci. Masukkan password di sidebar untuk mengaktifkan.")
                else:
                    cluster_key = "_".join(str(i) for i in indeks_artikel)
                    hasil_banding_key = f"banding_{cluster_key}"
                    if st.session_state.get(hasil_banding_key):
                        st.markdown("**✨ Perbandingan Sudut Pandang (AI):**")
                        st.write(st.session_state[hasil_banding_key])
                    else:
                        if st.button("🔍 Bandingkan Sudut Pandang dengan AI", key=f"btn_banding_{cluster_key}"):
                            with st.spinner("Membandingkan framing antar sumber dengan Gemini..."):
                                hasil_banding, error = bandingkan_sudut_pandang_gemini(artikel_klaster, GEMINI_API_KEY)
                            if hasil_banding:
                                st.session_state[hasil_banding_key] = hasil_banding
                                st.rerun()
                            else:
                                st.error(f"Gagal membandingkan: {error}")
                st.divider()

    for item in hasil_tampil:
        tgl_str = item["tanggal"].strftime("%d %B %Y") if item.get("tanggal") else (item.get("tanggal_teks") or "Tanggal tidak diketahui")
        with st.expander(f"📌 {item['judul']}  —  {item['sumber']} ({tgl_str})"):
            st.markdown(f"**Link:** [{item['link']}]({item['link']})")
            if item.get("catatan"):
                st.warning(item["catatan"])

            if item.get("ringkasan"):
                st.markdown("**📝 Ringkasan Otomatis:**")
                st.write(item["ringkasan"])

            konten_valid = item.get("konten") and item["konten"] not in KONTEN_INVALID

            if item.get("ringkasan_ai"):
                st.markdown("**✨ Ringkasan AI (Gemini):**")
                st.write(item["ringkasan_ai"])
            elif konten_valid:
                if not GEMINI_API_KEY:
                    st.caption("💡 Fitur ringkasan AI belum dikonfigurasi developer.")
                elif not st.session_state.get("ai_unlocked", False):
                    st.caption("🔒 Fitur ringkasan AI terkunci. Masukkan password di sidebar untuk mengaktifkan.")
                else:
                    if st.button("✨ Ringkas dengan AI (Gemini)", key=f"btn_ai_{item['link']}"):
                        with st.spinner("Meringkas dengan Gemini..."):
                            hasil_ai, error = ringkas_gemini(item["konten"], GEMINI_API_KEY)
                        if hasil_ai:
                            item["ringkasan_ai"] = hasil_ai
                            st.rerun()
                        else:
                            st.error(f"Gagal membuat ringkasan AI: {error}")

            if item.get("konten"):
                st.markdown("**📰 Isi Lengkap:**")
                st.write(item["konten"])

    # ---------------- EKSPOR ----------------
    st.subheader("⬇️ Unduh Hasil")

    def buat_csv(data):
        output = io.StringIO()
        kolom = ["sumber", "judul", "link", "tanggal_teks", "tanggal",
                 "ringkasan", "ringkasan_ai", "ringkasan_redaksi", "konten", "catatan"]
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
