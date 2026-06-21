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
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import xml.etree.ElementTree as ET

import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from supabase import create_client

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

@st.cache_resource
def get_supabase_client():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None

SUPABASE = get_supabase_client()

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
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
                   "image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    })
    return session

SESSION = buat_session()

def ambil_halaman(url, timeout=10, progress_callback=None):
    try:
        time.sleep(random.uniform(0.8, 1.6))
        resp = SESSION.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
        if progress_callback:
            progress_callback(f"⚠️ HTTP {resp.status_code}: {url[:90]}")
        return None
    except requests.RequestException as e:
        if progress_callback:
            progress_callback(f"⚠️ Gagal akses ({type(e).__name__}): {url[:90]}")
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

def hitung_top_keywords(hasil_tampil, n=15):
    """Kata yang paling sering muncul (setelah stopword removal) dari gabungan
    judul + ringkasan semua artikel yang tampil. Pelengkap word cloud yang lebih
    presisi/terbaca (angka pasti, bukan cuma ukuran visual)."""
    teks_gabungan = " ".join(
        f"{item.get('judul', '')} {item.get('ringkasan_redaksi') or item.get('ringkasan') or ''}"
        for item in hasil_tampil
    ).lower()
    if not teks_gabungan.strip():
        return []
    stopwords_id = set(StopWordRemoverFactory().get_stop_words()) | _STOPWORDS_TAMBAHAN
    kata_list = re.findall(r"[a-zà-öø-ÿ]{3,}", teks_gabungan)
    kata_list = [k for k in kata_list if k not in stopwords_id]
    return Counter(kata_list).most_common(n)

# =========================================================
# SENTIMENT TAGGING -- LEXICON-BASED SEDERHANA
# Bukan machine learning, cuma hitung selisih kata positif vs negatif dari
# daftar kata yang dikurasi manual. Gratis, instan, tanpa API -- tapi akurasinya
# terbatas (gak paham konteks/sarkasme/negasi). Cocok buat gambaran kasar arah
# pemberitaan, bukan analisis presisi.
# =========================================================
_KATA_POSITIF = {
    "untung", "menguat", "naik", "tumbuh", "sukses", "berhasil", "bagus", "baik",
    "positif", "meningkat", "maju", "unggul", "menang", "juara", "prestasi",
    "apresiasi", "dukungan", "optimis", "stabil", "pulih", "membaik", "lonjak",
    "rekor", "capaian", "inovasi", "terobosan", "kemajuan", "kemenangan",
    "keuntungan", "laba", "surplus", "bangkit", "solid", "kuat", "berkembang",
    "memuaskan", "gemilang", "cemerlang", "harmonis", "damai",
    "sepakat", "rukun", "akur", "lancar", "aman", "kondusif", "efektif",
    "efisien", "produktif", "berkah", "manfaat", "subur", "makmur", "sejahtera",
}
_KATA_NEGATIF = {
    "turun", "melemah", "anjlok", "rugi", "gagal", "buruk", "negatif", "krisis",
    "korupsi", "bencana", "kecelakaan", "kebakaran", "banjir", "longsor",
    "pembunuhan", "kematian", "tewas", "meninggal", "kerusuhan", "konflik",
    "perang", "serangan", "teror", "kekerasan", "kejahatan", "pencurian",
    "penipuan", "pelanggaran", "skandal", "kontroversi", "protes", "demo",
    "tuntutan", "kecaman", "kritik", "ancaman", "bahaya", "darurat", "waspada",
    "khawatir", "cemas", "takut", "marah", "kecewa", "sedih", "duka",
    "tertekan", "terpuruk", "ambruk", "hancur", "rusak", "defisit",
    "pemecatan", "resesi", "utang", "kebangkrutan",
    "tersangka", "terdakwa", "dipenjara", "ditangkap", "diciduk", "disidang",
}

def analisis_sentimen(teks):
    """Return (label, skor). Label salah satu dari 'Positif'/'Negatif'/'Netral'."""
    if not teks:
        return "Netral", 0
    kata_list = re.findall(r"[a-z]+", teks.lower())
    skor_pos = sum(1 for k in kata_list if k in _KATA_POSITIF)
    skor_neg = sum(1 for k in kata_list if k in _KATA_NEGATIF)
    skor = skor_pos - skor_neg
    if skor > 0:
        return "Positif", skor
    elif skor < 0:
        return "Negatif", skor
    return "Netral", skor

def cocok_frasa_persis(teks, frasa):
    """Match frasa persis pakai regex word-boundary, bukan substring polos.
    Lebih presisi: normalisasi spasi berlebih, dan gak ke-trigger kalau frasa
    nempel di tengah kata lain (mis. 'esport' gak nyangkut di 'megaesport')."""
    if not teks or not frasa:
        return False
    teks_norm = re.sub(r"\s+", " ", teks).lower()
    frasa_norm = re.sub(r"\s+", " ", frasa).lower().strip()
    if not frasa_norm:
        return False
    pola = r"\b" + re.escape(frasa_norm) + r"\b"
    return re.search(pola, teks_norm) is not None

# =========================================================
# SCRAPER PER SUMBER
# =========================================================
def scrape_cnbcindonesia(tema, max_halaman=3, tanggal_mulai=None, tanggal_akhir=None, progress_callback=None):
    """
    Search result CNBC di-load via JS/AJAX, jadi panggil langsung API JSON-nya.
    Bonus: response ini sudah include isi artikel lengkap (strisi),
    jadi gak perlu request tambahan ke halaman detail artikel.

    PENTING: fromdate/todate WAJIB dikirim kalau ada filter tanggal -- tanpa ini,
    isrelevance=1 (sort relevansi, bukan terbaru) bisa balikin artikel lama yang
    gak nyangkut sama sekali di rentang tanggal manapun, bikin hasil akhir
    kosong total setelah difilter client-side. Format sama kayak CNN: YYYY/MM/DD
    (CNBC & CNN satu platform/CMS yang sama).
    """
    hasil = []
    limit = 10
    for p in range(max_halaman):
        start = p * limit
        params = {
            "query": tema, "idtype": "1 4", "start": start,
            "limit": limit, "isrelevance": 1
        }
        if tanggal_mulai and tanggal_akhir:
            params["fromdate"] = tanggal_mulai.strftime("%Y/%m/%d")
            params["todate"] = tanggal_akhir.strftime("%Y/%m/%d")
        url = f"https://www.cnbcindonesia.com/api/v2/search-result?{urlencode(params)}"
        raw = ambil_halaman(url, progress_callback=progress_callback)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            if progress_callback:
                progress_callback(f"⚠️ CNBC: respons bukan JSON valid di halaman {p+1}")
            break
        if data.get("status") != "success":
            if progress_callback:
                progress_callback(f"⚠️ CNBC: status respons '{data.get('status')}' (bukan 'success')")
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

def scrape_cnnindonesia(tema, max_halaman=3, tanggal_mulai=None, tanggal_akhir=None, progress_callback=None):
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
        raw = ambil_halaman(url, progress_callback=progress_callback)
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

def scrape_detik(tema, max_halaman=3, tanggal_mulai=None, tanggal_akhir=None, progress_callback=None):
    hasil = []
    fromdatex = tanggal_mulai.strftime("%d/%m/%Y") if tanggal_mulai else None
    todatex = tanggal_akhir.strftime("%d/%m/%Y") if tanggal_akhir else None

    for p in range(1, max_halaman + 1):
        url = f"https://www.detik.com/search/searchall?query={tema}&result_type=latest&page={p}"
        if fromdatex and todatex:
            url += f"&fromdatex={quote(fromdatex, safe='')}&todatex={quote(todatex, safe='')}"

        html = ambil_halaman(url, progress_callback=progress_callback)
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

def scrape_kompas(tema, max_halaman=3, progress_callback=None):
    hasil = []
    for p in range(1, max_halaman + 1):
        url = f"https://search.kompas.com/search?q={tema}&page={p}"
        html = ambil_halaman(url, progress_callback=progress_callback)
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

def scrape_republika(tema, max_halaman=3, tanggal_mulai=None, tanggal_akhir=None, progress_callback=None):
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
        html = ambil_halaman(url, progress_callback=progress_callback)
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

def scrape_liputan6(tema, max_halaman=3, tanggal_mulai=None, tanggal_akhir=None, progress_callback=None):
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
        html = ambil_halaman(url, progress_callback=progress_callback)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        artikel_list = soup.select("article.articles--iridescent-list--item")
        if not artikel_list:
            if progress_callback and p == 1:
                progress_callback(
                    f"⚠️ Liputan6: halaman berhasil diambil ({len(html)} karakter) "
                    f"tapi 0 artikel cocok dengan selector -- kemungkinan struktur "
                    f"HTML berubah atau diblokir/disuguhi halaman beda dari biasanya."
                )
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
                    # .replace(tzinfo=None) WAJIB: fromisoformat() balikin datetime
                    # timezone-aware (ada offset +07:00), sementara dt_mulai/dt_akhir
                    # di cari_berita() naive. Banding aware vs naive -> TypeError,
                    # bikin SELURUH pencarian crash (bukan cuma sumber ini doang).
                    tanggal_obj = datetime.fromisoformat(time_tag["datetime"]).replace(tzinfo=None)
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

# =========================================================
# KUMPARAN -- via Google News Sitemap (bukan search page)
# Kumparan gak punya UI date filter & gak bisa di-search server-side seperti
# 6 sumber lain. Solusinya: baca sitemap berita resmi mereka (terdaftar di
# robots.txt), yang formatnya News Sitemap standar Google -- sudah include
# judul + tanggal presisi + keywords per artikel TANPA perlu fetch halaman
# detail satu-satu. Konsekuensi: sitemap ini cuma nyimpen ~48 jam terakhir
# (standar News Sitemap), jadi cocok buat recent/monitoring, kurang cocok
# buat pencarian historis jauh ke belakang.
# =========================================================
KUMPARAN_SITEMAP_CHANNELS = [
    "https://kumparan.com/sitemap_channel_news.xml",
    "https://kumparan.com/sitemap_channel_bisnis.xml",
    "https://kumparan.com/sitemap_channel_tekno-sains.xml",
    "https://kumparan.com/sitemap_channel_entertainment.xml",
    "https://kumparan.com/sitemap_channel_otomotif.xml",
    "https://kumparan.com/sitemap_channel_woman.xml",
    "https://kumparan.com/sitemap_channel_mom.xml",
    "https://kumparan.com/sitemap_channel_food-travel.xml",
    "https://kumparan.com/sitemap_channel_bola-sports.xml",
    "https://kumparan.com/sitemap_channel_bolanita.xml",
]

_NS_SITEMAP = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}

def _ambil_satu_sitemap_kumparan(url_sitemap, progress_callback=None):
    """Fetch + parse 1 file sitemap channel. Return SEMUA artikel di dalamnya
    (belum difilter keyword -- itu kerjaan scrape_kumparan)."""
    raw = ambil_halaman(url_sitemap, progress_callback=progress_callback)
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        if progress_callback:
            progress_callback(f"⚠️ Kumparan: XML gak valid -- {url_sitemap.split('/')[-1]}")
        return []

    artikel = []
    for url_elem in root.findall("sm:url", _NS_SITEMAP):
        loc_elem = url_elem.find("sm:loc", _NS_SITEMAP)
        news_elem = url_elem.find("news:news", _NS_SITEMAP)
        if loc_elem is None or news_elem is None or loc_elem.text is None:
            continue
        link = loc_elem.text.strip()

        judul_elem = news_elem.find("news:title", _NS_SITEMAP)
        tgl_elem = news_elem.find("news:publication_date", _NS_SITEMAP)
        kw_elem = news_elem.find("news:keywords", _NS_SITEMAP)

        judul = judul_elem.text.strip() if judul_elem is not None and judul_elem.text else ""
        tgl_teks = tgl_elem.text.strip() if tgl_elem is not None and tgl_elem.text else ""
        keywords = kw_elem.text.strip() if kw_elem is not None and kw_elem.text else ""

        tanggal_obj = None
        if tgl_teks:
            try:
                # .replace(tzinfo=None) WAJIB -- konsisten sama seluruh sumber
                # lain yang pakai naive datetime (lihat bugfix Liputan6 di atas).
                tanggal_obj = dateparser.parse(tgl_teks).replace(tzinfo=None)
            except (ValueError, OverflowError, AttributeError):
                tanggal_obj = None

        if judul and link:
            artikel.append({
                "judul": judul, "link": link,
                "tanggal_teks": tgl_teks, "tanggal": tanggal_obj,
                "keywords": keywords,
            })
    return artikel

def scrape_kumparan(tema, max_halaman=3, progress_callback=None):
    """
    Gak ada parameter query di sitemap, jadi filter keyword dilakukan DI SINI
    (bukan server-side) -- cocok kalau tema muncul di judul ATAU keywords.
    Semua channel di-fetch paralel (mirip pola paralel di cari_berita) karena
    ada 10 file sitemap terpisah yang perlu diambil.
    """
    semua_artikel = []
    with ThreadPoolExecutor(max_workers=min(len(KUMPARAN_SITEMAP_CHANNELS), 6)) as executor:
        futures = [
            executor.submit(_ambil_satu_sitemap_kumparan, url, progress_callback)
            for url in KUMPARAN_SITEMAP_CHANNELS
        ]
        for future in as_completed(futures):
            semua_artikel.extend(future.result())

    tema_lower = tema.lower().strip()
    hasil = []
    for art in semua_artikel:
        gabungan = f"{art['judul']} {art['keywords']}".lower()
        if tema_lower in gabungan:
            hasil.append({
                "sumber": "Kumparan",
                "judul": art["judul"],
                "link": art["link"],
                "ringkasan_redaksi": art["keywords"],
                "tanggal_teks": art["tanggal_teks"],
                "tanggal": art["tanggal"],
                "konten": None,
            })
    return hasil

SCRAPER_PER_SUMBER = {
    "CNBC Indonesia": scrape_cnbcindonesia,
    "CNN Indonesia": scrape_cnnindonesia,
    "Detik": scrape_detik,
    "Kompas": scrape_kompas,
    "Republika": scrape_republika,
    "Liputan6": scrape_liputan6,
    "Kumparan": scrape_kumparan,
}
SUMBER_DENGAN_FILTER_TANGGAL_NATIVE = {"Detik", "Republika", "CNN Indonesia", "Liputan6", "CNBC Indonesia"}

# =========================================================
# AMBIL ISI ARTIKEL (untuk sumber yang gak include konten lengkap di search result)
# =========================================================
SELECTOR_KONTEN = {
    "cnbcindonesia.com": ("div", {"class_": "detail-text"}),  # fallback, biasanya udah ada dari API
    "detik.com": ("div", {"class_": "detail__body-text"}),
    "kompas.com": ("div", {"class_": "read__content"}),
    "republika.co.id": ("div", {"class_": "article-content"}),  # best-effort dari CSS halaman search, tolong dicek
    "liputan6.com": ("div", {"class_": "article-content-body"}),  # ditemukan dari JS brand-safety checker di halaman search
    # kumparan.com pakai data-qa-id (bukan class CSS) -- terverifikasi dari page
    # source artikel asli. Lebih stabil drpada class CSS auto-generated
    # styled-components (mis. "c275a64364") yang berubah tiap mereka deploy ulang.
    "kumparan.com": ("main", {"attrs": {"data-qa-id": "article-main"}}),
}

def scrape_isi_artikel(url, progress_callback=None):
    html = ambil_halaman(url, progress_callback=progress_callback)
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
def _scrape_satu_sumber(nama_sumber, tema, max_halaman, tanggal_mulai, tanggal_akhir):
    """
    Helper buat dipanggil paralel oleh ThreadPoolExecutor.
    PENTING: progress_callback di sini LOKAL (cuma append ke list), TIDAK langsung
    menyentuh Streamlit. Streamlit gak thread-safe kalau widget di-update dari
    banyak thread sekaligus -- log dikumpulkan dulu di thread worker, baru
    di-flush ke UI asli dari main thread setelah future-nya selesai (lihat
    cari_berita di bawah).
    """
    fungsi = SCRAPER_PER_SUMBER[nama_sumber]
    local_log = []

    def _local_callback(msg):
        local_log.append(msg)

    try:
        if nama_sumber in SUMBER_DENGAN_FILTER_TANGGAL_NATIVE:
            hasil = fungsi(
                tema, max_halaman=max_halaman,
                tanggal_mulai=tanggal_mulai, tanggal_akhir=tanggal_akhir,
                progress_callback=_local_callback
            )
        else:
            hasil = fungsi(tema, max_halaman=max_halaman, progress_callback=_local_callback)
        return nama_sumber, hasil, local_log, None
    except Exception as e:
        return nama_sumber, [], local_log, str(e)

def cari_berita(tema, sumber_dipilih, tanggal_mulai, tanggal_akhir, max_halaman, ambil_konten, match_persis=False, progress_callback=None):
    semua_hasil = []

    # Scraping paralel per sumber (I/O-bound, jadi thread cocok dan jauh lebih
    # cepat daripada loop sekuensial). Log dari tiap thread di-flush ke UI
    # SETELAH future selesai, dari main thread -- ini yang bikin aman dipakai
    # bareng Streamlit.
    max_workers = max(1, min(len(sumber_dipilih), 6))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_scrape_satu_sumber, nama_sumber, tema, max_halaman, tanggal_mulai, tanggal_akhir): nama_sumber
            for nama_sumber in sumber_dipilih
        }
        for future in as_completed(futures):
            nama_sumber, hasil, local_log, error = future.result()
            semua_hasil.extend(hasil)
            if progress_callback:
                for msg in local_log:
                    progress_callback(f"[{nama_sumber}] {msg}")
                if error:
                    progress_callback(f"[ERROR] {nama_sumber}: {error}")
                else:
                    progress_callback(f"✅ {nama_sumber}: {len(hasil)} artikel ditemukan")

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
        hasil_unik = [
            item for item in hasil_unik
            if cocok_frasa_persis(f"{item.get('judul', '')} {item.get('ringkasan_redaksi') or ''}", tema)
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
                item["konten"] = scrape_isi_artikel(item["link"], progress_callback=progress_callback)

            konten = item.get("konten", "")
            if konten and konten not in KONTEN_INVALID:
                if progress_callback:
                    progress_callback(f"Meringkas [{i}/{total}]: {item['judul'][:60]}...")
                item["ringkasan"] = ringkas_ekstraktif(konten)

    # Sentiment tagging -- selalu jalan (gratis, instan), pakai teks terbaik yang
    # tersedia: isi lengkap kalau ada, kalau enggak fallback ke ringkasan redaksi/judul.
    for item in hasil_final:
        teks_sentimen = item.get("konten") or item.get("ringkasan_redaksi") or item.get("judul") or ""
        if teks_sentimen and teks_sentimen not in KONTEN_INVALID:
            label, skor = analisis_sentimen(teks_sentimen)
            item["sentimen"] = label
            item["sentimen_skor"] = skor

    return hasil_final

@st.cache_data(ttl=900, show_spinner=False)
def cari_berita_cached(tema, sumber_dipilih, tanggal_mulai, tanggal_akhir, max_halaman, ambil_konten, match_persis, _progress_callback=None):
    """
    Wrapper cache (TTL 15 menit) di atas cari_berita. Parameter diawali underscore
    (_progress_callback) dikecualikan Streamlit dari cache key -- soalnya fungsi
    gak bisa di-hash, dan emang gak relevan buat nentuin apakah hasilnya sama.
    Cache hit = gak ada scraping ulang sama sekali, jadi progress_callback gak
    akan kepanggil; itu sudah benar (gak ada progress buat dilaporkan).
    """
    return cari_berita(
        tema, sumber_dipilih, tanggal_mulai, tanggal_akhir,
        max_halaman, ambil_konten, match_persis,
        progress_callback=_progress_callback
    )

# =========================================================
# MONITORING: WATCHLIST & RIWAYAT (Supabase)
# =========================================================
def ambil_watchlist():
    """Ambil semua topik watchlist yang masih aktif (belum di-soft-delete)."""
    if not SUPABASE:
        return []
    try:
        resp = (SUPABASE.table("watchlist")
                .select("*")
                .eq("aktif", True)
                .order("dibuat_pada", desc=True)
                .execute())
        return resp.data or []
    except Exception as e:
        st.error(f"Gagal ambil watchlist: {e}")
        return []

def tambah_watchlist(nama_topik, kata_kunci, sumber_dipilih, exact_match=False):
    if not SUPABASE:
        return None
    try:
        resp = SUPABASE.table("watchlist").insert({
            "nama_topik": nama_topik,
            "kata_kunci": kata_kunci,
            "sumber_dipilih": sumber_dipilih,
            "exact_match": exact_match,
        }).execute()
        return resp.data
    except Exception as e:
        st.error(f"Gagal tambah watchlist: {e}")
        return None

def hapus_watchlist(watchlist_id):
    """Soft delete (set aktif=False) -- bukan DELETE beneran, supaya riwayat
    artikel_history yang sudah terkumpul tetap utuh kalau topiknya diaktifkan lagi."""
    if not SUPABASE:
        return False
    try:
        SUPABASE.table("watchlist").update({"aktif": False}).eq("id", watchlist_id).execute()
        return True
    except Exception as e:
        st.error(f"Gagal hapus watchlist: {e}")
        return False

def simpan_artikel_history(watchlist_id, hasil_artikel):
    """
    Insert artikel baru ke riwayat. Pakai upsert dengan ignore_duplicates=True
    di kombinasi kolom (watchlist_id, link) -- artikel yang linknya sudah pernah
    tersimpan buat watchlist itu otomatis dilewati oleh database sendiri (sesuai
    UNIQUE constraint di skema), jadi gak perlu cek manual satu-satu di sisi app.
    Return jumlah baris yang BENERAN baru tersimpan.
    """
    if not SUPABASE or not hasil_artikel:
        return 0
    baris = []
    for item in hasil_artikel:
        baris.append({
            "watchlist_id": watchlist_id,
            "sumber": item.get("sumber"),
            "judul": item.get("judul"),
            "link": item.get("link"),
            "tanggal_publish": item["tanggal"].isoformat() if item.get("tanggal") else None,
            "ringkasan": item.get("ringkasan") or item.get("ringkasan_redaksi"),
            "sentimen": item.get("sentimen"),
            "sentimen_skor": item.get("sentimen_skor"),
        })
    try:
        resp = SUPABASE.table("artikel_history").upsert(
            baris, on_conflict="watchlist_id,link", ignore_duplicates=True
        ).execute()
        return len(resp.data) if resp.data else 0
    except Exception as e:
        st.error(f"Gagal simpan riwayat: {e}")
        return 0

def ambil_ringkasan_watchlist(watchlist_id):
    """Ringkasan ringan buat ditampilkan di kartu watchlist: total artikel
    tersimpan & kapan checkpoint terakhir dilakukan."""
    if not SUPABASE:
        return {"total": 0, "terakhir": None}
    try:
        resp = (SUPABASE.table("artikel_history")
                .select("waktu_checkpoint", count="exact")
                .eq("watchlist_id", watchlist_id)
                .order("waktu_checkpoint", desc=True)
                .limit(1)
                .execute())
        total = resp.count or 0
        terakhir = resp.data[0]["waktu_checkpoint"] if resp.data else None
        return {"total": total, "terakhir": terakhir}
    except Exception as e:
        return {"total": 0, "terakhir": None}

def ambil_riwayat_watchlist(watchlist_id, limit=200):
    if not SUPABASE:
        return []
    try:
        resp = (SUPABASE.table("artikel_history")
                .select("*")
                .eq("watchlist_id", watchlist_id)
                .order("waktu_checkpoint", desc=True)
                .limit(limit)
                .execute())
        return resp.data or []
    except Exception as e:
        st.error(f"Gagal ambil riwayat: {e}")
        return []

def ambil_riwayat_dataframe(watchlist_id, limit=1000):
    """
    Riwayat watchlist sebagai DataFrame siap-agregasi, dipakai buat chart tren.
    'tanggal_efektif' pakai tanggal_publish kalau ada (lebih akurat -- itu
    kapan beritanya BENERAN terbit), fallback ke waktu_checkpoint kalau
    tanggal_publish gak kebaca dari sumbernya.
    """
    data = ambil_riwayat_watchlist(watchlist_id, limit=limit)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    tgl_publish = pd.to_datetime(df["tanggal_publish"], errors="coerce", utc=True)
    tgl_checkpoint = pd.to_datetime(df["waktu_checkpoint"], errors="coerce", utc=True)
    df["tanggal_efektif"] = tgl_publish.fillna(tgl_checkpoint).dt.date
    return df

# =========================================================
# UI STREAMLIT
# =========================================================
st.title("📰 Scraper Berita Indonesia")
st.caption("Cari berita dari CNBC Indonesia, CNN Indonesia, Detik, Kompas, Republika, dan Liputan6 berdasarkan tema dan rentang tanggal — gratis, tanpa API berbayar.")

with st.sidebar:
    st.header("⚙️ Pengaturan Pencarian")
    tema = st.text_input("Tema / kata kunci berita", placeholder="contoh: kaesang")
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
            f"biar lebih cepat & hemat resource."
        )

    with st.expander("⚙️ Pengaturan Lanjutan"):
        max_halaman = st.slider("Jumlah halaman per sumber", min_value=1, max_value=10, value=3)
        ambil_konten = st.checkbox("Ambil isi lengkap artikel + ringkasan otomatis (lebih lambat)", value=True)
        match_persis = st.checkbox(
            "Cari frasa persis (exact match)",
            value=False,
            help="Kalau tema lebih dari 1 kata (mis. 'indonesia esport'), hanya tampilkan "
                 "artikel yang judul/ringkasannya benar-benar mengandung frasa itu persis, "
                 "bukan yang cuma mengandung salah satu katanya saja."
        )

    tombol_cari = st.button("🔍 Cari Berita", type="primary", use_container_width=True)

    st.divider()
    st.subheader("✨ Fitur Terbatas (AI & Monitoring)")

    if "ai_unlocked" not in st.session_state:
        st.session_state.ai_unlocked = False

    if not AI_SUMMARY_PASSWORD:
        st.caption("Fitur belum dikonfigurasi developer. Coba lagi nanti.")
    elif st.session_state.ai_unlocked:
        st.success("✅ Fitur AI & Monitoring aktif untuk sesi ini.")
    else:
        with st.form("form_password_ai", clear_on_submit=True):
            password_input = st.text_input("Password akses", type="password")
            submit_password = st.form_submit_button("Aktifkan", use_container_width=True)
        if submit_password:
            if password_input == AI_SUMMARY_PASSWORD:
                st.session_state.ai_unlocked = True
                st.rerun()
            else:
                st.error(PESAN_HUBUNGI_DEVELOPER)

tab_cari, tab_monitoring = st.tabs(["🔍 Pencarian Cepat", "📡 Monitoring"])

with tab_cari:
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
                hasil = cari_berita_cached(
                    tema=tema,
                    sumber_dipilih=sumber_dipilih,
                    tanggal_mulai=tanggal_mulai,
                    tanggal_akhir=tanggal_akhir,
                    max_halaman=max_halaman,
                    ambil_konten=ambil_konten,
                    match_persis=match_persis,
                    _progress_callback=progress_callback,
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
        with st.expander("📊 Analisis Hasil (word cloud, kata kunci & sentimen)"):
            wc = buat_wordcloud(hasil_tampil)
            if wc:
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.caption("Belum cukup teks untuk membuat word cloud.")

            col_kw, col_sent = st.columns(2)
            with col_kw:
                st.caption("Top 15 kata kunci")
                top_kw = hitung_top_keywords(hasil_tampil)
                if top_kw:
                    df_kw = pd.DataFrame(top_kw, columns=["kata", "jumlah"]).set_index("kata")
                    st.bar_chart(df_kw)
                else:
                    st.caption("Belum cukup teks.")
            with col_sent:
                st.caption("Distribusi sentimen (lexicon-based, kasar)")
                sentimen_list = [item.get("sentimen") for item in hasil_tampil if item.get("sentimen")]
                if sentimen_list:
                    sentimen_counts = pd.Series(sentimen_list).value_counts()
                    st.bar_chart(sentimen_counts)
                else:
                    st.caption("Belum ada data sentimen (perlu ambil isi/ringkasan dulu).")

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

                if item.get("sentimen"):
                    emoji_sentimen = {"Positif": "🟢", "Negatif": "🔴", "Netral": "⚪"}.get(item["sentimen"], "⚪")
                    st.caption(f"{emoji_sentimen} Sentimen: {item['sentimen']} (perkiraan kasar, lexicon-based)")

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
                     "ringkasan", "ringkasan_ai", "ringkasan_redaksi", "sentimen", "konten", "catatan"]
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

with tab_monitoring:
    st.subheader("📡 Monitoring Watchlist")
    st.caption(
        "Track topik secara berkelanjutan. Beda dari Pencarian Cepat: hasil di sini "
        "TERSIMPAN permanen (Supabase) dan menumpuk jadi riwayat tiap kali Anda klik Refresh, "
        "jadi bisa dipakai buat lihat tren dari waktu ke waktu."
    )

    if not st.session_state.get("ai_unlocked", False):
        st.warning(
            "🔒 Fitur Monitoring terkunci. Masukkan password di sidebar "
            "(\"✨ Fitur Terbatas\") untuk mengaktifkan."
        )
    else:
        if not SUPABASE:
            st.error(
                "Supabase belum terkoneksi. Pastikan `SUPABASE_URL` dan `SUPABASE_KEY` "
                "sudah di-set di Streamlit secrets."
            )
        else:
            with st.expander("➕ Tambah Topik Baru"):
                with st.form("form_tambah_watchlist", clear_on_submit=True):
                    nama_topik_baru = st.text_input("Nama topik", placeholder="contoh: Rupiah, Kaesang Pangarep")
                    kata_kunci_baru = st.text_input("Kata kunci pencarian", placeholder="contoh: rupiah")
                    sumber_baru = st.multiselect(
                        "Sumber yang dipantau",
                        options=list(SCRAPER_PER_SUMBER.keys()),
                        default=list(SCRAPER_PER_SUMBER.keys())
                    )
                    exact_baru = st.checkbox("Cari frasa persis (exact match)", value=False, key="exact_match_watchlist")
                    submit_watchlist = st.form_submit_button("Tambah ke Watchlist", type="primary", use_container_width=True)

                if submit_watchlist:
                    if not nama_topik_baru.strip() or not kata_kunci_baru.strip():
                        st.warning("Nama topik dan kata kunci wajib diisi.")
                    elif not sumber_baru:
                        st.warning("Pilih minimal 1 sumber.")
                    else:
                        hasil_tambah = tambah_watchlist(
                            nama_topik_baru.strip(), kata_kunci_baru.strip(), sumber_baru, exact_baru
                        )
                        if hasil_tambah:
                            st.success(f"Topik '{nama_topik_baru}' ditambahkan ke watchlist.")
                            st.rerun()

            st.divider()

            daftar_watchlist = ambil_watchlist()

            if not daftar_watchlist:
                st.info("Belum ada topik yang di-track. Tambahkan lewat form di atas.")
            else:
                total_artikel_semua = sum(ambil_ringkasan_watchlist(wl["id"])["total"] for wl in daftar_watchlist)
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Topik Dipantau", len(daftar_watchlist))
                col_m2.metric("Total Artikel Tersimpan", total_artikel_semua)
                st.divider()

                for wl in daftar_watchlist:
                    with st.container(border=True):
                        col_info, col_refresh, col_hapus = st.columns([3, 1, 1])
                        with col_info:
                            st.markdown(f"**{wl['nama_topik']}**")
                            st.caption(
                                f"Keyword: `{wl['kata_kunci']}` · "
                                f"Sumber: {', '.join(wl['sumber_dipilih'])}"
                                + (" · exact match" if wl.get("exact_match") else "")
                            )
                            ringkasan_wl = ambil_ringkasan_watchlist(wl["id"])
                            if ringkasan_wl["total"] > 0:
                                terakhir_str = ringkasan_wl["terakhir"][:16].replace("T", " ") if ringkasan_wl["terakhir"] else "-"
                                st.caption(f"📚 {ringkasan_wl['total']} artikel tersimpan · checkpoint terakhir: {terakhir_str}")
                            else:
                                st.caption("📚 Belum ada riwayat -- klik Refresh untuk checkpoint pertama.")
                        with col_refresh:
                            if st.button("🔄 Refresh", key=f"refresh_wl_{wl['id']}", use_container_width=True):
                                with st.spinner(f"Refresh '{wl['nama_topik']}'..."):
                                    tgl_akhir_wl = datetime.now().date()
                                    tgl_mulai_wl = tgl_akhir_wl - timedelta(days=1)
                                    hasil_wl = cari_berita_cached(
                                        tema=wl["kata_kunci"],
                                        sumber_dipilih=wl["sumber_dipilih"],
                                        tanggal_mulai=tgl_mulai_wl,
                                        tanggal_akhir=tgl_akhir_wl,
                                        max_halaman=3,
                                        ambil_konten=True,
                                        match_persis=wl.get("exact_match", False),
                                        _progress_callback=None,
                                    )
                                    jumlah_baru = simpan_artikel_history(wl["id"], hasil_wl)
                                st.success(f"{jumlah_baru} artikel baru disimpan (dari {len(hasil_wl)} ditemukan).")
                                st.rerun()
                        with col_hapus:
                            if st.button("🗑️ Hapus", key=f"hapus_wl_{wl['id']}", use_container_width=True):
                                hapus_watchlist(wl["id"])
                                st.rerun()

                        with st.expander("📊 Analitik & Tren"):
                            df_riwayat = ambil_riwayat_dataframe(wl["id"])
                            if df_riwayat.empty:
                                st.caption("Belum cukup data buat dianalisis -- refresh dulu beberapa kali.")
                            else:
                                col_vol, col_sov = st.columns(2)
                                with col_vol:
                                    st.caption("Volume artikel per hari")
                                    volume_harian = df_riwayat.groupby("tanggal_efektif").size()
                                    st.bar_chart(volume_harian)
                                with col_sov:
                                    st.caption("Share of voice (per sumber)")
                                    sov = df_riwayat["sumber"].value_counts()
                                    st.bar_chart(sov)

                                if "sentimen" in df_riwayat.columns and df_riwayat["sentimen"].notna().any():
                                    st.caption("Tren sentimen per hari")
                                    sentimen_harian = (
                                        df_riwayat.dropna(subset=["sentimen"])
                                        .groupby(["tanggal_efektif", "sentimen"])
                                        .size()
                                        .unstack(fill_value=0)
                                    )
                                    st.bar_chart(sentimen_harian)

                        with st.expander("Lihat riwayat artikel"):
                            riwayat_wl = ambil_riwayat_watchlist(wl["id"], limit=50)
                            if riwayat_wl:
                                for art in riwayat_wl:
                                    emoji_sent = {"Positif": "🟢", "Negatif": "🔴", "Netral": "⚪"}.get(art.get("sentimen"), "⚪")
                                    st.markdown(f"{emoji_sent} [{art['judul']}]({art['link']}) — *{art['sumber']}*")
                            else:
                                st.caption("Belum ada riwayat tersimpan.")
