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
import plotly.express as px
import plotly.graph_objects as go
import math
import calendar
from supabase import create_client

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

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

GEMINI_MODEL = "gemini-2.5-flash"
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
def _panggil_gemini(prompt, api_key, timeout=30, max_retry=2):
    """Helper generik buat panggil Gemini API. Retry otomatis max 2x (jeda 3 detik)
    untuk error 503 (server overloaded). Dipakai oleh ringkas_gemini,
    bandingkan_sudut_pandang_gemini, dan fungsi PDF report."""
    if not api_key:
        return None, "API key Gemini belum di-set di Streamlit secrets."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(max_retry + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.Timeout:
            return None, "Request ke Gemini timeout, coba lagi."
        except requests.exceptions.RequestException as e:
            return None, f"Gagal menghubungi Gemini: {e}"

        if resp.status_code == 503:
            if attempt < max_retry:
                time.sleep(3)
                continue
            return None, "Gemini sedang sibuk (503). Coba generate ulang dalam beberapa saat."
        if resp.status_code == 429:
            return None, "Rate limit Gemini tercapai, coba lagi beberapa saat lagi."
        if resp.status_code != 200:
            return None, f"Gemini API error (status {resp.status_code})."
        break  # sukses, keluar dari retry loop

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
# CHART BUILDER -- Plotly (migrasi dari st.bar_chart/Altair)
# PENTING: axis tanggal SELALU dipaksa kategorikal (type="category") pakai
# label string ("21 Jun"), BUKAN datetime kontinu. Altair/st.bar_chart auto
# nampilin tick per-jam kalau rentang tanggalnya pendek -- ini fix-nya.
# Urutan kategori tanggal dihitung dari tanggal asli (kronologis), BUKAN
# alfabetis stringnya (soalnya "01 Jul" < "30 Jun" secara alfabet, salah).
# =========================================================
_WARNA_SENTIMEN = {"Positif": "#1D9E75", "Negatif": "#D85A30", "Netral": "#888780"}

def _layout_chart_ringkas(fig, tinggi=280):
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=tinggi)
    return fig

def buat_chart_per_sumber(df_stat):
    if df_stat.empty:
        return None
    counts = df_stat["sumber"].value_counts().reset_index()
    counts.columns = ["sumber", "jumlah"]
    fig = px.bar(counts, x="sumber", y="jumlah")
    fig.update_xaxes(type="category", title=None)
    fig.update_yaxes(title=None)
    fig.update_traces(marker_color="#378ADD")
    return _layout_chart_ringkas(fig)

def buat_chart_per_hari(df_stat):
    df = df_stat.dropna(subset=["tanggal"]).copy()
    if df.empty:
        return None
    df["tanggal_str"] = df["tanggal"].apply(lambda d: d.strftime("%d %b"))
    urutan = df.groupby("tanggal_str")["tanggal"].min().sort_values().index.tolist()
    counts = df.groupby("tanggal_str").size().reset_index(name="jumlah")
    fig = px.bar(counts, x="tanggal_str", y="jumlah", category_orders={"tanggal_str": urutan})
    fig.update_xaxes(type="category", title=None)
    fig.update_yaxes(title=None)
    fig.update_traces(marker_color="#378ADD")
    return _layout_chart_ringkas(fig)

def buat_chart_volume_sentimen_harian(hasil_tampil):
    """Stacked bar: volume artikel per hari, breakdown sentimen sebagai series."""
    baris = []
    for item in hasil_tampil:
        tgl = item.get("tanggal")
        if isinstance(tgl, datetime) and item.get("sentimen"):
            baris.append({"tanggal": tgl.date(), "sentimen": item["sentimen"]})
    if not baris:
        return None
    df = pd.DataFrame(baris)
    df["tanggal_str"] = df["tanggal"].apply(lambda d: d.strftime("%d %b"))
    urutan = df.groupby("tanggal_str")["tanggal"].min().sort_values().index.tolist()
    agg = df.groupby(["tanggal_str", "sentimen"]).size().reset_index(name="jumlah")
    fig = px.bar(
        agg, x="tanggal_str", y="jumlah", color="sentimen",
        category_orders={"tanggal_str": urutan, "sentimen": ["Positif", "Netral", "Negatif"]},
        color_discrete_map=_WARNA_SENTIMEN,
    )
    fig.update_xaxes(type="category", title=None)
    fig.update_yaxes(title=None)
    fig.update_layout(legend_title=None)
    return _layout_chart_ringkas(fig, tinggi=300)

def buat_chart_top_keywords(top_kw):
    if not top_kw:
        return None
    df_kw = pd.DataFrame(top_kw, columns=["kata", "jumlah"])
    fig = px.bar(df_kw, x="jumlah", y="kata", orientation="h")
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=None)
    fig.update_traces(marker_color="#378ADD")
    return _layout_chart_ringkas(fig, tinggi=340)

def buat_chart_sentimen_distribusi(hasil_tampil):
    sentimen_list = [item.get("sentimen") for item in hasil_tampil if item.get("sentimen")]
    if not sentimen_list:
        return None
    counts = pd.Series(sentimen_list).value_counts().reset_index()
    counts.columns = ["sentimen", "jumlah"]
    fig = px.bar(
        counts, x="sentimen", y="jumlah", color="sentimen",
        category_orders={"sentimen": ["Positif", "Netral", "Negatif"]},
        color_discrete_map=_WARNA_SENTIMEN,
    )
    fig.update_xaxes(type="category", title=None)
    fig.update_yaxes(title=None)
    fig.update_layout(showlegend=False)
    return _layout_chart_ringkas(fig)

# =========================================================
# FASE 2 — METRIK PR & MEDIA INTELLIGENCE
# =========================================================
def hitung_nss(hasil_tampil):
    """
    Net Sentiment Score: (positif - negatif) / total × 100. Range -100 s/d +100.
    Positif murni = +100, negatif murni = -100, seimbang = 0.
    """
    sentimen_list = [item.get("sentimen") for item in hasil_tampil if item.get("sentimen")]
    if not sentimen_list:
        return None
    positif = sentimen_list.count("Positif")
    negatif = sentimen_list.count("Negatif")
    total = len(sentimen_list)
    return round((positif - negatif) / total * 100, 1)

def hitung_source_diversity(hasil_tampil):
    """
    Normalized Shannon entropy (0-100). 0 = semua dari 1 sumber,
    100 = distribusi sempurna merata ke semua sumber aktif.
    """
    if not hasil_tampil:
        return None
    sumber_counter = Counter(item["sumber"] for item in hasil_tampil)
    n = len(sumber_counter)
    if n <= 1:
        return 0.0
    total = len(hasil_tampil)
    entropy = -sum((c / total) * math.log(c / total) for c in sumber_counter.values())
    return round(entropy / math.log(n) * 100, 1)

def hitung_prominence(hasil_tampil, tema):
    """
    Prominence Score: % artikel yang minimal 1 kata dari tema muncul di judul.
    Pakai cek per-kata (bukan frasa penuh) supaya query multi-kata seperti
    'jokowi psi' tetap ke-deteksi walau kedua kata tidak nempel berurutan di judul.
    Kata pendek (<3 huruf) diabaikan supaya tidak false-positive.
    """
    if not hasil_tampil or not tema:
        return None
    kata_tema = [k for k in tema.lower().strip().split() if len(k) >= 3]
    if not kata_tema:
        return None
    total = len(hasil_tampil)
    muncul_di_judul = sum(
        1 for item in hasil_tampil
        if any(k in (item.get("judul") or "").lower() for k in kata_tema)
    )
    return round(muncul_di_judul / total * 100, 1) if total else None

def hitung_konsensus_kontroversi(hasil_tampil):
    """
    Indikator konsensus vs kontroversial berdasarkan distribusi sentimen.
    Konsensus: 1 sentimen dominan >= 70%.
    Kontroversial: positif vs negatif bersaing ketat (selisih < 20%, keduanya >= 15%).
    Campuran: di antara keduanya.
    Return dict {label, emoji, detail} atau None kalau datanya kurang.
    """
    sentimen_list = [item.get("sentimen") for item in hasil_tampil if item.get("sentimen")]
    if len(sentimen_list) < 3:
        return None
    total = len(sentimen_list)
    pos = sentimen_list.count("Positif")
    neg = sentimen_list.count("Negatif")
    net = sentimen_list.count("Netral")
    pct_pos = pos / total * 100
    pct_neg = neg / total * 100
    pct_net = net / total * 100
    dominan_pct = max(pct_pos, pct_neg, pct_net)

    if dominan_pct >= 70:
        label_dom = "Positif" if pct_pos == dominan_pct else ("Negatif" if pct_neg == dominan_pct else "Netral")
        return {
            "label": "Konsensus",
            "emoji": "🤝",
            "detail": f"Pemberitaan didominasi sentimen **{label_dom}** ({dominan_pct:.0f}% dari {total} artikel)",
        }
    elif abs(pct_pos - pct_neg) <= 20 and min(pct_pos, pct_neg) >= 15:
        return {
            "label": "Kontroversial",
            "emoji": "⚡",
            "detail": f"Pemberitaan terpolarisasi — Positif {pct_pos:.0f}% vs Negatif {pct_neg:.0f}% dari {total} artikel",
        }
    else:
        return {
            "label": "Campuran",
            "emoji": "🔀",
            "detail": f"Sentimen beragam tanpa dominasi kuat (Positif {pct_pos:.0f}%, Netral {pct_net:.0f}%, Negatif {pct_neg:.0f}%)",
        }

# =========================================================
# FASE 3 — CHART INTERAKTIF DASHBOARD ANALITIK
# =========================================================
def buat_chart_nss_trend(hasil_tampil):
    """
    Line chart NSS per hari (Plotly interaktif, zoom via rangeslider).
    Tiap titik = NSS hari itu, warna titik ikut polaritas.
    Garis putus-putus di y=0 sebagai baseline netral.
    """
    baris = []
    for item in hasil_tampil:
        tgl = item.get("tanggal")
        if isinstance(tgl, datetime) and item.get("sentimen"):
            baris.append({"tanggal": tgl.date(), "sentimen": item["sentimen"]})
    if len(baris) < 2:
        return None

    df = pd.DataFrame(baris)

    def _nss_per_hari(grp):
        tot = len(grp)
        p = (grp["sentimen"] == "Positif").sum()
        n = (grp["sentimen"] == "Negatif").sum()
        return round((p - n) / tot * 100, 1)

    nss_harian = df.groupby("tanggal").apply(_nss_per_hari).reset_index()
    nss_harian.columns = ["tanggal", "nss"]
    nss_harian = nss_harian.sort_values("tanggal")
    nss_harian["tanggal_str"] = nss_harian["tanggal"].apply(lambda d: d.strftime("%d %b"))
    nss_harian["warna"] = nss_harian["nss"].apply(
        lambda v: "#1D9E75" if v > 0 else ("#D85A30" if v < 0 else "#888780")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nss_harian["tanggal_str"],
        y=nss_harian["nss"],
        mode="lines+markers",
        marker=dict(color=nss_harian["warna"].tolist(), size=9),
        line=dict(color="#378ADD", width=2),
        hovertemplate="<b>%{x}</b><br>NSS: %{y}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#888780", line_width=1)
    fig.update_layout(
        xaxis=dict(type="category", title=None),
        yaxis=dict(title="NSS", range=[-105, 105]),
        margin=dict(l=0, r=0, t=10, b=0),
        height=260,
    )
    return fig

def buat_chart_bias_sumber(hasil_tampil):
    """
    Stacked bar 100% — profil sentimen per sumber.
    Sumber diurutkan berdasarkan % Positif tertinggi ke terendah,
    memudahkan identifikasi sumber yang sistematis lebih positif/negatif.
    """
    baris = []
    for item in hasil_tampil:
        if item.get("sentimen") and item.get("sumber"):
            baris.append({"sumber": item["sumber"], "sentimen": item["sentimen"]})
    if not baris:
        return None
    df = pd.DataFrame(baris)
    agg = df.groupby(["sumber", "sentimen"]).size().reset_index(name="jumlah")
    total_per_sumber = df.groupby("sumber").size().reset_index(name="total")
    agg = agg.merge(total_per_sumber, on="sumber")
    agg["persen"] = round(agg["jumlah"] / agg["total"] * 100, 1)

    # Urutkan dari % Positif tertinggi
    pct_pos_per_sumber = (
        agg[agg["sentimen"] == "Positif"]
        .set_index("sumber")["persen"]
        .to_dict()
    )
    urutan_sumber = sorted(
        agg["sumber"].unique(),
        key=lambda s: pct_pos_per_sumber.get(s, 0),
        reverse=True,
    )

    fig = px.bar(
        agg, x="sumber", y="persen", color="sentimen",
        category_orders={
            "sumber": urutan_sumber,
            "sentimen": ["Positif", "Netral", "Negatif"],
        },
        color_discrete_map=_WARNA_SENTIMEN,
        barmode="stack",
    )
    fig.update_xaxes(type="category", title=None)
    fig.update_yaxes(title="%", range=[0, 105])
    fig.update_layout(
        legend_title=None,
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
    )
    return fig

# =========================================================
# FASE 5 — PDF REPORT GENERATION
# Arsitektur: Hierarchical Summarization
#   Layer 1: ringkas_ekstraktif() per artikel (sudah ada, gratis)
#   Layer 2: Gemini per cluster/kelompok sumber (~5-15 artikel per call)
#   Layer 3: Gemini meta-synthesis: gabung semua layer 2 + metrik → executive summary
# =========================================================

PDF_MAX_ARTIKEL = 30  # Batas auto-select artikel untuk report

def auto_pilih_artikel(hasil_tampil, max_artikel=PDF_MAX_ARTIKEL):
    """
    Auto-select top-N artikel berdasarkan skor relevansi komposit:
    - Punya konten lengkap (+3)
    - Tema muncul di judul (+2)
    - Sentimen ekstrem (positif/negatif, bukan netral) (+1)
    - Sumber unik bonus: prioritaskan diversitas sumber
    Return list index artikel yang dipilih (index di hasil_tampil).
    """
    if len(hasil_tampil) <= max_artikel:
        return list(range(len(hasil_tampil)))

    skor = []
    for i, item in enumerate(hasil_tampil):
        s = 0
        konten = item.get("konten", "")
        if konten and konten not in KONTEN_INVALID and len(konten) > 100:
            s += 3
        judul = (item.get("judul") or "").lower()
        # tema tidak tersedia di sini, pakai heuristic: judul panjang = lebih informatif
        if len(judul) > 30:
            s += 1
        if item.get("sentimen") in ("Positif", "Negatif"):
            s += 1
        skor.append((s, i))

    # Sort by skor desc, tapi pastikan distribusi sumber merata
    skor.sort(key=lambda x: x[0], reverse=True)
    dipilih = []
    sumber_count = {}
    # Ambil max 5 per sumber supaya tidak dominan satu sumber
    for _, idx in skor:
        sumber = hasil_tampil[idx].get("sumber", "")
        if sumber_count.get(sumber, 0) < 5:
            dipilih.append(idx)
            sumber_count[sumber] = sumber_count.get(sumber, 0) + 1
        if len(dipilih) >= max_artikel:
            break

    return sorted(dipilih)


def _buat_prompt_cluster(artikel_list, bahasa="Indonesia"):
    """Buat prompt Layer 2: analisis narasi per cluster/sumber."""
    lang_instruction = (
        "Respond in formal Bahasa Indonesia." if bahasa == "Indonesia"
        else "Respond in formal English."
    )
    bagian = []
    for i, art in enumerate(artikel_list, 1):
        ringkasan = (
            art.get("ringkasan")
            or art.get("ringkasan_redaksi")
            or art.get("judul", "")
        )[:600]
        bagian.append(
            f"Artikel {i} [{art.get('sumber','')}]: {art.get('judul','')}\n"
            f"Ringkasan: {ringkasan}"
        )
    konten_gabung = "\n\n".join(bagian)

    return (
        f"Kamu adalah analis media PR profesional. {lang_instruction}\n"
        f"Berikut {len(artikel_list)} artikel berita tentang topik yang sama.\n\n"
        f"{konten_gabung}\n\n"
        f"Analisis dalam format JSON (tanpa markdown fence, hanya JSON murni):\n"
        f'{{"narrative": "narasi/framing dominan 1 paragraf", '
        f'"key_facts": ["fakta 1", "fakta 2", "fakta 3"], '
        f'"tone": "positif/negatif/netral/campuran", '
        f'"notable_angle": "angle unik atau berbeda jika ada, atau null"}}\n'
        f"Maksimal 200 kata total. Bahasa formal."
    )


def _buat_prompt_executive(tema, tgl_mulai, tgl_akhir, metrik, cluster_summaries, bahasa="Indonesia"):
    """Buat prompt Layer 3: executive summary dari hasil cluster summaries."""
    lang_instruction = (
        "Tulis dalam Bahasa Indonesia formal." if bahasa == "Indonesia"
        else "Write in formal English."
    )
    metrik_str = (
        f"- Total artikel: {metrik['total']}\n"
        f"- Net Sentiment Score (NSS): {metrik['nss']}\n"
        f"- Share of Voice: {metrik['sov']}\n"
        f"- Prominence Score: {metrik['prominence']}%\n"
        f"- Indikator: {metrik['indikator']}"
    )
    cluster_str = "\n\n".join(
        f"Kelompok {i+1}: {s}" for i, s in enumerate(cluster_summaries)
    )
    return (
        f"Kamu adalah konsultan PR senior. {lang_instruction}\n\n"
        f"TOPIK: {tema}\n"
        f"PERIODE: {tgl_mulai} s/d {tgl_akhir}\n\n"
        f"METRIK:\n{metrik_str}\n\n"
        f"RINGKASAN PER KELOMPOK PEMBERITAAN:\n{cluster_str}\n\n"
        f"Tulis Executive Summary dalam 4 paragraf:\n"
        f"1. Gambaran umum lanskap pemberitaan\n"
        f"2. Tren sentimen dan apa yang mendorongnya\n"
        f"3. Dinamika antar media (siapa paling aktif, perbedaan framing)\n"
        f"4. Implikasi dan hal yang perlu diwaspadai\n\n"
        f"Tidak lebih dari 350 kata. Jangan gunakan bullet point."
    )


def generate_cluster_summaries(artikel_dipilih, api_key, bahasa="Indonesia", progress_callback=None):
    """
    Layer 2: Bagi artikel ke cluster per sumber, panggil Gemini per cluster.
    Return list of (cluster_label, summary_text). Cluster yang gagal di-skip.
    """
    # Kelompokkan per sumber
    per_sumber = {}
    for item in artikel_dipilih:
        sumber = item.get("sumber", "Lainnya")
        per_sumber.setdefault(sumber, []).append(item)

    summaries = []
    total = len(per_sumber)
    for idx, (sumber, arts) in enumerate(per_sumber.items(), 1):
        if progress_callback:
            progress_callback(f"Menganalisis liputan {sumber} ({idx}/{total})...")
        prompt = _buat_prompt_cluster(arts, bahasa)
        hasil, error = _panggil_gemini(prompt, api_key, timeout=45)
        if not hasil:
            summaries.append((sumber, f"[Analisis tidak tersedia: {error}]"))
            continue
        # Parse JSON — kalau gagal, pakai raw text
        try:
            clean = hasil.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            ringkas = (
                f"Narasi: {data.get('narrative','')}\n"
                f"Fakta utama: {'; '.join(data.get('key_facts',[]))}\n"
                f"Nada: {data.get('tone','')}\n"
                f"Angle khusus: {data.get('notable_angle') or '-'}"
            )
            summaries.append((sumber, ringkas))
        except (json.JSONDecodeError, AttributeError):
            summaries.append((sumber, hasil[:500]))

    return summaries


def generate_executive_summary(tema, tgl_mulai, tgl_akhir, metrik, cluster_summaries, api_key, bahasa="Indonesia"):
    """Layer 3: Executive summary dari hasil cluster summaries + metrik."""
    cluster_texts = [f"{label}:\n{text}" for label, text in cluster_summaries]
    prompt = _buat_prompt_executive(tema, tgl_mulai, tgl_akhir, metrik, cluster_texts, bahasa)
    return _panggil_gemini(prompt, api_key, timeout=60)


def _export_chart_png(fig, lebar=500, tinggi=220):
    """Export Plotly figure ke bytes PNG untuk di-embed di PDF."""
    try:
        return fig.to_image(format="png", width=lebar, height=tinggi, scale=2)
    except Exception:
        return None


def buat_pdf_report(
    tema, tgl_mulai, tgl_akhir,
    hasil_tampil, artikel_dipilih,
    executive_summary, cluster_summaries,
    metrik, bahasa="Indonesia"
):
    """
    Generate PDF bytes menggunakan ReportLab.
    Struktur:
      Hal 1: Cover
      Hal 2: Executive Summary
      Hal 3: Key Metrics + Charts
      Hal 4+: Cluster Analysis (per sumber)
      Hal terakhir: Article Index (tabel clickable) + Metodologi
    """
    buffer = io.BytesIO()
    PAGE_W, PAGE_H = A4
    MARGIN = 2 * cm

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"Media Intelligence Report — {tema}",
    )

    # ── Styles ────────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    st_cover_title = ParagraphStyle(
        "CoverTitle", parent=styles["Title"],
        fontSize=26, leading=32, spaceAfter=8, alignment=TA_CENTER,
        textColor=colors.HexColor("#1A2E4A"),
    )
    st_cover_sub = ParagraphStyle(
        "CoverSub", parent=styles["Normal"],
        fontSize=12, leading=16, alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
    )
    st_cover_meta = ParagraphStyle(
        "CoverMeta", parent=styles["Normal"],
        fontSize=10, leading=14, alignment=TA_CENTER,
        textColor=colors.HexColor("#888888"),
    )
    st_h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=16, leading=20, spaceBefore=12, spaceAfter=6,
        textColor=colors.HexColor("#1A2E4A"),
    )
    st_h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=12, leading=16, spaceBefore=8, spaceAfter=4,
        textColor=colors.HexColor("#378ADD"),
    )
    st_body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=6,
        textColor=colors.HexColor("#333333"),
    )
    st_caption = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=8, leading=11, spaceAfter=4,
        textColor=colors.HexColor("#777777"), alignment=TA_CENTER,
    )
    st_label = ParagraphStyle(
        "Label", parent=styles["Normal"],
        fontSize=9, leading=12,
        textColor=colors.HexColor("#555555"),
    )
    st_metric_val = ParagraphStyle(
        "MetricVal", parent=styles["Normal"],
        fontSize=20, leading=24, alignment=TA_CENTER,
        textColor=colors.HexColor("#1A2E4A"),
    )
    st_metric_lbl = ParagraphStyle(
        "MetricLbl", parent=styles["Normal"],
        fontSize=8, leading=10, alignment=TA_CENTER,
        textColor=colors.HexColor("#888888"),
    )
    st_link = ParagraphStyle(
        "Link", parent=styles["Normal"],
        fontSize=8, leading=11,
        textColor=colors.HexColor("#378ADD"),
    )

    story = []

    # ── HALAMAN 1: COVER ──────────────────────────────────────────────────────
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("📰 Media Intelligence Report", st_cover_title))
    story.append(Spacer(1, 0.5 * cm))

    story.append(HRFlowable(
        width="80%", thickness=2,
        color=colors.HexColor("#378ADD"),
        hAlign="CENTER", spaceAfter=16,
    ))

    story.append(Paragraph(f"<b>{tema.upper()}</b>", ParagraphStyle(
        "TopikBesar", parent=st_cover_sub,
        fontSize=18, leading=22, spaceAfter=12,
        textColor=colors.HexColor("#1A2E4A"),
    )))

    periode = (
        f"{tgl_mulai.strftime('%d %B %Y') if hasattr(tgl_mulai,'strftime') else tgl_mulai}"
        f" — "
        f"{tgl_akhir.strftime('%d %B %Y') if hasattr(tgl_akhir,'strftime') else tgl_akhir}"
    )
    story.append(Paragraph(f"Periode: {periode}", st_cover_sub))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"Dibuat: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}",
        st_cover_meta,
    ))
    story.append(Spacer(1, 0.2 * cm))
    lang_label = "Bahasa Indonesia" if bahasa == "Indonesia" else "English"
    story.append(Paragraph(f"Bahasa laporan: {lang_label}", st_cover_meta))
    story.append(Spacer(1, 1 * cm))

    story.append(HRFlowable(
        width="80%", thickness=1,
        color=colors.HexColor("#DDDDDD"),
        hAlign="CENTER", spaceAfter=8,
    ))
    story.append(Paragraph(
        "Scraper Berita Indonesia · Media Intelligence Dashboard",
        st_cover_meta,
    ))

    story.append(PageBreak())

    # ── HALAMAN 2: EXECUTIVE SUMMARY ─────────────────────────────────────────
    story.append(Paragraph(
        "Executive Summary" if bahasa == "English" else "Ringkasan Eksekutif",
        st_h1,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#378ADD"), spaceAfter=10))

    if executive_summary:
        for para in executive_summary.split("\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(para, st_body))
    else:
        story.append(Paragraph("Executive summary tidak tersedia.", st_body))

    story.append(PageBreak())

    # ── HALAMAN 3: KEY METRICS ────────────────────────────────────────────────
    story.append(Paragraph(
        "Key Metrics" if bahasa == "English" else "Metrik Utama",
        st_h1,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#378ADD"), spaceAfter=10))

    # Metric cards: 4 kolom tabel
    nss_str = f"{metrik['nss']:+.1f}" if metrik['nss'] is not None else "—"
    div_str = f"{metrik['diversity']:.1f}%" if metrik['diversity'] is not None else "—"
    prom_str = f"{metrik['prominence']:.1f}%" if metrik['prominence'] is not None else "—"
    total_str = str(metrik['total'])

    metric_data = [
        [
            Paragraph(nss_str, st_metric_val),
            Paragraph(div_str, st_metric_val),
            Paragraph(prom_str, st_metric_val),
            Paragraph(total_str, st_metric_val),
        ],
        [
            Paragraph("Net Sentiment Score", st_metric_lbl),
            Paragraph("Source Diversity", st_metric_lbl),
            Paragraph("Prominence Score", st_metric_lbl),
            Paragraph("Total Artikel", st_metric_lbl),
        ],
    ]
    col_w = (PAGE_W - 2 * MARGIN) / 4
    metric_table = Table(metric_data, colWidths=[col_w] * 4, rowHeights=[32, 18])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F7FB")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
    ]))
    story.append(metric_table)
    story.append(Spacer(1, 0.4 * cm))

    # Indikator konsensus
    if metrik.get("indikator"):
        ind = metrik["indikator"]
        story.append(Paragraph(
            f"{ind['emoji']} <b>{ind['label']}</b> — {ind['detail']}",
            st_body,
        ))
        story.append(Spacer(1, 0.3 * cm))

    # SoV tabel
    story.append(Paragraph(
        "Share of Voice per Sumber" if bahasa == "Indonesia" else "Share of Voice by Source",
        st_h2,
    ))
    sov_data = metrik.get("sov_data", [])
    if sov_data:
        sov_table_data = [
            [
                Paragraph("<b>Sumber</b>", st_label),
                Paragraph("<b>Artikel</b>", st_label),
                Paragraph("<b>SoV %</b>", st_label),
                Paragraph("<b>Sentimen Dominan</b>", st_label),
            ]
        ]
        for row in sov_data:
            sov_table_data.append([
                Paragraph(str(row["sumber"]), st_label),
                Paragraph(str(row["jumlah"]), st_label),
                Paragraph(f"{row['pct']:.1f}%", st_label),
                Paragraph(str(row["sentimen_dominan"]), st_label),
            ])
        sov_table = Table(
            sov_table_data,
            colWidths=[(PAGE_W - 2 * MARGIN) * x for x in [0.35, 0.15, 0.15, 0.35]],
        )
        sov_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A2E4A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(sov_table)

    # Chart NSS Trend (embed PNG)
    if metrik.get("fig_nss"):
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            "Tren Net Sentiment Score (NSS) Harian" if bahasa == "Indonesia"
            else "Daily NSS Trend",
            st_h2,
        ))
        png_bytes = _export_chart_png(metrik["fig_nss"], lebar=520, tinggi=200)
        if png_bytes:
            img = RLImage(io.BytesIO(png_bytes), width=16 * cm, height=6 * cm)
            story.append(img)
            story.append(Paragraph(
                "NSS = (Positif − Negatif) / Total × 100. Range −100 s/d +100.",
                st_caption,
            ))

    story.append(PageBreak())

    # ── HALAMAN 4+: CLUSTER ANALYSIS ─────────────────────────────────────────
    story.append(Paragraph(
        "Analisis per Sumber Media" if bahasa == "Indonesia" else "Analysis by Media Source",
        st_h1,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#378ADD"), spaceAfter=10))

    for sumber, summary_text in cluster_summaries:
        story.append(Paragraph(sumber, st_h2))
        for baris in summary_text.split("\n"):
            baris = baris.strip()
            if baris:
                story.append(Paragraph(baris, st_body))
        story.append(Spacer(1, 0.3 * cm))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#EEEEEE"), spaceAfter=6,
        ))

    story.append(PageBreak())

    # ── HALAMAN TERAKHIR: ARTICLE INDEX ──────────────────────────────────────
    story.append(Paragraph(
        "Indeks Artikel" if bahasa == "Indonesia" else "Article Index",
        st_h1,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#378ADD"), spaceAfter=10))

    story.append(Paragraph(
        f"Total {len(artikel_dipilih)} artikel dianalisis dalam laporan ini.",
        st_body,
    ))
    story.append(Spacer(1, 0.2 * cm))

    idx_data = [[
        Paragraph("<b>#</b>", st_label),
        Paragraph("<b>Judul</b>", st_label),
        Paragraph("<b>Sumber</b>", st_label),
        Paragraph("<b>Tanggal</b>", st_label),
        Paragraph("<b>Sentimen</b>", st_label),
    ]]
    for i, art in enumerate(artikel_dipilih, 1):
        tgl_art = art.get("tanggal")
        tgl_str = tgl_art.strftime("%d/%m/%y") if isinstance(tgl_art, datetime) else (art.get("tanggal_teks", "")[:10])
        judul = art.get("judul", "")[:80]
        link = art.get("link", "")
        sent = art.get("sentimen", "—")
        sent_color = {"Positif": "#1D9E75", "Negatif": "#D85A30"}.get(sent, "#888780")

        judul_cell = (
            Paragraph(f'<a href="{link}" color="#378ADD">{judul}</a>', st_link)
            if link else Paragraph(judul, st_label)
        )
        idx_data.append([
            Paragraph(str(i), st_label),
            judul_cell,
            Paragraph(art.get("sumber", ""), st_label),
            Paragraph(tgl_str, st_label),
            Paragraph(f'<font color="{sent_color}">{sent}</font>', st_label),
        ])

    idx_table = Table(
        idx_data,
        colWidths=[(PAGE_W - 2 * MARGIN) * x for x in [0.05, 0.45, 0.20, 0.13, 0.17]],
        repeatRows=1,
    )
    idx_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A2E4A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(idx_table)

    # Metodologi singkat
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(
        "Catatan Metodologi" if bahasa == "Indonesia" else "Methodology Notes",
        st_h2,
    ))
    metode_data = [
        [Paragraph("<b>Metrik</b>", st_label), Paragraph("<b>Cara Hitung</b>", st_label)],
        [Paragraph("NSS", st_label), Paragraph("(Positif − Negatif) / Total × 100. Range −100 s/d +100.", st_label)],
        [Paragraph("Share of Voice", st_label), Paragraph("% artikel per sumber dari total artikel.", st_label)],
        [Paragraph("Prominence Score", st_label), Paragraph("% artikel di mana tema muncul di judul.", st_label)],
        [Paragraph("Source Diversity", st_label), Paragraph("Shannon entropy dinormalisasi (0–100%). 100% = distribusi merata.", st_label)],
        [Paragraph("Sentimen", st_label), Paragraph("Lexicon-based (kata positif/negatif + negasi + penguat). Estimasi kasar.", st_label)],
        [Paragraph("Executive Summary", st_label), Paragraph("Dihasilkan oleh Gemini AI berdasarkan ringkasan artikel pilihan.", st_label)],
    ]
    metode_table = Table(
        metode_data,
        colWidths=[(PAGE_W - 2 * MARGIN) * x for x in [0.25, 0.75]],
    )
    metode_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A2E4A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(metode_table)

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def hitung_sov_data(hasil_tampil):
    """Hitung Share of Voice per sumber lengkap dengan sentimen dominan."""
    if not hasil_tampil:
        return []
    total = len(hasil_tampil)
    per_sumber = {}
    for item in hasil_tampil:
        s = item.get("sumber", "Lainnya")
        per_sumber.setdefault(s, []).append(item.get("sentimen", "Netral"))

    hasil = []
    for sumber, sentimen_list in per_sumber.items():
        jumlah = len(sentimen_list)
        cnt = Counter(sentimen_list)
        dominan = cnt.most_common(1)[0][0] if cnt else "Netral"
        hasil.append({
            "sumber": sumber,
            "jumlah": jumlah,
            "pct": round(jumlah / total * 100, 1),
            "sentimen_dominan": dominan,
        })
    return sorted(hasil, key=lambda x: x["jumlah"], reverse=True)


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

_KATA_NEGASI = {"tidak", "bukan", "jangan", "belum", "tanpa", "kurang", "gak", "ga"}
_KATA_PENGUAT = {"sangat", "sekali", "banget", "amat", "begitu", "sungguh"}

def analisis_sentimen(teks):
    """
    Return (label, skor). Label salah satu dari 'Positif'/'Negatif'/'Netral'.
    Versi 2: deteksi negasi (tidak/bukan/jangan dst di 2 kata sebelum kata
    sentimen membalik polaritasnya) dan kata penguat (sangat/sekali dst di
    2 kata sebelum ATAU sesudah menaikkan bobot jadi 2x). Tetap lexicon-based
    sederhana -- gak paham sarkasme/konteks kompleks, tapi lebih akurat dari
    versi polos buat kasus umum kayak "tidak optimis" atau "bagus sekali".
    """
    if not teks:
        return "Netral", 0
    kata_list = re.findall(r"[a-z]+", teks.lower())
    skor = 0
    for i, kata in enumerate(kata_list):
        polaritas = 1 if kata in _KATA_POSITIF else (-1 if kata in _KATA_NEGATIF else 0)
        if polaritas != 0:
            sebelum = kata_list[max(0, i - 2):i]
            sesudah = kata_list[i + 1:i + 3]
            ada_negasi = any(w in _KATA_NEGASI for w in sebelum)
            ada_penguat = any(w in _KATA_PENGUAT for w in sebelum + sesudah)
            bobot = 2 if ada_penguat else 1
            if ada_negasi:
                polaritas *= -1
            skor += polaritas * bobot
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

def cocok_semua_kata(teks, frasa):
    """Cek apakah SEMUA kata di frasa pencarian muncul di teks -- gak harus
    nempel/berurutan kayak cocok_frasa_persis, boleh tersebar di kalimat
    berbeda, asal semuanya ada di 1 teks (artikel) yang sama. Perlu isi
    lengkap artikel buat akurat -- gak cukup cuma judul/cuplikan singkat."""
    if not teks or not frasa:
        return False
    teks_lower = teks.lower()
    kata_list = frasa.lower().split()
    if not kata_list:
        return False
    return all(
        re.search(r"\b" + re.escape(kata) + r"\b", teks_lower)
        for kata in kata_list
    )

# =========================================================
# ENTITY/TOKOH EXTRACTION -- watchlist kurasi manual
# Bukan auto-NER (sengaja, NER Bahasa Indonesia gratis kurang reliable).
# User input daftar nama yang mau ditrack, tiap baris boleh berisi beberapa
# alias dipisah koma (mis. "Sri Mulyani Indrawati, Sri Mulyani, Menkeu")
# biar mention informal/parsial tetap ke-tangkep.
# =========================================================
def parse_daftar_tokoh(teks_input):
    """Parse input multi-baris jadi list {nama_utama, alias}. Baris kosong diabaikan."""
    daftar = []
    for baris in (teks_input or "").strip().split("\n"):
        alias_list = [a.strip() for a in baris.split(",") if a.strip()]
        if alias_list:
            daftar.append({"nama_utama": alias_list[0], "alias": alias_list})
    return daftar

def ekstrak_tokoh_disebut(teks, daftar_tokoh):
    """Return list nama_utama tokoh yang disebut di teks (cek semua alias-nya)."""
    if not teks or not daftar_tokoh:
        return []
    return [
        tokoh["nama_utama"] for tokoh in daftar_tokoh
        if any(cocok_frasa_persis(teks, alias) for alias in tokoh["alias"])
    ]

def buat_leaderboard_tokoh(hasil_tampil, daftar_tokoh):
    """
    Hitung jumlah sebutan + sentimen rata-rata per tokoh dari hasil pencarian.
    Return list of dict, urut dari paling banyak disebut.
    """
    if not daftar_tokoh:
        return []
    data = {}
    for item in hasil_tampil:
        teks = f"{item.get('judul', '')} {item.get('konten') or item.get('ringkasan_redaksi') or ''}"
        for nama in ekstrak_tokoh_disebut(teks, daftar_tokoh):
            entri = data.setdefault(nama, {"nama": nama, "jumlah": 0, "skor_total": 0})
            entri["jumlah"] += 1
            entri["skor_total"] += item.get("sentimen_skor", 0) or 0
    hasil = list(data.values())
    for entri in hasil:
        entri["skor_rata"] = round(entri["skor_total"] / entri["jumlah"], 2) if entri["jumlah"] else 0
    hasil.sort(key=lambda x: x["jumlah"], reverse=True)
    return hasil

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

def scrape_tempo(tema, max_halaman=3, tanggal_mulai=None, tanggal_akhir=None, progress_callback=None):
    """
    Tempo.co via POST JSON API (/api/search/articles) -- Nuxt.js frontend.
    Filter tanggal pakai unix timestamp UTC. Pagination via offset (kelipatan 20).
    Ringkasan: summary array (artikel premium) atau description (strip HTML).

    BUGFIX v2: Pakai fresh session khusus Tempo (bukan SESSION global) dengan
    header minimal persis seperti script Colab yang terbukti berhasil.
    SESSION global punya banyak HTML-scraping header (Sec-Fetch-User, Upgrade-
    Insecure-Requests, Accept: text/html, dll.) yang membuat Cloudflare Tempo
    memperlakukan request secara berbeda vs. session bersih. Fresh session juga
    menghindari interferensi cookie dari scraper lain yang pakai SESSION yang sama.
    """
    hasil = []
    limit = 20
    url_referer = f"https://www.tempo.co/search?q={quote(tema)}"

    # Fresh session minimal -- persis seperti Colab yang terbukti berhasil
    sesi_tempo = requests.Session()
    sesi_tempo.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/149.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    })

    # Warm-up GET: buka halaman search supaya sesi_tempo mendapat cookie
    # Cloudflare (_cfuvid) sebelum POST ke API. Terbukti di Colab: tanpa ini hits=[].
    try:
        time.sleep(random.uniform(0.8, 1.5))
        warmup = sesi_tempo.get(url_referer, timeout=15)
        if warmup.status_code != 200 and progress_callback:
            progress_callback(f"⚠️ Tempo warm-up: HTTP {warmup.status_code}")
    except requests.RequestException as e:
        if progress_callback:
            progress_callback(f"⚠️ Tempo warm-up gagal: {type(e).__name__}")

    # Header XHR untuk POST -- sama seperti yang dikirim browser saat fetch() ke API
    headers_api = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Origin": "https://www.tempo.co",
        "Referer": url_referer,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }

    # Konversi date ke unix timestamp UTC
    unix_mulai = None
    unix_akhir = None
    if tanggal_mulai and tanggal_akhir:
        unix_mulai = int(calendar.timegm(tanggal_mulai.timetuple()))
        unix_akhir = int(calendar.timegm(tanggal_akhir.timetuple())) + 86399  # 23:59:59

    for p in range(max_halaman):
        offset = p * limit

        if unix_mulai and unix_akhir:
            filter_str = (
                f'unix_published_at >= {unix_mulai} AND unix_published_at <= {unix_akhir}'
                f' AND domain.domain = "www.tempo.co"'
            )
        else:
            filter_str = 'domain.domain = "www.tempo.co"'

        payload = {
            "q": tema,
            "offset": offset,
            "limit": limit,
            "attributesToSearchOn": ["title_digital"],
            "filter": filter_str,
            "sort": ["unix_published_at:desc"],
        }

        try:
            time.sleep(random.uniform(0.8, 1.5))
            resp = sesi_tempo.post(
                "https://www.tempo.co/api/search/articles",
                json=payload,
                headers=headers_api,
                timeout=15,
            )
            if resp.status_code != 200:
                if progress_callback:
                    progress_callback(f"⚠️ Tempo: HTTP {resp.status_code} di halaman {p+1}")
                break
            data = resp.json()
        except requests.RequestException as e:
            if progress_callback:
                progress_callback(f"⚠️ Tempo: {type(e).__name__} di halaman {p+1}")
            break

        hits = data.get("hits", [])
        if not hits:
            if progress_callback and p == 0:
                progress_callback(
                    f"⚠️ Tempo: API mengembalikan 0 hasil "
                    f"(estimatedTotalHits={data.get('estimatedTotalHits', '?')})"
                )
            break

        for art in hits:
            judul = art.get("title_digital")
            canonical = art.get("canonical_url")
            if not judul or not canonical:
                continue

            link = f"https://www.tempo.co/{canonical}"

            tanggal_obj = None
            tanggal_teks = art.get("published_at") or ""
            if tanggal_teks:
                try:
                    tanggal_obj = datetime.strptime(tanggal_teks[:19], "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    tanggal_obj = parse_tanggal(tanggal_teks)

            summary_list = art.get("summary") or []
            if summary_list:
                ringkasan_redaksi = " ".join(summary_list)
            else:
                desc_html = art.get("description") or ""
                ringkasan_redaksi = bersihkan_isi_html(desc_html) if desc_html else None

            hasil.append({
                "sumber": "Tempo.co",
                "judul": judul,
                "link": link,
                "ringkasan_redaksi": ringkasan_redaksi,
                "tanggal_teks": tanggal_teks,
                "tanggal": tanggal_obj,
                "konten": None,
            })

        total_hits = data.get("estimatedTotalHits", 0)
        if not hits or offset + limit >= total_hits or len(hits) < limit:
            break

    return hasil


def scrape_theconversation(tema, max_halaman=3, tanggal_mulai=None, tanggal_akhir=None, progress_callback=None):
    """
    The Conversation Indonesia (theconversation.com/id) -- server-side rendered,
    requests + BeautifulSoup biasa cukup (SSR, bukan JS-rendered).
    Date filter native via date_from/date_to (format YYYY-MM-DD).
    Pagination via page=N. Stop kalau 0 artikel ditemukan atau < 10 (halaman terakhir).
    Konten: artikel akademik/analitis panjang, biasanya berkualitas tinggi untuk
    topik kebijakan/ekonomi/sosial.
    """
    hasil = []
    base_url = "https://theconversation.com/id/search"

    for p in range(1, max_halaman + 1):
        params = {
            "q": tema,
            "sort": "relevancy",
            "language": "id",
        }
        if tanggal_mulai and tanggal_akhir:
            params["date"] = "custom"
            params["date_from"] = tanggal_mulai.strftime("%Y-%m-%d")
            params["date_to"] = tanggal_akhir.strftime("%Y-%m-%d")
        if p > 1:
            params["page"] = p

        url = f"{base_url}?{urlencode(params)}"
        html = ambil_halaman(url, progress_callback=progress_callback)
        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")
        artikel_list = soup.select("article.result.published")
        if not artikel_list:
            if progress_callback and p == 1:
                progress_callback(
                    f"⚠️ The Conversation: 0 artikel cocok selector di halaman {p} "
                    f"-- struktur HTML mungkin berubah."
                )
            break

        for art in artikel_list:
            judul_tag = art.select_one("h3 a")
            time_tag = art.select_one("time[datetime]")
            # Excerpt ada di <p> di dalam article, tapi bukan p.byline
            desc_tag = art.select_one("div.grid-nine > p:not(.byline)")

            if not judul_tag:
                continue

            judul = judul_tag.get("title") or judul_tag.get_text(strip=True)
            href = judul_tag.get("href", "")
            link = f"https://theconversation.com{href}" if href.startswith("/") else href

            tanggal_obj = None
            tanggal_teks = None
            if time_tag:
                tanggal_teks = time_tag.get_text(strip=True)
                dt_attr = time_tag.get("datetime", "")
                if dt_attr:
                    try:
                        tanggal_obj = datetime.strptime(dt_attr[:19], "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        tanggal_obj = parse_tanggal(tanggal_teks)

            ringkasan_redaksi = desc_tag.get_text(strip=True) if desc_tag else None

            if judul and link:
                hasil.append({
                    "sumber": "The Conversation",
                    "judul": judul,
                    "link": link,
                    "ringkasan_redaksi": ringkasan_redaksi,
                    "tanggal_teks": tanggal_teks,
                    "tanggal": tanggal_obj,
                    "konten": None,
                })

        # Halaman terakhir biasanya < 10 hasil
        if len(artikel_list) < 10:
            break

    return hasil

SCRAPER_PER_SUMBER = {
    "CNBC Indonesia": scrape_cnbcindonesia,
    "CNN Indonesia": scrape_cnnindonesia,
    "Detik": scrape_detik,
    "Kompas": scrape_kompas,
    "Republika": scrape_republika,
    "Liputan6": scrape_liputan6,
    "Kumparan": scrape_kumparan,
    "Tempo.co": scrape_tempo,
    "The Conversation": scrape_theconversation,
}
SUMBER_DENGAN_FILTER_TANGGAL_NATIVE = {"Detik", "Republika", "CNN Indonesia", "Liputan6", "CNBC Indonesia", "Tempo.co", "The Conversation"}

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
    # tempo.co: konten terpecah di banyak div#content-wrapper (bukan satu container tunggal).
    # Ditangani sebagai special case di scrape_isi_artikel(), bukan via selector generik.
    # Entry ini sengaja dihapus -- kalau ikut di sini, logic find() akan selalu None.
    # theconversation.com: SSR bersih, konten ada di div.article-body
    "theconversation.com": ("div", {"class_": "article-body"}),
}

def scrape_isi_artikel(url, progress_callback=None):
    # ── Special case: Tempo.co ────────────────────────────────────────────────
    # Masalah 1: SESSION global punya HTML-scraping headers yang membuat
    # Cloudflare Tempo return halaman berbeda (tanpa div#content-wrapper).
    # Solusi: fresh session minimal, persis seperti Colab yang terbukti berhasil.
    # Masalah 2: isi artikel terpecah di banyak div#content-wrapper (bukan
    # satu container tunggal). Wrapper dengan class paywall-content berisi
    # "Pilihan Editor:" -- di-skip, bukan isi artikel.
    if "tempo.co" in url:
        sesi_artikel = requests.Session()
        sesi_artikel.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/149.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        })
        try:
            time.sleep(random.uniform(0.8, 1.5))
            resp = sesi_artikel.get(url, timeout=15)
            if resp.status_code != 200:
                return f"Gagal mengambil halaman (HTTP {resp.status_code})"
            html = resp.text
        except requests.RequestException as e:
            return f"Gagal mengambil halaman ({type(e).__name__})"

        soup = BeautifulSoup(html, "html.parser")
        wrappers = soup.find_all("div", id="content-wrapper")
        teks_list = []
        for w in wrappers:
            # Skip wrapper paywall -- berisi "Pilihan Editor:" bukan isi artikel
            if "paywall-content" in (w.get("class") or []):
                continue
            teks = w.get_text(separator=" ", strip=True)
            if teks:
                teks_list.append(teks)

        if not teks_list:
            return "Konten tidak ditemukan"

        isi = " ".join(teks_list)
        # Bersihkan sisa noise "Pilihan Editor: ..." kalau masih ada
        isi = re.sub(r'Pilihan Editor\s*:?\s*[^.!?]*[.!?]?', '', isi, flags=re.IGNORECASE)
        isi = re.sub(r'\s{2,}', ' ', isi).strip()
        return isi if isi else "Konten tidak ditemukan"

    # ── Generic selector untuk sumber lain ────────────────────────────────────
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

def cari_berita(tema, sumber_dipilih, tanggal_mulai, tanggal_akhir, max_halaman, ambil_konten, match_persis=False, match_semua_kata=False, progress_callback=None):
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

    # Filter "semua kata" -- BEDA dari frasa persis, ini jalan SETELAH isi
    # lengkap di-fetch (di atas), karena wajib cek seluruh isi artikel, bukan
    # cuma judul/cuplikan. Trade-off: gak bisa "buang duluan sebelum fetch"
    # kayak frasa persis, semua kandidat harus selesai di-fetch dulu baru disaring.
    if match_semua_kata and tema.strip():
        hasil_final = [
            item for item in hasil_final
            if cocok_semua_kata(f"{item.get('judul', '')} {item.get('konten') or ''}", tema)
        ]

    return hasil_final

@st.cache_data(ttl=900, show_spinner=False)
def cari_berita_cached(tema, sumber_dipilih, tanggal_mulai, tanggal_akhir, max_halaman, ambil_konten, match_persis, match_semua_kata=False, _progress_callback=None):
    """
    Wrapper cache (TTL 15 menit) di atas cari_berita. Parameter diawali underscore
    (_progress_callback) dikecualikan Streamlit dari cache key -- soalnya fungsi
    gak bisa di-hash, dan emang gak relevan buat nentuin apakah hasilnya sama.
    Cache hit = gak ada scraping ulang sama sekali, jadi progress_callback gak
    akan kepanggil; itu sudah benar (gak ada progress buat dilaporkan).
    """
    return cari_berita(
        tema, sumber_dipilih, tanggal_mulai, tanggal_akhir,
        max_halaman, ambil_konten, match_persis, match_semua_kata,
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
st.caption("Cari berita dari CNBC Indonesia, CNN Indonesia, Detik, Kompas, Republika, Kumparan, Tempo.co, dan The Conversation berdasarkan tema dan rentang tanggal — gratis, tanpa API berbayar.")

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
        mode_pencarian = st.radio(
            "Mode pencocokan kata kunci",
            options=["Standar", "Frasa persis", "Semua kata (isi lengkap)"],
            index=0,
            help=(
                "**Standar**: query dikirim apa adanya ke search engine tiap situs "
                "(perilaku pencocokan tergantung algoritma masing-masing situs).\n\n"
                "**Frasa persis**: hanya artikel yang judul/cuplikannya mengandung "
                "frasa PERSIS nempel berurutan (mis. 'indonesia esport' harus nempel "
                "persis begitu). Dicek SEBELUM ambil isi lengkap, jadi hemat resource "
                "tapi cuma nyentuh judul + cuplikan singkat, bukan isi lengkap.\n\n"
                "**Semua kata (isi lengkap)**: semua kata harus ada di artikel, BOLEH "
                "tersebar di kalimat berbeda (mis. 'presiden indonesia optimis' cocok "
                "walau 3 kata itu di 3 kalimat terpisah). Wajib cek isi lengkap, jadi "
                "otomatis mengaktifkan opsi 'Ambil isi lengkap' di atas."
            )
        )
        match_persis = mode_pencarian == "Frasa persis"
        match_semua_kata = mode_pencarian == "Semua kata (isi lengkap)"
        if match_semua_kata and not ambil_konten:
            ambil_konten = True
            st.caption("ℹ️ 'Ambil isi lengkap' otomatis diaktifkan -- mode ini wajib cek isi artikel.")

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
                    match_semua_kata=match_semua_kata,
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

        sub_artikel, sub_dashboard = st.tabs(["📰 Daftar Artikel", "📊 Dashboard Analitik"])

        with sub_artikel:
            # ── Panel Filter & Sort ───────────────────────────────────────────
            with st.container():
                col_sort, col_sent, col_src = st.columns([1, 1, 2])
                with col_sort:
                    sort_by = st.selectbox(
                        "Urutkan",
                        ["Tanggal (terbaru)", "Tanggal (terlama)", "Sentimen (positif dulu)", "Sentimen (negatif dulu)", "Sumber (A–Z)"],
                        label_visibility="collapsed",
                    )
                with col_sent:
                    filter_sentimen = st.multiselect(
                        "Sentimen",
                        ["Positif", "Netral", "Negatif"],
                        default=["Positif", "Netral", "Negatif"],
                        placeholder="Semua sentimen",
                        label_visibility="collapsed",
                    )
                with col_src:
                    filter_sumber = st.multiselect(
                        "Sumber",
                        sumber_unik,
                        default=sumber_unik,
                        placeholder="Semua sumber",
                        label_visibility="collapsed",
                    )

            # Terapkan filter
            hasil_tampil = [
                h for h in hasil
                if h["sumber"] in (filter_sumber or sumber_unik)
                and h.get("sentimen", "Netral") in (filter_sentimen or ["Positif", "Netral", "Negatif"])
            ]

            # Terapkan sort
            _URUT_SENT = {"Positif": 0, "Netral": 1, "Negatif": 2}
            if sort_by == "Tanggal (terbaru)":
                hasil_tampil.sort(key=lambda x: x.get("tanggal") or datetime.min, reverse=True)
            elif sort_by == "Tanggal (terlama)":
                hasil_tampil.sort(key=lambda x: x.get("tanggal") or datetime.min)
            elif sort_by == "Sentimen (positif dulu)":
                hasil_tampil.sort(key=lambda x: _URUT_SENT.get(x.get("sentimen", "Netral"), 1))
            elif sort_by == "Sentimen (negatif dulu)":
                hasil_tampil.sort(key=lambda x: _URUT_SENT.get(x.get("sentimen", "Netral"), 1), reverse=True)
            elif sort_by == "Sumber (A–Z)":
                hasil_tampil.sort(key=lambda x: x.get("sumber", ""))

            st.caption(f"Menampilkan {len(hasil_tampil)} dari {len(hasil)} artikel")
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

        with sub_dashboard:
            # ── Fase 2+3: Score Card Row ──────────────────────────────────────
            nss_val = hitung_nss(hasil_tampil)
            div_val = hitung_source_diversity(hasil_tampil)
            prom_val = hitung_prominence(hasil_tampil, tema)
            indikator = hitung_konsensus_kontroversi(hasil_tampil)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                if nss_val is not None:
                    warna_nss = "normal" if nss_val >= 0 else "inverse"
                    st.metric(
                        "Net Sentiment Score",
                        f"{nss_val:+.1f}",
                        help="(Positif − Negatif) / Total × 100. Range −100 s/d +100.",
                        delta_color=warna_nss,
                    )
                else:
                    st.metric("Net Sentiment Score", "—")
            with col_m2:
                st.metric(
                    "Source Diversity",
                    f"{div_val:.1f}%" if div_val is not None else "—",
                    help="Shannon entropy dinormalisasi. 100% = distribusi sempurna merata antar sumber.",
                )
            with col_m3:
                st.metric(
                    "Prominence Score",
                    f"{prom_val:.1f}%" if prom_val is not None else "—",
                    help="% artikel di mana tema muncul tepat di judul (bukan cuma di isi).",
                )
            with col_m4:
                tgl_unik = sorted({
                    item["tanggal"].date() for item in hasil_tampil
                    if isinstance(item.get("tanggal"), datetime)
                })
                rentang_str = (
                    f"{tgl_unik[0].strftime('%d %b')} – {tgl_unik[-1].strftime('%d %b')}"
                    if len(tgl_unik) >= 2 else
                    (tgl_unik[0].strftime('%d %b') if tgl_unik else "—")
                )
                st.metric(
                    "Total Artikel",
                    len(hasil_tampil),
                    delta=rentang_str,
                    help="Total artikel unik yang ditampilkan saat ini.",
                )

            # ── Fase 3: Indikator Konsensus / Kontroversial ──────────────────
            if indikator:
                st.info(
                    f"{indikator['emoji']} **{indikator['label']}** — {indikator['detail']}",
                )

            st.divider()

            # ── Fase 3: Tren NSS per Hari (interaktif) ───────────────────────
            fig_nss_trend = buat_chart_nss_trend(hasil_tampil)
            if fig_nss_trend:
                st.caption("📈 Tren Net Sentiment Score (NSS) per hari")
                st.plotly_chart(fig_nss_trend, use_container_width=True)

            # ── Word Cloud + Top Keywords (existing) ─────────────────────────
            wc = buat_wordcloud(hasil_tampil)
            if wc:
                fig_wc, ax_wc = plt.subplots(figsize=(10, 3.5))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)
                plt.close(fig_wc)
            else:
                st.caption("Belum cukup teks untuk membuat word cloud.")

            top_kw = hitung_top_keywords(hasil_tampil)
            fig_kw = buat_chart_top_keywords(top_kw)
            if fig_kw:
                st.caption("Top 15 kata kunci")
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.caption("Belum cukup teks buat kata kunci.")

            st.divider()

            # ── Volume per Sumber & per Hari (existing) ───────────────────────
            df_stat = buat_dataframe_statistik(hasil_tampil)
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Jumlah artikel per sumber")
                fig_sumber = buat_chart_per_sumber(df_stat)
                if fig_sumber:
                    st.plotly_chart(fig_sumber, use_container_width=True)
                else:
                    st.caption("Belum ada data.")
            with col_b:
                st.caption("Jumlah artikel per hari")
                fig_hari = buat_chart_per_hari(df_stat)
                if fig_hari:
                    st.plotly_chart(fig_hari, use_container_width=True)
                else:
                    st.caption("Tanggal tidak terbaca untuk sebagian besar artikel.")

            # ── Fase 3: Profil Bias Sentimen per Sumber ──────────────────────
            fig_bias = buat_chart_bias_sumber(hasil_tampil)
            if fig_bias:
                st.caption("🎯 Profil bias sentimen per sumber (% dari total artikel sumber tsb.)")
                st.plotly_chart(fig_bias, use_container_width=True)

            # ── Volume × Sentimen Harian (existing) ───────────────────────────
            st.caption("Volume artikel per hari × sentimen")
            fig_vol_sent = buat_chart_volume_sentimen_harian(hasil_tampil)
            if fig_vol_sent:
                st.plotly_chart(fig_vol_sent, use_container_width=True)
            else:
                st.caption("Belum cukup data sentimen+tanggal buat chart ini.")

            st.caption("Distribusi sentimen (lexicon-based, kasar)")
            fig_sent_dist = buat_chart_sentimen_distribusi(hasil_tampil)
            if fig_sent_dist:
                st.plotly_chart(fig_sent_dist, use_container_width=True)
            else:
                st.caption("Belum ada data sentimen (perlu ambil isi/ringkasan dulu).")

            # ── Leaderboard Tokoh (existing) ──────────────────────────────────
            with st.expander("🎭 Daftar Tokoh yang Dipantau"):
                st.text_area(
                    "1 tokoh per baris. Alias boleh dipisah koma supaya sebutan informal/parsial "
                    "tetap ke-tangkep (contoh: Sri Mulyani Indrawati, Sri Mulyani, Menkeu)",
                    height=100,
                    key="input_tokoh",
                )
            daftar_tokoh = parse_daftar_tokoh(st.session_state.get("input_tokoh", ""))

            if daftar_tokoh:
                st.divider()
                st.caption("🎭 Leaderboard tokoh")
                leaderboard = buat_leaderboard_tokoh(hasil_tampil, daftar_tokoh)
                if leaderboard:
                    for entri in leaderboard:
                        emoji_tokoh = "🟢" if entri["skor_rata"] > 0 else ("🔴" if entri["skor_rata"] < 0 else "⚪")
                        st.write(f"{emoji_tokoh} **{entri['nama']}** — {entri['jumlah']}x disebut · sentimen rata-rata: {entri['skor_rata']}")
                else:
                    st.caption("Belum ada tokoh dari daftar Anda yang ke-deteksi di hasil ini.")

            # ── Liputan Lintas Sumber (existing) ──────────────────────────────
            klaster = st.session_state.get("klaster", [])
            klaster_tampil = [
                k for k in klaster
                if any(hasil[i] in hasil_tampil for i in k)
            ]
            if klaster_tampil:
                st.divider()
                st.caption(f"🔗 Liputan Sama dari Beberapa Sumber ({len(klaster_tampil)} topik)")
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

            # ── FASE 5: Generate PDF Report ───────────────────────────────────
            st.divider()
            if not GEMINI_API_KEY:
                st.caption("📄 Fitur PDF Report belum dikonfigurasi developer.")
            elif not st.session_state.get("ai_unlocked", False):
                st.caption("🔒 **PDF Report** terkunci. Masukkan password di sidebar untuk mengaktifkan.")
            elif not ambil_konten:
                st.warning(
                    "📄 **Generate PDF Report** membutuhkan isi lengkap artikel. "
                    "Aktifkan opsi **'Ambil isi lengkap artikel'** di Pengaturan Lanjutan lalu cari ulang."
                )
            else:
                st.subheader("📄 Generate PDF Report")

                with st.expander("⚙️ Konfigurasi Report", expanded=True):
                    col_lang, col_info = st.columns([1, 2])
                    with col_lang:
                        bahasa_pdf = st.selectbox(
                            "Bahasa laporan",
                            ["Indonesia", "English"],
                            key="bahasa_pdf",
                        )
                    with col_info:
                        total_art = len(hasil_tampil)
                        if total_art > PDF_MAX_ARTIKEL:
                            st.info(
                                f"Ada **{total_art} artikel** — sistem akan otomatis memilih "
                                f"**{PDF_MAX_ARTIKEL} terbaik** berdasarkan relevansi & diversitas sumber. "
                                f"Anda bisa deselect artikel di bawah."
                            )
                        else:
                            st.success(f"Semua **{total_art} artikel** akan dimasukkan ke report.")

                    # Auto-select artikel
                    indeks_auto = auto_pilih_artikel(hasil_tampil, PDF_MAX_ARTIKEL)
                    opsi_artikel = {
                        f"[{item.get('sumber','')}] {item.get('judul','')[:70]}": i
                        for i, item in enumerate(hasil_tampil)
                    }
                    opsi_dipilih_default = [
                        label for label, idx in opsi_artikel.items()
                        if idx in indeks_auto
                    ]
                    artikel_terpilih_label = st.multiselect(
                        f"Artikel yang masuk report (maks {PDF_MAX_ARTIKEL})",
                        options=list(opsi_artikel.keys()),
                        default=opsi_dipilih_default,
                        key="pdf_artikel_select",
                    )
                    artikel_dipilih_final = [
                        hasil_tampil[opsi_artikel[label]]
                        for label in artikel_terpilih_label
                        if label in opsi_artikel
                    ]

                    if len(artikel_dipilih_final) > PDF_MAX_ARTIKEL:
                        st.warning(f"Terlalu banyak artikel dipilih. Maksimal {PDF_MAX_ARTIKEL}. Silakan deselect beberapa.")
                        artikel_dipilih_final = artikel_dipilih_final[:PDF_MAX_ARTIKEL]

                if artikel_dipilih_final and st.button(
                    f"🚀 Generate & Download PDF ({len(artikel_dipilih_final)} artikel)",
                    type="primary",
                    use_container_width=True,
                    key="btn_generate_pdf",
                ):
                    progress_pdf = st.empty()
                    error_pdf = None
                    pdf_bytes = None

                    def _pdf_progress(msg):
                        progress_pdf.info(f"⏳ {msg}")

                    try:
                        # Layer 2: cluster summaries
                        _pdf_progress("Menganalisis narasi per sumber media...")
                        cluster_sums = generate_cluster_summaries(
                            artikel_dipilih_final, GEMINI_API_KEY,
                            bahasa=bahasa_pdf,
                            progress_callback=_pdf_progress,
                        )

                        # Hitung metrik untuk PDF
                        nss_pdf = hitung_nss(artikel_dipilih_final)
                        div_pdf = hitung_source_diversity(artikel_dipilih_final)
                        prom_pdf = hitung_prominence(artikel_dipilih_final, tema)
                        ind_pdf = hitung_konsensus_kontroversi(artikel_dipilih_final)
                        sov_pdf = hitung_sov_data(artikel_dipilih_final)
                        fig_nss_pdf = buat_chart_nss_trend(artikel_dipilih_final)

                        metrik_pdf = {
                            "total": len(artikel_dipilih_final),
                            "nss": nss_pdf,
                            "diversity": div_pdf,
                            "prominence": prom_pdf,
                            "indikator": ind_pdf,
                            "sov": ", ".join(f"{r['sumber']} {r['pct']:.0f}%" for r in sov_pdf[:5]),
                            "sov_data": sov_pdf,
                            "fig_nss": fig_nss_pdf,
                        }

                        # Layer 3: executive summary
                        _pdf_progress("Menyusun executive summary dengan Gemini...")
                        exec_sum, err_exec = generate_executive_summary(
                            tema=tema,
                            tgl_mulai=tanggal_mulai,
                            tgl_akhir=tanggal_akhir,
                            metrik=metrik_pdf,
                            cluster_summaries=cluster_sums,
                            api_key=GEMINI_API_KEY,
                            bahasa=bahasa_pdf,
                        )
                        if err_exec:
                            exec_sum = f"[Executive summary tidak tersedia: {err_exec}]"

                        # Build PDF
                        _pdf_progress("Membuat file PDF...")
                        pdf_bytes = buat_pdf_report(
                            tema=tema,
                            tgl_mulai=tanggal_mulai,
                            tgl_akhir=tanggal_akhir,
                            hasil_tampil=hasil_tampil,
                            artikel_dipilih=artikel_dipilih_final,
                            executive_summary=exec_sum,
                            cluster_summaries=cluster_sums,
                            metrik=metrik_pdf,
                            bahasa=bahasa_pdf,
                        )

                    except Exception as e:
                        error_pdf = str(e)

                    progress_pdf.empty()

                    if error_pdf:
                        st.error(f"Gagal generate PDF: {error_pdf}")
                    elif pdf_bytes:
                        nama_file = (
                            f"media_report_{re.sub(r'[^a-zA-Z0-9]', '_', tema)}"
                            f"_{datetime.now().strftime('%Y%m%d')}.pdf"
                        )
                        st.success("✅ PDF berhasil dibuat!")

                        # Preview executive summary sebelum download
                        if exec_sum and not exec_sum.startswith("["):
                            with st.expander("📋 Preview Executive Summary"):
                                st.write(exec_sum)

                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_bytes,
                            file_name=nama_file,
                            mime="application/pdf",
                            use_container_width=True,
                            type="primary",
                        )
                elif not artikel_dipilih_final:
                    st.warning("Pilih minimal 1 artikel untuk generate report.")

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
