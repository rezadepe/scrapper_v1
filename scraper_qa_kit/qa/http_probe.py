"""
http_probe.py
Menyadap semua request HTTP yang dibuat scraper (lewat requests.Session.request),
tanpa perlu mengubah app.py. Tiga mode:

- "monitor": request tetap ke internet, tapi dicatat (status, durasi, ukuran,
              deteksi halaman blokir/Cloudflare). Dipakai health check & benchmark.
- "record" : seperti monitor, plus isi respons disimpan ke "tape" (fixture).
- "replay" : TIDAK ke internet. Respons diambil dari tape. Dipakai unit test offline.

Catatan: semua scraper di app.py memakai requests.Session (SESSION global maupun
sesi khusus Tempo), dan Session.get/post memanggil Session.request -> semua tersadap.
"""
import hashlib
import json
import threading
import time
from collections import Counter

import requests
from requests.structures import CaseInsensitiveDict

PENANDA_BLOKIR = (
    "cf-chl", "challenge-platform", "just a moment...", "attention required! | cloudflare",
    "cf-browser-verification", "captcha", "access denied", "request blocked",
)
_request_asli = requests.Session.request


def _url_lengkap(method, url, params):
    try:
        return requests.Request(method.upper(), url, params=params).prepare().url
    except Exception:
        return url


def _body_teks(data, json_body):
    if json_body is not None:
        return json.dumps(json_body, sort_keys=True, ensure_ascii=False)
    if data is None:
        return ""
    if isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8", errors="replace")
    if isinstance(data, dict):
        return json.dumps(data, sort_keys=True, ensure_ascii=False)
    return str(data)


def kunci_request(method, url_lengkap, body):
    return hashlib.sha1(f"{method.upper()} {url_lengkap}\n{body}".encode("utf-8")).hexdigest()


def terlihat_diblokir(teks):
    awal = (teks or "")[:8000].lower()
    return any(p in awal for p in PENANDA_BLOKIR)


class HttpProbe:
    def __init__(self, mode="monitor", tape=None):
        assert mode in ("monitor", "record", "replay")
        self.mode = mode
        self.tape = tape if tape is not None else {}
        self.catatan = []          # satu dict per request
        self.miss = []             # request yang tidak ada di tape (mode replay)
        self._lock = threading.Lock()
        self._label = None

    # -- label dipakai untuk mengelompokkan request per sumber ---------------
    def set_label(self, label):
        self._label = label

    # -- pasang / lepas ------------------------------------------------------
    def __enter__(self):
        probe = self

        def request_tersadap(sesi, method, url, params=None, data=None, json=None, **kwargs):
            return probe._tangani(sesi, method, url, params, data, json, kwargs)

        requests.Session.request = request_tersadap
        if self.mode == "replay":
            self._sleep_asli = time.sleep
            time.sleep = lambda *_a, **_k: None  # jeda acak scraper tidak perlu saat replay
        return self

    def __exit__(self, *args):
        requests.Session.request = _request_asli
        if self.mode == "replay":
            time.sleep = self._sleep_asli
        return False

    # -- inti ----------------------------------------------------------------
    def _tangani(self, sesi, method, url, params, data, json_body, kwargs):
        url_full = _url_lengkap(method, url, params)
        body = _body_teks(data, json_body)
        kunci = kunci_request(method, url_full, body)
        entri = {"label": self._label, "method": method.upper(), "url": url_full,
                 "status": None, "durasi_s": 0.0, "bytes": 0, "error": None, "diblokir": False}

        if self.mode == "replay":
            simpanan = self.tape.get(kunci)
            if simpanan is None:
                with self._lock:
                    self.miss.append({"method": method.upper(), "url": url_full, "body": body[:300]})
                return self._respons_palsu(url_full, 599, "", {})
            entri.update(status=simpanan["status"], bytes=len(simpanan["text"]))
            with self._lock:
                self.catatan.append(entri)
            return self._respons_palsu(url_full, simpanan["status"], simpanan["text"], simpanan.get("headers", {}))

        t0 = time.perf_counter()
        try:
            resp = _request_asli(sesi, method, url, params=params, data=data, json=json_body, **kwargs)
        except Exception as e:
            entri.update(durasi_s=round(time.perf_counter() - t0, 3), error=type(e).__name__)
            with self._lock:
                self.catatan.append(entri)
            raise
        teks = resp.text
        entri.update(
            status=resp.status_code,
            durasi_s=round(time.perf_counter() - t0, 3),
            bytes=len(resp.content or b""),
            diblokir=resp.status_code in (403, 429, 503) or terlihat_diblokir(teks),
        )
        with self._lock:
            self.catatan.append(entri)
            if self.mode == "record":
                self.tape[kunci] = {
                    "method": method.upper(), "url": url_full, "body": body,
                    "status": resp.status_code, "text": teks,
                    "headers": {"Content-Type": resp.headers.get("Content-Type", "")},
                }
        return resp

    @staticmethod
    def _respons_palsu(url, status, teks, headers):
        r = requests.Response()
        r.status_code = status
        r._content = (teks or "").encode("utf-8")
        r.encoding = "utf-8"
        r.url = url
        r.headers = CaseInsensitiveDict(headers or {})
        return r

    # -- ringkasan -----------------------------------------------------------
    def ringkas(self, label=None):
        data = [c for c in self.catatan if label is None or c["label"] == label]
        status = Counter(str(c["status"] if c["status"] is not None else c["error"]) for c in data)
        gagal = [c for c in data if c["error"] or (c["status"] or 0) >= 400]
        durasi = [c["durasi_s"] for c in data if c["durasi_s"]]
        return {
            "request_total": len(data),
            "request_gagal": len(gagal),
            "status_http": dict(status),
            "terdeteksi_blokir": sum(1 for c in data if c["diblokir"]),
            "total_kb": round(sum(c["bytes"] for c in data) / 1024, 1),
            "latensi_rata_s": round(sum(durasi) / len(durasi), 3) if durasi else None,
            "latensi_maks_s": round(max(durasi), 3) if durasi else None,
            "contoh_url_gagal": [f'{c["status"] or c["error"]} {c["url"][:150]}' for c in gagal[:3]],
        }
