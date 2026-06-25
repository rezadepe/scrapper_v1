"""
event_tracker.py
Modul tracking user activity ke Supabase.
- Dipanggil dari app.py via: from event_tracker import log_event
- Tidak ada Streamlit dependency — pure Python + supabase-py
- Silent-fail: error tracking TIDAK boleh ganggu fitur utama
- Cleanup otomatis 2% probabilitas per panggilan (hapus data >45 hari)
"""

import random
from datetime import datetime


def log_event(supabase, session_id: str, event_type: str, payload: dict = None):
    """
    Insert satu event ke tabel Supabase `events`.

    Args:
        supabase   : Supabase client (dari get_supabase_client() di app.py)
        session_id : UUID string dari st.session_state["session_id"]
        event_type : Salah satu dari: search, tab_switch, ai_unlock_attempt,
                     gemini_summary, gemini_compare, pdf_generate,
                     watchlist_action, error
        payload    : Dict bebas, disimpan sebagai JSONB. Opsional.
    """
    if not supabase:
        return

    try:
        supabase.table("events").insert({
            "session_id": session_id,
            "event_type": event_type,
            "payload": payload or {},
        }).execute()
    except Exception:
        # Silent-fail — tracking error tidak boleh crash app
        pass

    # Cleanup probabilistik 2%: hapus event lebih dari 45 hari
    if random.random() < 0.02:
        try:
            supabase.rpc("delete_old_events").execute()
        except Exception:
            # Fallback: pakai query langsung kalau RPC belum dibuat
            try:
                supabase.table("events").delete().lt(
                    "created_at",
                    (datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
                ).execute()
            except Exception:
                pass
