# app.py (fixed for mistral-7b-instruct-v0.2.Q3_K_M.gguf)
import os, json, sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
from ctransformers import AutoModelForCausalLM

# ------------------ Paths / Files ------------------ #
DB_PATH       = Path("bikemarket.db")
CATALOG_CSV   = Path("data/bikes.csv")
MODEL_DIR     = Path("models")
MODEL_FILE    = "mistral-7b-instruct-v0.2.Q3_K_M.gguf"   # ✅ your new model
MODEL_PATH    = MODEL_DIR / MODEL_FILE

st.set_page_config(page_title="BikeBot-AI - Buy & Sell Bikes", page_icon="🏍️", layout="wide")

def find_model():
    """Pick first .gguf in models/ if MODEL_FILE not found."""
    if MODEL_PATH.exists():
        return MODEL_FILE
    ggufs = list(MODEL_DIR.glob("*.gguf"))
    return ggufs[0].name if ggufs else None


# ------------------ Model Loader ------------------ #
def load_llm():
    """Load GGUF model via ctransformers (CPU by default)."""
    model_file = find_model()
    if not model_file:
        st.error(f"No .gguf model found in {MODEL_DIR.resolve()}. Please place your model there.")
        st.stop()

    return AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    model_file=model_file,
    model_type="mistral",
    gpu_layers=0,
    max_new_tokens=128,
)


_llm = None
def get_llm():
    global _llm
    if _llm is None:
        _llm = load_llm()
    return _llm

def mistral_inst(prompt: str) -> str:
    """Wrap user prompt for Mistral Instruct format."""
    return f"<s>[INST]{prompt}[/INST]"

def ai_extract_filters(user_text: str, brand_choices, city_choices):
    """Ask the model to return ONLY JSON describing search action + filters."""
    schema = f"""
You are an assistant inside a bike marketplace for Pakistan.

Extract filters from the user's message and return ONLY valid JSON with this shape:
{{
  "action": "search" | "general",
  "filters": {{
    "budget_max": int | null,
    "city": string | null,            // one of: {', '.join(city_choices) if city_choices else 'any'}
    "brand": string | null,           // one of: {', '.join(brand_choices) if brand_choices else 'any'}
    "cc_lo": int | null,
    "cc_hi": int | null,
    "condition": "New" | "Used" | null
  }},
  "answer": string
}}

Rules:
- If budget mentioned like 'under 2.5 lac', set budget_max=250000.
- If cc range '70-150', set cc_lo=70 and cc_hi=150.
- If unclear, leave fields null.
- Output JSON only, no extra words.

User: {user_text}
"""
    try:
        llm = get_llm()
        raw = llm(mistral_inst(schema), temperature=0.1, stream=False)
        text = str(raw).strip()
        start, end = text.find("{"), text.rfind("}") + 1
        if start == -1 or end <= start:
            return {"action": "general", "filters": {}, "answer": "Please share budget, city, brand/cc preference."}
        return json.loads(text[start:end])
    except Exception as e:
        return {"action": "general", "filters": {}, "answer": f"(AI error: {e})"}


# ------------------ DB Setup ------------------ #
def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS listings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, phone TEXT, city TEXT,
                brand TEXT, model TEXT, year INTEGER,
                cc INTEGER, type TEXT, condition TEXT,
                price INTEGER, mileage INTEGER, description TEXT, created_at TEXT
            )
        """)
        con.commit()

def table_empty():
    with get_conn() as con:
        cur = con.execute("SELECT COUNT(*) FROM listings")
        return cur.fetchone()[0] == 0

def seed_from_csv():
    if not CATALOG_CSV.exists():
        return
    df = pd.read_csv(CATALOG_CSV)
    if "name" not in df.columns: df["name"] = "Dealer"
    if "phone" not in df.columns: df["phone"] = "03xx-xxxxxxx"
    if "description" not in df.columns: df["description"] = "Seed listing"
    df["created_at"] = datetime.utcnow().isoformat()
    cols = ["name","phone","city","brand","model","year","cc","type","condition","price","mileage","description","created_at"]
    for c in cols:
        if c not in df.columns: df[c] = None
    with get_conn() as con:
        df[cols].to_sql("listings", con, if_exists="append", index=False)

def fetch_listings(filters=None):
    q, params = "SELECT * FROM listings WHERE 1=1", []
    if filters:
        if filters.get("brand"):
            q += " AND brand = ?"
            params.append(str(filters["brand"]).title())
        if filters.get("city"):
            q += " AND city = ?"
            params.append(str(filters["city"]).title())
        if filters.get("condition"):
            q += " AND condition = ?"
            params.append(str(filters["condition"]).title())
        if filters.get("cc_lo") is not None and filters.get("cc_hi") is not None:
            q += " AND cc BETWEEN ? AND ?"
            params += [int(filters["cc_lo"]), int(filters["cc_hi"])]
        if filters.get("budget_max") is not None:
            q += " AND price <= ?"
            params.append(int(filters["budget_max"]))
    with get_conn() as con:
        return pd.read_sql_query(q, con, params=params)

def insert_listing(rec: dict):
    with get_conn() as con:
        cols = ",".join(rec.keys())
        qmarks = ",".join(["?"] * len(rec))
        con.execute(f"INSERT INTO listings ({cols}) VALUES ({qmarks})", list(rec.values()))
        con.commit()

# ------------------ UI Helpers ------------------ #
def card(row):
    return f"""
<div style="border:1px solid #eee;border-radius:16px;padding:16px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div style="font-weight:700;font-size:18px">{row.brand} {row.model} ({int(row.year)})</div>
      <div style="opacity:0.8">{int(row.cc)}cc • {row.condition} • {row.type} • {int(row.mileage)} km</div>
      <div style="margin-top:6px;">📍 {row.city}</div>
      <div style="margin-top:6px;">📞 {row.phone}</div>
    </div>
    <div style="text-align:right;font-size:20px;font-weight:700">Rs {int(row.price):,}</div>
  </div>
  <div style="margin-top:8px">{row.description or ""}</div>
  <div style="margin-top:8px;opacity:0.6">Listed: {str(row.created_at).split("T")[0]}</div>
</div>
"""

def render_cards(df):
    for _, r in df.iterrows():
        st.markdown(card(r), unsafe_allow_html=True)

def sidebar_about():
    with st.sidebar:
        st.markdown("## BikeBot-AI 🏍️")
        st.caption("Offline AI-enabled bike marketplace (ctransformers + SQLite + Streamlit)")
        st.markdown("---")
        st.markdown("- **AI Chat**: search by talking\n- **Browse**: filters\n- **Sell**: add a listing")
        st.code(f"Model file: {MODEL_PATH}", language="bash")

# ------------------ Main App ------------------ #
def main():
    sidebar_about()
    st.title("🏍️ BikeBot-AI — Buy & Sell Bikes")

    # DB init + seed
    init_db()
    if table_empty():
        seed_from_csv()

    # Tabs
    tab_chat, tab_browse, tab_sell = st.tabs(["💬 AI Chat", "📋 Browse", "🛒 Sell"])

    # ---------- Chat Tab ---------- #
    with tab_chat:
        st.caption("Describe your need (budget, city, brand/cc, new/used). The AI extracts filters and shows matches.")
        all_df = fetch_listings({})
        brand_choices = sorted(all_df["brand"].dropna().unique().tolist()) if not all_df.empty else []
        city_choices = sorted(all_df["city"].dropna().unique().tolist()) if not all_df.empty else []

        if "ai_msgs" not in st.session_state:
            st.session_state.ai_msgs = [
                {"role":"assistant","content":"Hi! Tell me your budget, city, and any brand/cc preference."}
            ]

        for m in st.session_state.ai_msgs:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_input = st.chat_input("Type your message...")
        if user_input:
            st.session_state.ai_msgs.append({"role":"user","content":user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            ai = ai_extract_filters(user_input, brand_choices, city_choices)
            answer = ai.get("answer","")
            filters = ai.get("filters") or {}
            for k in ["city","brand","condition"]:
                if filters.get(k):
                    filters[k] = str(filters[k]).title()

            results = pd.DataFrame()
            if ai.get("action") == "search" or any(v is not None for v in filters.values()):
                results = fetch_listings(filters)
                if results.empty and filters.get("brand"):
                    # soften brand if too restrictive
                    f2 = dict(filters); f2["brand"] = None
                    results = fetch_listings(f2)

            with st.chat_message("assistant"):
                st.markdown(answer or "Here’s what I found.")
                if not results.empty:
                    st.markdown(f"**{len(results)}** match(es):")
                    render_cards(results.sort_values("price").head(8))
                else:
                    st.caption("No exact matches — try changing budget/brand/cc or ask to see all in a city.")

            st.session_state.ai_msgs.append({"role":"assistant","content":answer or "Done."})

        if st.button("🔄 Restart Chat"):
            st.session_state.ai_msgs = [
                {"role":"assistant","content":"Hi! Tell me your budget, city, and any brand/cc preference."}
            ]

    # ---------- Browse Tab ---------- #
    with tab_browse:
        st.subheader("Browse Listings")
        all_df = fetch_listings({})
        if all_df.empty:
            st.info("No listings yet. Add the first one in the Sell tab.")
        else:
            brands = sorted(all_df.brand.dropna().unique().tolist())
            cities = sorted(all_df.city.dropna().unique().tolist())
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                f_brands = st.multiselect("Brand", brands)
            with c2:
                f_cities = st.multiselect("City", cities)
            with c3:
                f_cond = st.selectbox("Condition", ["Any","New","Used"])
                f_cond = None if f_cond=="Any" else f_cond
            with c4:
                cc_lo, cc_hi = st.slider("CC Range", 50, 250, (50, 250), step=10)
            with c5:
                price_lo, price_hi = st.slider("Price (PKR)", 50000, 600000, (50000, 600000), step=5000)

            filters = {
                "brand": f_brands[0] if len(f_brands)==1 else None,
                "city": f_cities[0] if len(f_cities)==1 else None,
                "condition": f_cond,
                "cc_lo": cc_lo,
                "cc_hi": cc_hi,
                "budget_max": price_hi
            }
            df = fetch_listings(filters)
            st.caption(f"{len(df)} result(s)")
            if df.empty:
                st.info("No results. Adjust filters and try again.")
            else:
                render_cards(df.sort_values("price"))

    # ---------- Sell Tab ---------- #
    with tab_sell:
        st.subheader("Post Your Bike for Sale")
        with st.form("sell_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                name = st.text_input("Your Name")
                phone = st.text_input("Phone (e.g., 03xx-xxxxxxx)")
                city = st.text_input("City")
            with c2:
                brand = st.text_input("Brand", placeholder="Honda / Yamaha / Suzuki ...")
                model = st.text_input("Model", placeholder="CG 125 / YBR 125 ...")
                year = st.number_input("Year", min_value=2000, max_value=datetime.now().year, value=2022)
            with c3:
                cc = st.number_input("Engine CC", 50, 500, 125, step=5)
                btype = st.selectbox("Type", ["Commuter","Sport","Off-road","Other"])
                condition = st.selectbox("Condition", ["Used","New"])
            price = st.number_input("Price (PKR)", 10000, 2000000, 150000, step=5000)
            mileage = st.number_input("Mileage (km)", 0, 200000, 10000, step=500)
            description = st.text_area("Description", placeholder="Clean bike, single owner...")
            submitted = st.form_submit_button("Post Listing ✅")
        if submitted:
            rec = {
                "name": name.strip(),
                "phone": phone.strip(),
                "city": city.strip().title(),
                "brand": brand.strip().title(),
                "model": model.strip().title(),
                "year": int(year),
                "cc": int(cc),
                "type": btype,
                "condition": condition,
                "price": int(price),
                "mileage": int(mileage),
                "description": description.strip(),
                "created_at": datetime.utcnow().isoformat()
            }
            insert_listing(rec)
            st.success("Listing posted! Check it in Browse.")

if __name__ == "__main__":
    main()
