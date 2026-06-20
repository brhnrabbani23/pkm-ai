import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from groq import Groq

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="GreenEconomy-AI", page_icon="\U0001F33F", layout="wide")

# Catatan: warna utama (background, sidebar, teks, tombol) diatur lewat
# .streamlit/config.toml supaya SEMUA komponen bawaan Streamlit (dropdown,
# input angka, kotak chat, dsb) ikut tema hijau secara konsisten -- bukan
# cuma elemen yang di-custom lewat CSS. Ini yang bikin tampilan sebelumnya
# kelihatan belang (sebagian hijau terang, sebagian gelap bawaan browser).

# ==========================================
# 2. CSS -- SENTUHAN VISUAL "GREENECONOMY-AI"
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');

    :root {
        --ge-forest: #15331F;
        --ge-leaf: #2E9B5C;
        --ge-leaf-deep: #1F7A47;
        --ge-sky: #2F8FBF;
        --ge-card: #FFFFFF;
        --ge-card-border: #BFE3CC;
    }

    /* --- Elemen bawaan Streamlit --- */
    #MainMenu {visibility: visible;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header[data-testid="stHeader"] { background-color: transparent; }
    div[data-testid="stDecoration"] { visibility: hidden; }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1rem !important;
    }

    /* --- Tipografi --- */
    h1, h2, h3 {
        font-family: 'Fraunces', serif !important;
        color: var(--ge-forest) !important;
        letter-spacing: -0.01em;
    }
    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    /* --- Header / Logo "GreenEconomy-AI" (tanpa daun besar di pojok) --- */
    .ge-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 4px 0 18px 0;
        border-bottom: 1px solid var(--ge-card-border);
        margin-bottom: 18px;
    }
    .ge-logo-badge { flex-shrink: 0; }
    .ge-wordmark { display: flex; flex-direction: column; line-height: 1.15; }
    .ge-name {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.75rem;
        color: var(--ge-forest);
    }
    .ge-name .ge-ai-tag {
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        color: #FFFFFF;
        background: linear-gradient(135deg, var(--ge-sky), #1F6F94);
        padding: 2px 8px;
        border-radius: 6px;
        margin-left: 8px;
        vertical-align: middle;
        letter-spacing: 0.03em;
    }
    .ge-tagline {
        font-family: 'Manrope', sans-serif;
        font-size: 0.82rem;
        color: var(--ge-forest);
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 3px;
    }

    /* --- Brand mark mini di sidebar --- */
    .ge-sidebar-brand {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.05rem;
        color: var(--ge-forest);
        margin-bottom: 2px;
    }

    /* --- Kartu konten: putih solid supaya kontras & rapi di atas latar hijau --- */
    .adaptive-box {
        background-color: var(--ge-card);
        color: var(--ge-forest);
        padding: 16px 18px;
        border-radius: 12px;
        border-left: 5px solid var(--ge-leaf);
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(21,51,31,0.08);
    }
    .info-box {
        background-color: var(--ge-card);
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid var(--ge-card-border);
        border-left: 4px solid var(--ge-sky);
        margin-top: 15px;
        font-size: 0.9rem;
        color: var(--ge-forest);
    }

    /* --- Tombol utama --- */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--ge-leaf), var(--ge-leaf-deep));
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }

    /* --- Metric --- */
    [data-testid="stMetricValue"] {
        color: var(--ge-leaf-deep);
        font-family: 'Fraunces', serif;
    }

    /* --- Animasi --- */
    .fade-in { animation: fadeIn 1.0s ease-in-out; }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(8px);}
        to {opacity: 1; transform: translateY(0);}
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD DATA & LOAD MODEL (.PKL)
# ==========================================
@st.cache_resource
def load_data_and_model():
    # --- A. LOAD DATASET BUAT SCALER ---
    try:
        df = pd.read_csv('dataset worldbank PKM fixed.csv', sep=';')
    except FileNotFoundError:
        st.error("File CSV gak ketemu! Pastikan 'dataset worldbank PKM fixed.csv' ada.")
        st.stop()

    # Cleaning Data (Tetap diperlukan untuk fit Scaler)
    def bersihin_angka(nilai, is_target=False):
        val = str(nilai).strip()
        if val in ['..', 'nan', '', 'NaN']: return np.nan
        val = val.replace(',', '')
        if is_target:
            try: return float(val)
            except: return np.nan
        else:
            val_clean = re.sub(r'\.', '', val)
            try:
                angka = float(val_clean)
                if angka > 100000000000: angka = angka / 1e12
                return angka
            except: return np.nan

    col_pop = 'Population total'
    col_gdp = 'GDP per capita (current US$)'
    col_energy = 'Energy use (kg of oil equivalent per capita)'
    col_target = 'Renewable energy consumption (% of total final energy consumption)'

    if 'Country Name' in df.columns: df['Country Name'] = df['Country Name'].ffill()

    if col_pop in df.columns: df[col_pop] = df[col_pop].apply(lambda x: bersihin_angka(x, False))
    if col_gdp in df.columns: df[col_gdp] = df[col_gdp].apply(lambda x: bersihin_angka(x, False))
    if col_energy in df.columns: df[col_energy] = df[col_energy].apply(lambda x: bersihin_angka(x, False))
    if col_target in df.columns: df[col_target] = df[col_target].apply(lambda x: bersihin_angka(x, True))

    df_model = df.dropna(subset=[col_pop, col_gdp, col_energy, col_target])

    X = df_model[[col_pop, col_gdp, col_energy]]
    y = df_model[col_target]

    # --- B. FIT SCALER (PENTING!) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- C. LOAD MODEL PRE-TRAINED ---
    model = None
    try:
        with open('model_xgboost.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.warning("File 'model_xgboost.pkl' gak ketemu! Melatih model baru secara otomatis...")
        model = XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=7)
        model.fit(X_scaled, y)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    return model, scaler, df, df_model

try:
    model, scaler, df_full, df_clean = load_data_and_model()
except Exception as e:
    st.error(f"Error sistem saat load data: {e}")
    st.stop()

# ==========================================
# 4. SIDEBAR (PANEL KONTROL)
# ==========================================
st.sidebar.markdown('<div class="ge-sidebar-brand">\U0001F33F GreenEconomy-AI</div>', unsafe_allow_html=True)
st.sidebar.header("\U0001F39B\uFE0F Panel Kontrol")

# --- PILIH NEGARA ---
st.sidebar.subheader("1. Pilih Negara")
list_negara = sorted(df_full['Country Name'].dropna().unique())
CUSTOM_OPTION = "--- Custom (Input Bebas) ---"
list_negara.insert(0, CUSTOM_OPTION)
idx_default = list_negara.index('Indonesia') if 'Indonesia' in list_negara else 0
selected_option = st.sidebar.selectbox("Cari negara / Mode Custom:", list_negara, index=idx_default)

# --- FUNGSI HELPER ---
def get_display_val(val, is_target=False):
    if pd.isna(val) or str(val).strip() == '..': return 0.0
    val_str = str(val).replace(',', '')
    if not is_target: val_str = val_str.replace('.', '')
    try:
        angka = float(val_str)
        if not is_target and angka > 100000000000: angka = angka / 1e12
        return angka
    except: return 0.0

if selected_option == CUSTOM_OPTION:
    def_pop, def_gdp, def_energy = 1000000.0, 1000.0, 500.0
    display_name = "Skenario Kustom"
else:
    country_data = df_full[df_full['Country Name'] == selected_option].iloc[-1]
    def_pop = get_display_val(country_data['Population total'])
    def_gdp = get_display_val(country_data['GDP per capita (current US$)'])
    def_energy = get_display_val(country_data['Energy use (kg of oil equivalent per capita)'])
    display_name = selected_option

st.sidebar.markdown("---")
st.sidebar.subheader("2. Ubah Indikator")

pop_input = st.sidebar.number_input("Populasi", min_value=0.0, value=def_pop, step=100000.0, format="%f")
gdp_input = st.sidebar.number_input("GDP per Kapita (USD)", min_value=0.0, value=def_gdp, step=100.0, format="%.2f")
energy_input = st.sidebar.number_input("Energy Use (kg of oil/kapita)", min_value=0.0, value=def_energy, step=10.0, format="%.2f")

st.sidebar.markdown("---")
btn_predict = st.sidebar.button("\U0001F680 PREDIKSI SKENARIO", type="primary")

# ==========================================
# 5. LOGIKA PROSES & AI
# ==========================================
if "prediction_state" not in st.session_state:
    st.session_state.prediction_state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if btn_predict:
    input_data = pd.DataFrame([[pop_input, gdp_input, energy_input]],
                              columns=['Population total', 'GDP per capita (current US$)', 'Energy use (kg of oil equivalent per capita)'])
    input_scaled = scaler.transform(input_data)
    raw_prediction = model.predict(input_scaled)[0]
    result = max(0.0, min(100.0, float(raw_prediction)))

    ai_analysis = ""
    try:
        if "GROQ_API_KEY" in st.secrets:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            negara_text = display_name if selected_option != CUSTOM_OPTION else "sebuah Negara Hipotetis"
            prompt_text = f"""
            Analisis untuk {negara_text}:
            - Populasi: {pop_input}
            - GDP: US$ {gdp_input}
            - Energy Use: {energy_input} kg of oil equivalent per capita
            Prediksi: {result:.2f}% energi terbarukan.
            Berikan: 1. Analisis Singkat, 2. 3 Rekomendasi Kebijakan Konkret.
            """
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_text}],
                model="llama-3.3-70b-versatile",
            )
            ai_analysis = chat_completion.choices[0].message.content
        else:
            ai_analysis = "API Key Groq tidak ditemukan di st.secrets."

    except Exception as e:
        ai_analysis = f"Gagal mengambil analisis AI: {e}"

    st.session_state.prediction_state = {
        "result": result,
        "pop": pop_input,
        "gdp": gdp_input,
        "energy": energy_input,
        "name": display_name,
        "analysis": ai_analysis
    }
    st.session_state.chat_history = []

# ==========================================
# 6. DASHBOARD UTAMA
# ==========================================
st.markdown("""
<div class="ge-header">
    <div class="ge-logo-badge">
        <svg width="54" height="54" viewBox="0 0 54 54" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="geRing" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#3FB873"/>
                    <stop offset="100%" stop-color="#1F7A47"/>
                </linearGradient>
            </defs>
            <circle cx="27" cy="27" r="26" fill="url(#geRing)"/>
            <circle cx="27" cy="27" r="26" fill="none" stroke="#163C24" stroke-opacity="0.15" stroke-width="1"/>
            <path d="M27 13 C36 17 41 26 36 35 C32 41 22 42 17 36 C13 31 14 21 21 16 C23 14.5 25 13.6 27 13 Z" fill="#FFFFFF"/>
            <path d="M27 13 C23 21 20 28 19 36" stroke="#3FB873" stroke-width="1.6" fill="none" stroke-linecap="round"/>
            <circle cx="19" cy="36" r="2" fill="#2F8FBF"/>
        </svg>
    </div>
    <div class="ge-wordmark">
        <span class="ge-name">GreenEconomy<span class="ge-ai-tag">AI</span></span>
        <span class="ge-tagline">Prediksi &amp; Strategi Transisi Energi Berkelanjutan</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"Analisis untuk **{display_name}**.")

st.markdown("""
    <div class="adaptive-box">
        <b>Catatan Analitis:</b> Hasil prediksi ini mengevaluasi dampak indikator ekonomi terhadap strategi transisi energi.
    </div>
    """, unsafe_allow_html=True)

# TAMPILKAN HASIL
if st.session_state.prediction_state:
    data = st.session_state.prediction_state

    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.info("Hasil Prediksi Kuantitatif")

        fig = go.Figure(data=[go.Pie(
            values=[data['result'], 100 - data['result']],
            labels=['Energi Terbarukan', 'Energi Non-Terbarukan (Fosil)'],
            hole=0.65,
            marker=dict(colors=['#2E9B5C', '#163C24']),
            textinfo='percent',
            textfont=dict(size=14),
            hoverinfo='label+percent'
        )])
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=10, b=0, l=0, r=0),
            height=280,
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric(label="Persentase Adopsi", value=f"{data['result']:.2f}%")

        if data['result'] < 20: st.error("Status: RENDAH")
        elif data['result'] < 50: st.warning("Status: MENENGAH")
        else: st.success("Status: TINGGI")

        st.markdown("""
        <div class="info-box">
        <b>Penjelasan Konsep:</b><br>
        Prediksi <b>% energi terbarukan</b> menunjukkan potensi energi bersih berdasarkan kondisi ekonomi
                    serta menggambarkan tingkat kesiapan negara tersebut dalam melakukan transisi menuju ekonomi hijau.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.success(f"Analisis Kebijakan ({data['name']})")
        st.markdown(data['analysis'])
        st.markdown("---")
        st.caption("Parameter Input:")
        st.code(f"Populasi: {data['pop']:,.0f} | GDP: US$ {data['gdp']:,.2f} | Energy: {data['energy']:,.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 7. FITUR CHAT LANJUTAN
    # ==========================================
    st.markdown("---")
    st.subheader("Asisten Kebijakan Interaktif")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanyakan detail kebijakan..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        context_prompt = f"""
        Konteks: Negara {data['name']}, Populasi {data['pop']}, GDP {data['gdp']}, Energy Use {data['energy']}.
        Prediksi Renewable Energy: {data['result']:.2f}%.
        Pertanyaan User: {prompt}. Jawab ringkas & solutif.
        """

        with st.chat_message("assistant"):
            msg_ph = st.empty()
            try:
                if "GROQ_API_KEY" in st.secrets:
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                    res = client.chat.completions.create(
                        messages=[{"role": "user", "content": context_prompt}],
                        model="llama-3.3-70b-versatile",
                    )
                    full_res = res.choices[0].message.content
                    msg_ph.markdown(full_res)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_res})
                else:
                    msg_ph.error("API Key Groq hilang.")
            except Exception as e:
                msg_ph.error(f"Error: {e}")

else:
    st.info("Pilih negara dan klik tombol 'PREDIKSI SKENARIO' untuk mulai.")
