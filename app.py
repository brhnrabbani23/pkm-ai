import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import pickle  # <--- TAMBAHAN PENTING
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from groq import Groq

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="TIM BEJO", page_icon="🌿", layout="wide")

# ==========================================
# 2. CSS "SAPU BERSIH" & TEMA ADAPTIF
# ==========================================
st.markdown("""
<style>
    /* 1. ATUR ELEMENT BAWAAN STREAMLIT */
    #MainMenu {visibility: visible;} 
    footer {visibility: hidden;} 
    
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    div[data-testid="stDecoration"] {
        visibility: hidden;
    }

    /* 2. ATUR PADDING */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }

    /* 3. SEMBUNYIKAN TOMBOL DEPLOY */
    .stDeployButton {display:none;}
    
    /* 4. KOTAK ADAPTIF */
    .adaptive-box {
        background-color: var(--secondary-background-color); 
        color: var(--text-color);
        padding: 16px; 
        border-radius: 10px; 
        border-left: 5px solid #22c55e; 
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background-color: var(--secondary-background-color); 
        padding: 14px; 
        border-radius: 10px; 
        border-left: 4px solid #38bdf8; 
        margin-top: 15px;
        font-size: 0.9rem;
        color: var(--text-color);
    }
    
    /* 5. ANIMASI */
    .fade-in {
        animation: fadeIn 1.2s ease-in-out;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
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
        st.error("❌ File CSV gak ketemu! Pastikan 'dataset worldbank PKM fixed.csv' ada.")
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
    # Kita harus tetap bikin scaler dari data asli biar input user nanti skalanya pas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- C. LOAD MODEL PRE-TRAINED ---
    model = None
    try:
        with open('model_xgboost.pkl', 'rb') as file:
            model = pickle.load(file)
        # st.toast("✅ Model Pre-trained berhasil dimuat!") # Optional: notif kecil
    except FileNotFoundError:
        st.warning("⚠️ File 'model_xgboost.pkl' gak ketemu! Melatih model baru secara otomatis...")
        # Fallback: Kalau file pkl gak ada, latih ulang (biar gak error)
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
st.sidebar.header("🎛️ Panel Kontrol")

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

pop_input = st.sidebar.number_input("👥 Populasi", min_value=0.0, value=def_pop, step=100000.0, format="%f")
gdp_input = st.sidebar.number_input("💰 GDP per Kapita (USD)", min_value=0.0, value=def_gdp, step=100.0, format="%.2f")
energy_input = st.sidebar.number_input("⚡ Energy Use (kg of oil/kapita)", min_value=0.0, value=def_energy, step=10.0, format="%.2f")

st.sidebar.markdown("---")
btn_predict = st.sidebar.button("🚀 PREDIKSI SKENARIO", type="primary")

# ==========================================
# 5. LOGIKA PROSES & AI
# ==========================================
if "prediction_state" not in st.session_state:
    st.session_state.prediction_state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if btn_predict:
    # 1. Prediksi XGBoost (Pake Model .pkl tadi)
    input_data = pd.DataFrame([[pop_input, gdp_input, energy_input]], 
                              columns=['Population total', 'GDP per capita (current US$)', 'Energy use (kg of oil equivalent per capita)'])
    input_scaled = scaler.transform(input_data)
    raw_prediction = model.predict(input_scaled)[0]
    result = max(0.0, min(100.0, float(raw_prediction)))

    # 2. Analisis Awal AI
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
            ai_analysis = "⚠️ API Key Groq tidak ditemukan di st.secrets."
            
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
st.title("🌍 Sistem Cerdas Prediksi dan Analisis Kebijakan Transisi Energi Berbasis Data dengan TIM BEJO")
st.markdown(f"Analisis untuk **{display_name}**.")

st.markdown("""
    <div class="adaptive-box">
        <b>📌 Catatan Analitis:</b> Hasil prediksi ini mengevaluasi dampak indikator ekonomi terhadap strategi transisi energi.
    </div>
    """, unsafe_allow_html=True)

# TAMPILKAN HASIL
if st.session_state.prediction_state:
    data = st.session_state.prediction_state
    
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.info("📊 Hasil Prediksi Kuantitatif")
        
        fig = go.Figure(data=[go.Pie(
            values=[data['result'], 100 - data['result']],
            labels=['Energi Terbarukan', 'Energi Non-Terbarukan (Fosil)'],
            hole=0.65, 
            marker=dict(colors=['#22c55e', '#334155']),
            textinfo='percent',
            textfont=dict(size=14),
            hoverinfo='label+percent'
        )])
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=0, b=0, l=0, r=0),
            height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric(label="Persentase Adopsi", value=f"{data['result']:.2f}%")
        
        if data['result'] < 20: st.error("Status: RENDAH")
        elif data['result'] < 50: st.warning("Status: MENENGAH")
        else: st.success("Status: TINGGI")
        
        st.markdown("""
        <div class="info-box">
        <b>ℹ️ Penjelasan Konsep:</b><br>
        Prediksi <b>% energi terbarukan</b> menunjukkan potensi energi bersih berdasarkan kondisi ekonomi 
                    serta menggambarkan tingkat kesiapan negara tersebut dalam melakukan transisi menuju ekonomi hijau.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.success(f"🤖 Analisis Kebijakan ({data['name']})")
        st.markdown(data['analysis'])
        st.markdown("---")
        st.caption("🔍 Parameter Input:")
        st.code(f"Populasi: {data['pop']:,.0f} | GDP: US$ {data['gdp']:,.2f} | Energy: {data['energy']:,.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 7. FITUR CHAT LANJUTAN
    # ==========================================
    st.markdown("---")
    st.subheader("💬 Asisten Kebijakan Interaktif")
    
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
    st.info("👈 Pilih negara dan klik tombol 'PREDIKSI SKENARIO' untuk mulai.")
    