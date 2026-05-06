import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─── KONFIGURASI HALAMAN ───────────────────────────────────
st.set_page_config(
    page_title="GROMENT - Grosir Segmentation AI",
    page_icon="🛍️",
    layout="wide"
)

# ─── LOAD DATA & MODEL ─────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Wholesale customers data.csv")

df = load_data()
FITUR = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

if not os.path.exists("cluster3.joblib"):
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=2, random_state=42, n_init=10))
    ])
    model_pipeline.fit(df[FITUR].values)
    joblib.dump(model_pipeline, "cluster3.joblib")

model = joblib.load("cluster3.joblib")

# ─── CUSTOM CSS (REVISI: SOFT COLORS, SMALL FONTS) ─────────
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E2E8F0; font-family: 'Inter', sans-serif; }
    h1 { font-size: 1.5rem !important; color: #F472B6 !important; font-weight: 800; }
    h2 { font-size: 1.2rem !important; color: #F472B6 !important; font-weight: 700; margin-top: 1.5rem; }
    h3 { font-size: 1.0rem !important; color: #F472B6 !important; }
    p, li, label, div { font-size: 0.85rem !important; color: #CBD5E0; line-height: 1.5; }
    
    [data-testid="stSidebar"] { background-color: #161922; border-right: 1px solid rgba(244, 114, 182, 0.1); }
    
    .logo-container { text-align: center; padding: 10px 0 10px 0; }
    .logo-text-groment { font-family: 'Syne', sans-serif; font-size: 1.4rem !important; font-weight: 800; color: #F472B6; }

    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(244, 114, 182, 0.1);
        padding: 12px;
        padding-bottom: 30px;
        border-radius: 8px;
        height: 100%;
    }
    
    .notebook-cell {
        background-color: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #F472B6;
        padding: 15px;
        margin-bottom: 10px;
    }
    .code-output {
        background-color: #050505;
        padding: 8px;
        color: #93C5FD;
        font-size: 0.75rem !important;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR: LOGO & PETUNJUK ─────────────────────────────
with st.sidebar:
    
    st.markdown('<div style="font-size:3rem; text-align:center;">🛍️</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="logo-container"><span class="logo-text-groment">GROMENT</span></div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 Panduan Prediksi")
    st.write("""
    1. Masukkan angka pengeluaran tahunan pelanggan pada kolom di sebelah kanan.
    2. Gunakan angka positif tanpa koma.
    3. Klik **Analisis Sekarang** untuk melihat kategori pelanggan tersebut.
    """)
    st.markdown("---")
    st.caption("Developed by Sabdo Winarah")

# ─── MAIN CONTENT ──────────────────────────────────────────
tab_prediksi, tab_analisis, tab_kode, tab_about = st.tabs([
    "🔮 Prediksi", "📊 Analisis Data", "💻 Kode", "👤 About Me"
])


# ==========================================
# TAB 1: PREDIKSI
# ==========================================
with tab_prediksi:
    st.markdown("<h1>Prediksi Segmentasi Pelanggan</h1>",unsafe_allow_html=True)
    st.markdown("### 📝 Parameter Pengeluaran")
    c1, c2, c3 = st.columns(3)
    with c1:
        fresh = st.number_input("Fresh (Produk Segar)", 0, 150000, 10000)
        frozen = st.number_input("Frozen (Produk Beku)", 0, 150000, 3000)
    with c2:
        milk = st.number_input("Milk (Produk Susu)", 0, 150000, 5000)
        detergents = st.number_input("Detergents & Paper", 0, 150000, 2500)
    with c3:
        grocery = st.number_input("Grocery (Sembako)", 0, 150000, 8000)
        delicassen = st.number_input("Delicassen", 0, 150000, 1500)

    btn_analisis = st.button("🚀 Analisis Sekarang", use_container_width=True)

    if btn_analisis:
        input_data = np.array([[fresh, milk, grocery, frozen, detergents, delicassen]])
        prediction = model.predict(input_data)[0]
        
        info = {
            0: {"nama": "Restaurant / HoReCa", "desc": "Bisnis yang fokus pada bahan mentah segar."},
            1: {"nama": "Retail Store", "desc": "Bisnis yang fokus pada stok sembako dan kebutuhan rumah tangga."}
        }
        res = info[prediction]

        st.markdown(f"<div style='background:rgba(244,114,182,0.1); padding:15px; border-radius:10px; border:1px solid #F472B6;'><h2>Hasil: {res['nama']}</h2><p>{res['desc']}</p></div>", unsafe_allow_html=True)

        st.markdown("### 📈 Visualisasi Pendukung")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            # Radar Chart
            avg_df = df.copy()
            avg_df['Cluster'] = model.predict(df[FITUR].values)
            c_mean = avg_df[avg_df['Cluster'] == prediction][FITUR].mean().values
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=input_data[0], theta=FITUR, fill='toself', name='Data Anda', line_color='#F472B6'))
            fig_radar.add_trace(go.Scatterpolar(r=c_mean, theta=FITUR, fill='toself', name='Rata-rata Cluster', line_color='rgba(255,255,255,0.2)'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_radar, use_container_width=True)
        with col_v2:
            # Scatter Plot
            fig_pos = px.scatter(df, x="Fresh", y="Grocery", opacity=0.2, template="plotly_dark")
            fig_pos.add_trace(go.Scatter(x=[fresh], y=[grocery], mode='markers', marker=dict(size=12, color='#F472B6', symbol='x'), name='Posisi Anda'))
            fig_pos.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_pos, use_container_width=True)

# ==========================================
# TAB 2: ANALISIS DATA (UPDATE LENGKAP)
# ==========================================
with tab_analisis:
    st.markdown("<h1>mengenal Sumber Data</h1>",unsafe_allow_html=True)
    st.write(" mari kita pahami karakteristik data yang kita gunakan data ini diambil dai perilaku 440 pelanggan dari sebuat tempat di portugal dan model yang saya "
            "buat bisa membantu untuk memprediksi pelanggan termasuk tipe ***Retail***" atau pemilik ***Horeca*** dan membantu pengelola untuk mengatur stok di gudang.)
    
    # 1. Konteks Data
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Total Data", "440 Pelanggan")
    c_m2.metric("Kategori Produk", "6 Jenis")
    c_m3.metric("Wilayah Pengamatan", "Portugal")
    
    st.markdown("---")
    
    # 2. Terjemahan Fitur
    st.markdown("## 🔍 Apa Saja yang Kita Amati?")
    st.write("Berikut adalah 6 kategori produk utama yang menjadi dasar AI dalam mengelompokkan pelanggan:")
    
    f_c1, f_c2, f_c3 = st.columns(3)
    with f_c1:
        st.markdown('<div class="feature-card"><b>🥦 Fresh</b><br>Sayur-sayur an, buah-buah an, dan daging segar harian.</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        with f_c1: st.markdown('<div class="feature-card"><b>🧊 Frozen</b><br>Makanan beku seperti nugget,sosis,Bakso,Patty,Fish Roll,dll.</div>', unsafe_allow_html=True)
    with f_c2:
        st.markdown('<div class="feature-card"><b>🥛 Milk</b><br>Produk olahan susu seperti keju dan yogurt.</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        with f_c2: st.markdown('<div class="feature-card"><b>🧴 Detergents & Paper</b><br>Sabun, tisu, dan alat kebersihan.</div>', unsafe_allow_html=True)
    with f_c3:
        st.markdown('<div class="feature-card"><b>🛍️ Grocery</b><br>Sembako pokok seperti beras, minyak, dan tepung.</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        with f_c3: st.markdown('<div class="feature-card"><b>🧀 Delicassen</b><br>Daging olahan premium dan makanan siap saji.</div>', unsafe_allow_html=True)

    # 3. Tren Umum
    st.markdown("## 📈 Produk Mana yang Paling Banyak Dibeli?")
    st.write("Secara rata-rata, pelanggan menghabiskan uang paling banyak pada produk **Fresh** dan **Grocery**.")
    mean_all = df[FITUR].mean().sort_values(ascending=False)
    fig_bar_all = px.bar(mean_all, x=mean_all.index, y=mean_all.values, color=mean_all.values, color_continuous_scale="RdPu")
    fig_bar_all.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=20))
    st.plotly_chart(fig_bar_all, use_container_width=True)

    # 4. Hubungan Antar Produk (Heatmap + Story)
    st.markdown("## 🔥 Hubungan Antar Produk")
    col_h1, col_h2 = st.columns([1.5, 1])
    with col_h1:
        fig_heat = px.imshow(df[FITUR].corr(), text_auto=".2f", color_continuous_scale="RdPu", template="plotly_dark")
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=0, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)
    with col_h2:
        st.markdown("### 💡 Fakta Menarik")
        st.write("""
        Jika Anda melihat kotak yang sangat cerah antara **Grocery** dan **Detergents_Paper**, itu artinya:
        
        *Pelanggan yang membeli banyak sembako hampir pasti juga membeli banyak sabun dan tisu.*
        
        Hal ini sangat logis karena Toko Kelontong biasanya menyetok kedua barang ini secara bersamaan untuk dijual kembali.
        """)

    # 5. Mengapa AI Groment Dibutuhkan?
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(244,114,182,0.05); padding:20px; border-radius:10px; border-left:4px solid #F472B6;">
        <h3 style='margin:0;'>Mengapa AI Groment Dibutuhkan?</h3>
        <p>Melihat pola belanja yang sangat beragam di atas, mustahil bagi distributor untuk memberikan promo secara manual satu per satu. 
        Disinilah <b>AI Groment</b> masuk untuk mengelompokkan pelanggan ke dalam 'Cluster' secara otomatis, sehingga strategi pemasaran (diskon, bundling) bisa langsung tepat sasaran.</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# TAB 3: KODE
# ==========================================
with tab_kode:
    st.markdown("<h1>Dokumentasi Kode Model</h1>",unsafe_allow_html=True)
    steps = [
        {"title": "1. Import Library", "code": "import pandas as pd\nfrom sklearn.cluster import KMeans", "out": "Libraries Loaded."},
        {"title": "2. Preprocessing", "code": "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(df[FITUR])", "out": "Data Scaled."},
        {"title": "3. K-Means", "code": "model = KMeans(n_clusters=2, random_state=42)\nmodel.fit(X_scaled)", "out": "2 Clusters Formed."}
    ]
    for s in steps:
        st.markdown(f'<div class="notebook-cell"><h3>{s["title"]}</h3>', unsafe_allow_html=True)
        st.code(s["code"], language="python")
        st.markdown(f'<div class="code-output">{s["out"]}</div></div>', unsafe_allow_html=True)

# ==========================================
# TAB 4: ABOUT ME
# ==========================================
with tab_about:
    c_m1, c_m2 = st.columns([1, 4])
    with c_m1:
        st.image("https://via.placeholder.com/150x150.png?text=Sabdo", width=120)
    with c_m2:
        st.markdown('<div style="background:rgba(244,114,182,0.05); padding:20px; border-radius:10px; border:1px solid rgba(244,114,182,0.2);"><h2>Sabdo Winarah</h2><p>Pelajar / ML Developer. Fokus pada solusi segmentasi cerdas.</p></div>', unsafe_allow_html=True)
