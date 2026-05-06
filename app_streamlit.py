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
    page_title="SEGMENTAI - Customer Insights",
    page_icon="📊",
    layout="wide"
)

# ─── LOAD DATA & MODEL ─────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Wholesale customers data.csv")

df = load_data()
FITUR = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

# Load model pipeline
if not os.path.exists("cluster3.joblib"):
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=2, random_state=42, n_init=10))
    ])
    model_pipeline.fit(df[FITUR].values)
    joblib.dump(model_pipeline, "cluster3.joblib")

model = joblib.load("cluster3.joblib")

# ─── CUSTOM CSS (REVISI: SOFT COLORS & FONT SIZES) ─────────
st.markdown("""
<style>
    /* Main Theme: Dark Slate & Soft Pink */
    .stApp { 
        background-color: #0E1117; 
        color: #E2E8F0; 
        font-family: 'Inter', sans-serif;
    }
    
    /* Font Size Adjustments */
    h1 { font-size: 1.8rem !important; color: #F472B6 !important; font-weight: 800; }
    h2 { font-size: 1.3rem !important; color: #F472B6 !important; font-weight: 700; }
    h3 { font-size: 1.1rem !important; color: #F472B6 !important; }
    p, li, label, div { font-size: 0.9rem !important; color: #CBD5E0; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { 
        background-color: #1A1C24; 
        border-right: 1px solid rgba(244, 114, 182, 0.2); 
    }
    
    /* Notebook Style Cells */
    .notebook-cell {
        background-color: rgba(244, 114, 182, 0.03);
        border-left: 4px solid #F472B6;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    .code-output {
        background-color: #050505;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 8px;
        border-radius: 4px;
        color: #93C5FD;
        font-size: 0.8rem !important;
    }

    /* Result Card */
    .result-card {
        background: rgba(244, 114, 182, 0.1);
        border: 1px solid #F472B6;
        padding: 20px;
        border-radius: 12px;
    }

    /* Logo Sidebar */
    .logo-text {
        font-size: 1.5rem !important;
        font-weight: 800;
        color: #F472B6;
        text-align: center;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(244, 114, 182, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR: LOGO & PETUNJUK ─────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-text">SEGMENTAI</div>', unsafe_allow_html=True)
    st.markdown("### 📖 Panduan Pengisian")
    st.write("""
    **Ikuti langkah berikut:**
    1. Masukkan total pengeluaran tahunan pelanggan pada tiap kolom input.
    2. Pastikan nominal dalam angka positif (satuan mata uang).
    3. Klik tombol **'Analisis'** untuk memproses data menggunakan model K-Means.
    4. Hasil akan menampilkan label cluster serta perbandingannya dengan data historis.
    """)
    st.markdown("---")
    st.caption("Developed by Sabdo Winarah")

# ─── MAIN CONTENT: TABS ───────────────────────────────────
tab_prediksi, tab_analisis, tab_kode, tab_about = st.tabs([
    "🔮 Prediksi", "📊 Analisis Data", "💻 Kode", "👤 About Me"
])

# ==========================================
# TAB 1: PREDIKSI (HALAMAN UTAMA)
# ==========================================
with tab_prediksi:
    st.markdown("<h1>Prediksi Segmentasi Pelanggan</h1>", unsafe_allow_html=True)
    st.markdown("<p>Sistem ini mengelompokkan pelanggan berdasarkan pola belanja tahunan mereka.</p>", unsafe_allow_html=True)

    # Input Grid
    st.markdown("### 📝 Masukkan Data Pengeluaran")
    c1, c2, c3 = st.columns(3)
    with c1:
        fresh = st.number_input("Fresh (Segar)", 0, 150000, 12000)
        frozen = st.number_input("Frozen (Beku)", 0, 150000, 2000)
    with c2:
        milk = st.number_input("Milk (Susu)", 0, 150000, 9000)
        detergents = st.number_input("Detergents & Paper", 0, 150000, 3000)
    with c3:
        grocery = st.number_input("Grocery (Sembako)", 0, 150000, 7000)
        delicassen = st.number_input("Delicassen", 0, 150000, 1500)

    btn_analisis = st.button("🚀 Analisis Cluster Sekarang", use_container_width=True)

    if btn_analisis:
        input_data = np.array([[fresh, milk, grocery, frozen, detergents, delicassen]])
        prediction = model.predict(input_data)[0]
        
        cluster_info = {
            0: {"nama": "Restaurant / HoReCa", "color": "#10B981", "desc": "Didominasi pengeluaran produk segar."},
            1: {"nama": "Retail Store", "color": "#3B82F6", "desc": "Didominasi produk Grocery dan Detergents."}
        }
        res = cluster_info[prediction]

        st.markdown("---")
        
        # Header Hasil
        st.markdown(f"""
        <div class="result-card">
            <h2 style='margin:0;'>Hasil Prediksi: {res['nama']}</h2>
            <p style='margin-top:5px;'>{res['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Visualisasi Tambahan (Row 1)
        st.markdown("### 📈 Visualisasi Pendukung")
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            # 1. RADAR CHART
            st.markdown("#### Perbandingan Pola Belanja")
            avg_val = df.copy()
            avg_val['Cluster'] = model.predict(df[FITUR].values)
            cluster_avg = avg_val[avg_val['Cluster'] == prediction][FITUR].mean().values
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=input_data[0], theta=FITUR, fill='toself', name='Input Anda', line_color='#F472B6'))
            fig_radar.add_trace(go.Scatterpolar(r=cluster_avg, theta=FITUR, fill='toself', name='Rata-rata Cluster', line_color='rgba(255,255,255,0.2)'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=30, b=30))
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_v2:
            # 2. SCATTER PLOT (POSISI DATA)
            st.markdown("#### Posisi Anda dalam Distribusi")
            fig_scatter = px.scatter(df, x="Fresh", y="Grocery", opacity=0.3, template="plotly_dark", color_discrete_sequence=["#4A5568"])
            fig_scatter.add_trace(go.Scatter(x=[fresh], y=[grocery], mode='markers', marker=dict(size=15, color='#F472B6', symbol='star'), name='Lokasi Anda'))
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=30, b=30))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Visualisasi Tambahan (Row 2)
        st.markdown("#### Perbandingan Terhadap Rata-rata Global")
        global_avg = df[FITUR].mean()
        compare_df = pd.DataFrame({
            'Kategori': FITUR,
            'Input Anda': input_data[0],
            'Rata-rata Semua': global_avg.values
        })
        fig_bar = px.bar(compare_df, x='Kategori', y=['Input Anda', 'Rata-rata Semua'], barmode='group', color_discrete_map={'Input Anda': '#F472B6', 'Rata-rata Semua': '#4A5568'})
        fig_bar.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 2: ANALISIS DATA
# ==========================================
with tab_analisis:
    st.markdown("<h1>Analisis Karakteristik Pelanggan</h1>", unsafe_allow_html=True)
    st.write("Eksplorasi data historis untuk memahami perbedaan antar segmen bisnis.")
    
    # Heatmap
    st.markdown("### 🔥 Korelasi Antar Produk")
    st.write("Warna yang lebih cerah menunjukkan bahwa pelanggan yang membeli produk A cenderung membeli produk B.")
    fig_heat = px.imshow(df[FITUR].corr(), text_auto=".2f", color_continuous_scale="RdPu", template="plotly_dark")
    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# TAB 3: KODE (NOTEBOOK STYLE)
# ==========================================
with tab_kode:
    st.markdown("<h1>Dokumentasi Pengembangan Model</h1>", unsafe_allow_html=True)
    
    st.markdown('<div class="notebook-cell">', unsafe_allow_html=True)
    st.markdown("### 1. Inisialisasi Data")
    st.code("df = pd.read_csv('Wholesale customers data.csv')\nprint(df.head())", language="python")
    st.markdown('<div class="code-output">Output: DataFrame loaded with 440 rows.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="notebook-cell">', unsafe_allow_html=True)
    st.markdown("### 2. Training Model K-Means")
    st.code("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)""", language="python")
    st.markdown('<div class="code-output">Model trained. Optimal K selected: 2.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 4: ABOUT ME
# ==========================================
with tab_about:
    c_me1, c_me2 = st.columns([1, 3])
    with c_me1:
        st.image("https://via.placeholder.com/200x200.png?text=Sabdo+Winarah", width=180)
    with c_me2:
        st.markdown(f"""
        <div style="background:rgba(244, 114, 182, 0.1); padding:25px; border-radius:15px; border: 1px solid rgba(244, 114, 182, 0.3);">
            <h2 style='margin:0; color:#F472B6 !important;'>Sabdo Winarah</h2>
            <p><strong>Status:</strong> Pelajar / Machine Learning Developer</p>
            <p><strong>Bahasa:</strong> Python, JavaScript, HTML/CSS</p>
            <hr style='border-color:rgba(244, 114, 182, 0.2)'>
            <p>Halo! Saya adalah seorang pelajar yang tertarik pada bidang Data Science. Project ini mengeksplorasi penggunaan K-Means Clustering untuk segmentasi bisnis grosir.</p>
        </div>
        """, unsafe_allow_html=True)
