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

# ─── Konfigurasi Dasar & Load Model ───────────────────────
if not os.path.exists("cluster3.joblib"):
    df_train = pd.read_csv("Wholesale customers data.csv")
    FITUR_TRAIN = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
    X = df_train[FITUR_TRAIN].values
    model_baru = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=2, random_state=42, n_init=10))
    ])
    model_baru.fit(X)
    joblib.dump(model_baru, "cluster3.joblib")

model = joblib.load("cluster3.joblib")
df = pd.read_csv("Wholesale customers data.csv")
FITUR = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

st.set_page_config(page_title="Customer Segmentation AI", page_icon="📊", layout="wide")

# ─── Custom CSS (Tema Navy & Pink) ────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #00033D; color: #FFCCF2; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #001550;
        border-right: 2px solid #FF007F;
    }
    
    /* Logo Styling */
    .sidebar-logo {
        text-align: center;
        padding: 20px 0;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.8rem;
        color: #FF007F;
        border-bottom: 1px solid rgba(255, 0, 127, 0.3);
        margin-bottom: 20px;
    }

    /* Notebook Style Cells */
    .notebook-cell {
        background-color: rgba(255, 0, 127, 0.05);
        border-left: 5px solid #FF007F;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 4px 4px 15px rgba(0,0,0,0.3);
    }

    /* Profile Card */
    .profile-card {
        background: #FFCCF2;
        color: #00033D;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
    }
    
    /* Utility */
    h1, h2, h3 { color: #FF007F !important; font-family: 'Syne', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #FFCCF2;
    }
    .stTabs [aria-selected="true"] { background-color: #FF007F !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR: Logo & Petunjuk ─────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">SEGMENTAI</div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 Petunjuk Penggunaan")
    st.info("""
    1. **Pilih Tab Prediksi** untuk mulai.
    2. **Input Pengeluaran**: Masukkan angka pengeluaran tahunan pelanggan untuk setiap kategori produk (Fresh, Milk, dll).
    3. **Klik Tombol Prediksi**: Sistem akan menganalisis cluster pelanggan tersebut.
    4. **Lihat Analisis**: Hasil akan menampilkan tipe pelanggan beserta grafik pendukung.
    """)
    
    st.markdown("---")
    st.caption("Data Source: UCI Machine Learning Repository")

# ─── MAIN CONTENT: Tabs ───────────────────────────────────
tab_prediksi, tab_analisis, tab_about = st.tabs(["🔮 Prediksi", "📊 Analisis Data", "👤 About Me"])

# ==========================================
# TAB 1: PREDIKSI (Halaman Utama)
# ==========================================
with tab_prediksi:
    st.markdown("## Prediksi Segmentasi Pelanggan")
    st.write("Gunakan form di bawah untuk memprediksi kategori pelanggan baru.")
    
    col_input, col_result = st.columns([1, 1.2])
    
    with col_input:
        st.markdown('<div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius:15px;">', unsafe_allow_html=True)
        f1 = st.number_input("🥦 Fresh (Produk Segar)", 0, 100000, 5000)
        f2 = st.number_input("🥛 Milk (Produk Susu)", 0, 100000, 5000)
        f3 = st.number_input("🛍️ Grocery (Kebutuhan Pokok)", 0, 100000, 5000)
        f4 = st.number_input("🧊 Frozen (Produk Beku)", 0, 100000, 5000)
        f5 = st.number_input("🧴 Detergents & Paper", 0, 100000, 5000)
        f6 = st.number_input("🧀 Delicassen (Makanan Spesial)", 0, 100000, 5000)
        
        btn_prediksi = st.button("Analisis Sekarang", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if btn_prediksi:
            data_in = np.array([[f1, f2, f3, f4, f5, f6]])
            res = model.predict(data_in)[0]
            
            # Info Cluster
            cls_map = {
                0: {"nama": "Restaurant / HoReCa", "desc": "Fokus pada produk Fresh dan Frozen.", "recom": "Tawarkan paket bahan baku segar harian."},
                1: {"nama": "Retail Store", "desc": "Fokus pada Grocery dan Detergents.", "recom": "Tawarkan diskon bundling produk kebutuhan rumah tangga."}
            }
            
            st.success(f"### Hasil: {cls_map[res]['nama']}")
            st.write(cls_map[res]['desc'])
            
            # Grafik Pendukung (Radar Chart)
            st.markdown("#### Perbandingan Data Input vs Rata-rata Cluster")
            avg_cluster = df[model.predict(df[FITUR].values) == res][FITUR].mean().values
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=data_in[0], theta=FITUR, fill='toself', name='Input Kamu', line_color='#FF007F'))
            fig.add_trace(go.Scatterpolar(r=avg_cluster, theta=FITUR, fill='toself', name='Rata-rata Cluster', line_color='rgba(255,255,255,0.5)'))
            
            fig.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark", 
                              paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(l=40, r=40, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 **Rekomendasi Strategi:** {cls_map[res]['recom']}")
        else:
            st.info("Silakan isi data di sebelah kiri dan klik tombol prediksi.")

# ==========================================
# TAB 2: ANALISIS DATA (Dataset + Code)
# ==========================================
with tab_analisis:
    st.markdown("## Eksplorasi & Analisis Data")
    
    # Bagian Dataset
    with st.expander("📂 Informasi Dataset (UCI Machine Learning Repository)", expanded=True):
        st.write("Dataset ini berisi data transaksi tahunan dari 440 pelanggan distributor grosir.")
        st.dataframe(df.head(), use_container_width=True)
        st.caption("Sumber: https://archive.ics.uci.edu/ml/datasets/Wholesale+customers")

    # Bagian Notebook Style
    st.markdown("### Proses Pemodelan (Jupyter Notebook Style)")
    
    # Sel 1
    st.markdown('<div class="notebook-cell">', unsafe_allow_html=True)
    st.markdown("#### 1. Preprocessing & Scaling")
    st.code("""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FITUR])
    """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sel 2
    st.markdown('<div class="notebook-cell">', unsafe_allow_html=True)
    st.markdown("#### 2. Mencari K-Optimal (Elbow Method)")
    # Gambar/Grafik Elbow bisa ditaruh di sini
    st.code("""
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: ABOUT ME
# ==========================================
with tab_about:
    col_img, col_bio = st.columns([1, 2])
    
    with col_img:
        # Gunakan placeholder jika file foto belum ada
        st.image("https://via.placeholder.com/300x300.png?text=Foto+Profil", width=250)
        
    with col_bio:
        st.markdown(f"""
        <div class="profile-card">
            <h2 style="margin:0;">Sabdo Winarah</h2>
            <p style="font-weight:bold; font-size:1.2rem;">Pelajar SMK / Developer</p>
            <hr style="border-color: #00033D">
            <div style="text-align:left;">
                <p><strong>Bahasa Pemrograman:</strong> Python, HTML, CSS, JavaScript</p>
                <p><strong>Biografi Singkat:</strong><br>
                Saya adalah seorang pelajar yang sangat tertarik dengan dunia Machine Learning dan Web Development. 
                Project ini merupakan bagian dari tugas sekolah saya untuk mengimplementasikan algoritma clustering 
                ke dalam aplikasi berbasis web yang interaktif.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
