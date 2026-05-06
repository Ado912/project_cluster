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
    page_title="SEGMENTAI - Pelanggan Grosir",
    page_icon="🛒",
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
    # Fallback jika file hilang: Train ulang cepat
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=2, random_state=42, n_init=10))
    ])
    model_pipeline.fit(df[FITUR].values)
    joblib.dump(model_pipeline, "cluster3.joblib")

model = joblib.load("cluster3.joblib")

# ─── CUSTOM CSS (THEME NAVY & PINK) ────────────────────────
st.markdown("""
<style>
    /* Main Theme */
    .stApp { background-color: #00033D; color: #FFCCF2; font-family: 'Syne', sans-serif; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #001550; border-right: 2px solid #FF007F; }
    
    /* Notebook Style for Code Tab */
    .notebook-cell {
        background-color: rgba(255, 0, 127, 0.05);
        border-left: 6px solid #FF007F;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .code-output {
        background-color: #00033D;
        border: 1px solid rgba(255, 204, 242, 0.2);
        padding: 10px;
        border-radius: 5px;
        color: #4F8EF7;
        font-family: monospace;
        margin-top: 5px;
    }

    /* Cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid #FF007F;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }

    /* Custom Headers */
    h1, h2, h3 { color: #FF007F !important; }
    
    /* Sidebar Logo */
    .logo-text {
        font-size: 2rem;
        font-weight: 800;
        color: #FF007F;
        text-align: center;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(255, 0, 127, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR: LOGO & PETUNJUK LENGKAP ─────────────────────
with st.sidebar:
    st.markdown('<div class="logo-text">SEGMENTAI</div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 Petunjuk Pengisian")
    st.write("""
    **Langkah-langkah Prediksi:**
    1. **Input Nominal**: Masukkan total pengeluaran tahunan pelanggan pada tiap kolom di sebelah kanan. 
    2. **Satuan**: Gunakan angka murni (contoh: `12000`). Satuan yang digunakan adalah *Monetary Units*.
    3. **Kategori Produk**:
       - *Fresh*: Sayur, buah, daging segar.
       - *Milk*: Susu, keju, yogurt.
       - *Grocery*: Sembako, minyak, beras.
       - *Frozen*: Makanan beku (nugget, dll).
       - *Detergents*: Sabun, detergen, tisu.
       - *Delicassen*: Makanan khusus/siap saji.
    4. **Proses**: Klik tombol **'Analisis Cluster Sekarang'**.
    5. **Hasil**: Lihat hasil klasifikasi dan grafik radar di bawah tombol.
    """)
    st.markdown("---")
    st.caption("v2.0 - Developed by Sabdo Winarah")

# ─── MAIN CONTENT: TABS ───────────────────────────────────
tab_prediksi, tab_analisis, tab_kode, tab_about = st.tabs([
    "🔮 Prediksi", "📊 Analisis Data", "💻 Kode", "👤 About Me"
])

# ==========================================
# TAB 1: PREDIKSI (HALAMAN UTAMA)
# ==========================================
with tab_prediksi:
    st.title("Sistem Prediksi Segmentasi Pelanggan")
    st.write("Masukkan data pengeluaran tahunan pelanggan untuk menentukan cluster.")

    # Layout Input Menyamping (Grid 3x2)
    st.markdown("### 📝 Input Parameter")
    c1, c2, c3 = st.columns(3)
    with c1:
        fresh = st.number_input("🥦 Fresh", 0, 150000, 10000)
        frozen = st.number_input("🧊 Frozen", 0, 150000, 3000)
    with c2:
        milk = st.number_input("🥛 Milk", 0, 150000, 5000)
        detergents = st.number_input("🧴 Detergents & Paper", 0, 150000, 2000)
    with c3:
        grocery = st.number_input("🛍️ Grocery", 0, 150000, 8000)
        delicassen = st.number_input("🧀 Delicassen", 0, 150000, 1500)

    btn_analisis = st.button("🚀 Analisis Cluster Sekarang", use_container_width=True, type="primary")

    st.markdown("---")

    if btn_analisis:
        # Prediksi
        input_data = np.array([[fresh, milk, grocery, frozen, detergents, delicassen]])
        prediction = model.predict(input_data)[0]
        
        # Informasi Hasil
        cluster_info = {
            0: {
                "nama": "🍽️ Restaurant / HoReCa",
                "desc": "Pelanggan ini cenderung membeli produk segar dan beku dalam jumlah besar. Biasanya merupakan bisnis restoran, hotel, atau kafe.",
                "recom": "Fokus pada penawaran bahan baku segar harian dan stok produk beku.",
                "color": "#34C47C"
            },
            1: {
                "nama": "🏪 Retail Store",
                "desc": "Pelanggan ini sangat dominan dalam pembelian Grocery (sembako) dan Detergents. Biasanya merupakan toko kelontong atau minimarket.",
                "recom": "Tawarkan paket diskon untuk produk rumah tangga dan sembako (bundling).",
                "color": "#4F8EF7"
            }
        }
        
        res = cluster_info[prediction]
        
        # Visualisasi Hasil
        col_res1, col_res2 = st.columns([1, 1.5])
        
        with col_res1:
            st.success(f"## Cluster: {res['nama']}")
            st.write(f"**Analisis:** {res['desc']}")
            st.info(f"💡 **Rekomendasi:** {res['recom']}")
            
        with col_res2:
            # Radar Chart Pendukung
            st.markdown("#### Visualisasi Perbandingan")
            # Ambil rata-rata cluster dari dataset asli untuk pembanding
            df_labeled = df.copy()
            df_labeled['Cluster'] = model.predict(df[FITUR].values)
            avg_val = df_labeled[df_labeled['Cluster'] == prediction][FITUR].mean().values
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=input_data[0], theta=FITUR, fill='toself', name='Input Kamu', line_color='#FF007F'
            ))
            fig.add_trace(go.Scatterpolar(
                r=avg_val, theta=FITUR, fill='toself', name='Rata-rata Cluster', line_color='rgba(255, 255, 255, 0.4)'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=False)),
                paper_bgcolor="rgba(0,0,0,0)",
                template="plotly_dark",
                height=350,
                margin=dict(l=50, r=50, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("<br><h4 style='text-align:center; color:#6B7280;'>Silakan isi data di atas dan klik tombol untuk melihat hasil</h4>", unsafe_allow_html=True)

# ==========================================
# TAB 2: ANALISIS DATA (DATASET EXPLORATION)
# ==========================================
with tab_analisis:
    st.title("Eksplorasi Dataset")
    st.write("Dataset ini berasal dari UCI Machine Learning Repository, berisi 440 data pelanggan grosir.")

    # 1. Statistik Dasar untuk Orang Awam
    st.markdown("### 🌍 Gambaran Umum")
    c_s1, c_s2, c_s3 = st.columns(3)
    c_s1.metric("Total Data", "440 Pelanggan")
    c_s2.metric("Fitur Utama", "6 Kategori Produk")
    c_s3.metric("Wilayah", "3 Wilayah (Lisbon, Oporto, Other)")

    # 2. Penjelasan Fitur (Notebook Content)
    st.markdown("#### Apa Saja Yang Kita Amati?")
    st.write("""
    Berdasarkan analisis awal, terdapat dua pilar utama dalam data ini:
    - **Sektor Retail**: Toko yang menyetok barang-barang rumah tangga (Grocery, Detergents).
    - **Sektor Horeca**: Bisnis konsumsi langsung (Fresh, Frozen).
    """)
    
    # Visualisasi Sederhana (Bar Chart rata-rata)
    st.markdown("#### Rata-rata Pembelian per Kategori")
    mean_all = df[FITUR].mean().sort_values(ascending=False)
    fig_bar = px.bar(mean_all, x=mean_all.index, y=mean_all.values, 
                     color=mean_all.values, color_continuous_scale="RdPu")
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", template="plotly_dark", height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Korelasi (Heatmap)
    st.markdown("#### 🔥 Heatmap Korelasi")
    st.write("Visualisasi ini menunjukkan hubungan antar produk. Jika dua produk berwarna terang, artinya pelanggan yang membeli produk A cenderung membeli produk B.")
    corr = df[FITUR].corr()
    fig_heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="PuRd", template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# TAB 3: KODE (JUPYTER NOTEBOOK FULL)
# ==========================================
with tab_kode:
    st.title("Dokumentasi Kode Model")
    st.write("Berikut adalah langkah-langkah pembuatan model dari awal hingga akhir.")

    # Sel 1
    st.markdown('<div class="notebook-cell">', unsafe_allow_html=True)
    st.markdown("##### [1] Import Library")
    st.code("""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
    """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)

    # Sel 2
    st.markdown('<div class="notebook-cell">', unsafe_allow_html=True)
    st.markdown("##### [2] Load Dataset & Pembersihan")
    st.code("""
df = pd.read_csv("Wholesale customers data.csv")
FITUR = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
X = df[FITUR].values
print(df.head())
    """, language="python")
    st.markdown('<div class="code-output">Head dataset terdeteksi: 440 rows x 8 columns</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Sel 3
    st.markdown('<div class="notebook-cell">', unsafe_allow_html=True)
    st.markdown("##### [3] Membangun Pipeline Model")
    st.code("""
# Pipeline menggabungkan scaling dan model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=2, random_state=42))
])

model.fit(X)
joblib.dump(model, "cluster3.joblib")
    """, language="python")
    st.markdown('<div class="code-output">Model dilatih menggunakan K=2 (Optimal berdasarkan Elbow Method)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 4: ABOUT ME
# ==========================================
with tab_about:
    st.markdown("<br>", unsafe_allow_html=True)
    col_me1, col_me2 = st.columns([1, 2])
    
    with col_me1:
        # Gunakan foto profil Anda di sini
        st.image("https://via.placeholder.com/300x300.png?text=Sabdo+Winarah", width=250)
    
    with col_me2:
        st.markdown(f"""
        <div style="background:#FFCCF2; color:#00033D; padding:30px; border-radius:20px;">
            <h2 style="color:#00033D !important; margin:0;">Sabdo Winarah</h2>
            <p style="font-weight:700; font-size:1.1rem;">Pelajar SMK / Developer Machine Learning</p>
            <hr style="border-color:#00033D">
            <p><strong>Bahasa Pemrograman:</strong> Python, JavaScript, SQL, HTML/CSS</p>
            <p><strong>Biografi:</strong><br>
            Saya adalah seorang pelajar yang fokus mendalami Data Science dan Machine Learning. 
            Project 'SEGMENTAI' ini dibuat untuk membantu distributor grosir memahami karakter pelanggan mereka secara otomatis 
            berdasarkan data historis transaksi.</p>
        </div>
        """, unsafe_allow_html=True)
