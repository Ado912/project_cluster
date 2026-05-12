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
import nbformat
import html

# ─── KONFIGURASI HALAMAN ───────────────────────────────────
st.set_page_config(
    page_title="GROMENT - Grosir Segmentation AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
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

# ─── CUSTOM CSS (UI SUPERIOR) ──────────────────────────────
st.markdown("""
<style>
    /* Main Theme: Modern Dark */
    .stApp { 
        background-color: #0B0E14; 
        color: #E2E8F0; 
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography */
    h1, h2, h3 { color: #F472B6 !important; font-family: 'Syne', sans-serif; }
    p, li, label, div { font-size: 0.9rem !important; color: #CBD5E0; line-height: 1.5; }
    
    /* Hero Banner di Tab Prediksi */
    .hero-section {
        background: linear-gradient(135deg, rgba(244,114,182,0.15) 0%, rgba(56,189,248,0.05) 100%);
        border: 1px solid rgba(244, 114, 182, 0.2);
        padding: 40px 30px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    .hero-title { font-size: 2.2rem !important; font-weight: 800; color: #F472B6; margin-bottom: 10px; }
    .hero-subtitle { font-size: 1.1rem !important; color: #94A3B8; max-width: 700px; margin: 0 auto; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] { 
        background-color: #12151C; 
        border-right: 1px solid rgba(244, 114, 182, 0.1); 
    }
    .logo-container { text-align: center; padding: 10px 0 20px 0; }
    .logo-text-groment { font-size: 2rem !important; font-weight: 800; color: #F472B6; letter-spacing: -1px; }
    .sidebar-info { background: rgba(56,189,248,0.05); border-left: 3px solid #38BDF8; padding: 15px; border-radius: 6px; margin-top: 20px; }

    /* Cards */
    .result-card {
        background: linear-gradient(to right, rgba(244, 114, 182, 0.05), rgba(0,0,0,0));
        border-left: 4px solid #F472B6;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 3px solid #F472B6;
        padding: 20px;
        border-radius: 10px;
        height: 100%;
        transition: transform 0.2s;
    }
    .feature-card:hover { transform: translateY(-5px); border-top: 3px solid #38BDF8; }

    /* Skill Badges untuk About Me */
    /* Skill Badges untuk About Me */
    .skill-badge {
        display: inline-block;
        background: rgba(244, 114, 182, 0.1);
        color: #F472B6;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem !important;
        margin: 4px 4px 4px 0;
        border: 1px solid rgba(244, 114, 182, 0.3);
    }
    .skill-badge.tech { background: rgba(56,189,248,0.1); color: #38BDF8; border-color: rgba(56,189,248,0.3); }

    /* Notebook Style Cells */
    .notebook-cell { background-color: rgba(255, 255, 255, 0.02); border-left: 3px solid #F472B6; padding: 15px; border-radius: 4px; margin-bottom: 10px; }
    .code-output { background-color: #050505; padding: 10px; border-radius: 6px; color: #93C5FD; font-size: 0.8rem !important; font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    # 1. LOGO BERBASIS TEKS (VINTAGE RETRO TECH STYLE)
    st.markdown("""
    <div style="background-color: #00033D; padding: 25px 15px; border: 2px solid #FF2D78; box-shadow: 4px 4px 0px #FF2D78; text-align: center; margin-bottom: 30px; border-radius: 4px;">
        <h1 style="color: #FF2D78 !important; font-family: 'Courier New', Courier, monospace; font-size: 2.2rem !important; font-weight: 900; margin: 0; letter-spacing: 2px; text-transform: uppercase; text-shadow: 2px 2px 0px rgba(0,0,0,0.8);">GROMENT</h1>
        <p style="color: #E2E8F0; font-size: 0.75rem !important; letter-spacing: 2px; margin-top: 5px; font-weight: bold;">SEGMENTATION AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. PANDUAN PENGGUNAAN
    st.markdown("### ⚙️ Panel Kontrol")
    st.info("""
    **Cara Penggunaan:**
    1. Input nilai belanja tahunan pelanggan.
    2. Gunakan angka bulat (tanpa koma/titik).
    3. Klik **Analisis Pola Sekarang**.
    """)
    
    # 3. SPESIFIKASI MODEL (MENGISI KEKOSONGAN AGAR TERLIHAT PRO)
    st.markdown("<hr style='border: 1px dashed #FF2D78; opacity: 0.5; margin: 25px 0;'>", unsafe_allow_html=True)
    st.markdown("### 🧬 Spesifikasi Model")
    st.markdown("""
    <div style="background: rgba(255, 45, 120, 0.05); border-left: 3px solid #FF2D78; padding: 12px; border-radius: 4px;">
        <ul style="margin:0; padding-left: 15px; font-size: 0.85rem; color: #CBD5E0; line-height: 1.8;">
            <li><b>Algoritma:</b> K-Means Clustering</li>
            <li><b>K-Value:</b> 2 (Segmen)</li>
            <li><b>Scaler:</b> StandardScaler</li>
            <li><b>Library:</b> Scikit-Learn</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 4. FOOTER
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #94A3B8; font-size: 0.75rem;">
        &copy; 2026 | Built by <b>Sabdo Winarah</b>
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN CONTENT: TABS ───────────────────────────────────
tab_prediksi, tab_analisis, tab_kode, tab_about = st.tabs([
    "🔮 Prediksi", "📊 Analisis Data", "💻 Kode", "👤 About Me"
])

# ==========================================
# TAB 1: PREDIKSI (DENGAN HERO SECTION)
# ==========================================
with tab_prediksi:
    # HERO SECTION BARU
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Segmentasi Pelanggan ML</div>
        <div class="hero-subtitle">Menganalisis kebiasaan belanja distributor grosir menggunakan algoritma Machine Learning (K-Means) untuk menentukan strategi pemasaran yang akurat.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 Masukkan Parameter Belanja Pelanggan")
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            fresh = st.number_input("🥦 Fresh (Produk Segar)", 0, 150000, 12000)
            frozen = st.number_input("🧊 Frozen (Produk Beku)", 0, 150000, 3000)
        with c2:
            milk = st.number_input("🥛 Milk (Produk Susu)", 0, 150000, 5000)
            detergents = st.number_input("🧴 Detergents & Paper", 0, 150000, 2500)
        with c3:
            grocery = st.number_input("🛍️ Grocery (Sembako)", 0, 150000, 8000)
            delicassen = st.number_input("🧀 Delicassen", 0, 150000, 1500)

    st.markdown("<br>", unsafe_allow_html=True)
    btn_analisis = st.button("🚀 Analisis Pola Sekarang", use_container_width=True, type="primary")

    if btn_analisis:
        input_data = np.array([[fresh, milk, grocery, frozen, detergents, delicassen]])
        prediction = model.predict(input_data)[0]
        
        cluster_info = {
            0: {"nama": "Restaurant / HoReCa", "desc": "Segmen ini didominasi oleh pengeluaran bahan segar harian.", "recom": "Berikan penawaran bahan baku segar volume besar."},
            1: {"nama": "Retail Store", "desc": "Segmen ini dominan pada produk Grocery dan kebutuhan rumah tangga.", "recom": "Tawarkan paket bundling sembako dan detergen."}
        }
        res = cluster_info[prediction]
# 2. LOGIKA DINAMIS: Cari pembelian paling tinggi dari input user
        nama_kategori = ["Fresh (Produk Segar)", "Milk (Produk Susu)", "Grocery (Sembako)", "Frozen (Produk Beku)", "Detergents & Paper (Sabun/Tisu)", "Delicassen (Daging Premium/Siap Saji)"]
        idx_tertinggi = np.argmax(input_data[0]) # Mencari index dengan nilai tertinggi
        kategori_tertinggi = nama_kategori[idx_tertinggi]
        
        # 3. Buat Rekomendasi Spesifik Berdasarkan Kategori Tertinggi
        aksi_spesifik = {
            0: "Pastikan armada logistik pendingin (cold-chain) selalu siap, karena demand sayur/daging pelanggan ini sangat besar.",
            1: "Tawarkan keanggotaan premium (supplier prioritas) untuk pasokan produk susu dan keju rutin dengan harga khusus.",
            2: "Siapkan palet barang di area gudang yang mudah dijangkau armada. Pastikan stok sembako (beras/minyak) untuk pelanggan ini tidak pernah putus.",
            3: "Pelanggan ini memiliki kapasitas *freezer* yang besar. Tawarkan promo *bundling* produk beku keluaran terbaru.",
            4: "Tawarkan katalog produk kebersihan ukuran jerigen/karton besar (skala industri) untuk menekan harga beli mereka.",
            5: "Tawarkan produk-produk impor atau daging olahan premium edisi terbatas untuk melengkapi etalase/menu spesial mereka."
        }
        saran_tindakan = aksi_spesifik[idx_tertinggi]
         # Card Hasil Prediksi
        st.markdown(f"""
        <div class="result-card">
            <h1 style='margin:0;'>{res['nama']}</h1>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
         <div style="background: rgba(56,189,248,0.1); border-left: 4px solid #38BDF8; padding: 15px; border-radius: 6px; margin-bottom: 15px; margin-top:15px;">
            <h4 style="color: #38BDF8; margin-top: 0; margin-bottom: 10px;">💡 Rekomendasi Tindakan </h4>
            <p style="margin:0; font-size: 0.95rem;">Sistem mendeteksi bahwa fokus utama belanja pelanggan ini ada pada <b>{kategori_tertinggi}</b>.</p>
            <p style="margin: 8px 0 0 0; font-size: 0.95rem;"><b>Tindakan:</b> {saran_tindakan}</p>
         </div>
          """, unsafe_allow_html=True)
        
        # VISUALISASI PENDUKUNG
        st.markdown("### 📈 Visualisasi Pendukung Prediksi")
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            avg_df = df.copy()
            avg_df['Cluster'] = model.predict(df[FITUR].values)
            c_mean = avg_df[avg_df['Cluster'] == prediction][FITUR].mean().values
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=input_data[0], theta=FITUR, fill='toself', name='Data Anda', line_color='#38BDF8'))
            fig_radar.add_trace(go.Scatterpolar(r=c_mean, theta=FITUR, fill='toself', name='Rata-rata Cluster', line_color='rgba(244,114,182,0.4)'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(t=30, b=30))
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_v2:
            fig_pos = px.scatter(df, x="Fresh", y="Grocery", opacity=0.3, template="plotly_dark", color_discrete_sequence=["#475569"])
            fig_pos.add_trace(go.Scatter(x=[fresh], y=[grocery], mode='markers', marker=dict(size=14, color='#38BDF8', symbol='star'), name='Posisi Anda'))
            fig_pos.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(t=30, b=30))
            st.plotly_chart(fig_pos, use_container_width=True)

        st.markdown("#### Detail Komparasi Global")
        g_avg = df[FITUR].mean()
        comp_df = pd.DataFrame({'Kategori': FITUR, 'Input Anda': input_data[0], 'Rata-rata Global': g_avg.values})
        fig_bar = px.bar(comp_df, x='Kategori', y=['Input Anda', 'Rata-rata Global'], barmode='group', color_discrete_map={'Input Anda': '#38BDF8', 'Rata-rata Global': '#475569'})
        fig_bar.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 2: ANALISIS DATA
# ==========================================
with tab_analisis:
    st.markdown("<h1>Analisis Dataset</h1>", unsafe_allow_html=True)
    st.write("""
    Dataset **Wholesale Customers** ini saya dapatkan dari ***UCI Machine Learning Repository***. 
    Data ini berasal dari sebuah distributor grosir di ***Portugal*** dan merekam kebiasaan belanja tahunan dari **440 pelanggan**.
    
    Data aslinya memiliki 8 fitur (kategori), yaitu: `FRESH`, `MILK`, `GROCERY`, `FROZEN`, `DETERGENTS_PAPER`, `DELICASSEN`, `CHANNEL`, dan `REGION`. 
    Namun, pada pemodelan ini, saya **tidak menggunakan fitur Region dan Channel** agar model AI murni berfokus pada pola jumlah barang yang dibeli pelanggan.
    """)
    
    # PERBAIKAN METRIC TOTAL PELANGGAN MENJADI 440
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Pelanggan", "440 Data")
    c2.metric("Jumlah Cluster", "2 Segmen")
    c3.metric("Fitur Observasi", "6 Kategori")
    c4.metric("Silhouette Score", "0.5483")
    
    st.write("""
    Dengan implementasi Machine Learning ini, pihak manajemen grosir diharapkan mampu merumuskan langkah bisnis yang sangat strategis karena telah memahami target penjualannya secara presisi, serta menjaga stabilitas stok gudang agar barang yang tinggi permintaannya selalu tersedia.
    """)
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)

    st.markdown("### 🛒 Karakteristik Fitur Produk")
    f_c1, f_c2, f_c3 = st.columns(3) 
    with f_c1: 
        st.markdown('<div class="feature-card"><span style="font-size:1.5rem;">🥦</span><br><b style="color:#F472B6; font-size:1.1rem;">Fresh</b><br>Sayur, buah, dan daging segar harian.</div>', unsafe_allow_html=True) 
        st.markdown('<br>', unsafe_allow_html=True) 
        st.markdown('<div class="feature-card"><span style="font-size:1.5rem;">🧊</span><br><b style="color:#F472B6; font-size:1.1rem;">Frozen</b><br>Makanan beku (nugget, sosis, patty, dll).</div>', unsafe_allow_html=True) 
    with f_c2: 
        st.markdown('<div class="feature-card"><span style="font-size:1.5rem;">🥛</span><br><b style="color:#F472B6; font-size:1.1rem;">Milk</b><br>Produk olahan susu seperti keju dan yogurt.</div>', unsafe_allow_html=True) 
        st.markdown('<br>', unsafe_allow_html=True) 
        st.markdown('<div class="feature-card"><span style="font-size:1.5rem;">🧴</span><br><b style="color:#F472B6; font-size:1.1rem;">Detergents</b><br>Sabun, tisu, dan alat kebersihan.</div>', unsafe_allow_html=True) 
    with f_c3: 
        st.markdown('<div class="feature-card"><span style="font-size:1.5rem;">🛍️</span><br><b style="color:#F472B6; font-size:1.1rem;">Grocery</b><br>Sembako pokok seperti beras, minyak, tepung.</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card"><span style="font-size:1.5rem;">🧀</span><br><b style="color:#F472B6; font-size:1.1rem;">Delicassen</b><br>Daging premium dan makanan siap saji.</div>', unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    st.markdown("## 📈 Insight Visualisasi Data")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("#### Produk Paling Banyak Dibeli")
        st.write("Rata-rata pengeluaran tertinggi jatuh pada produk **Fresh** dan **Grocery**.")
        mean_all = df[FITUR].mean().sort_values(ascending=False)
        fig_bar_all = px.bar(mean_all, x=mean_all.index, y=mean_all.values, color=mean_all.values, color_continuous_scale="RdPu")
        fig_bar_all.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=20))
        st.plotly_chart(fig_bar_all, use_container_width=True)

    with col_a2:
        st.markdown("#### Porsi Penjualan Terbesar")
        st.write("Distribusi dari keseluruhan uang yang dibelanjakan oleh 440 pelanggan.")
        fig_donut = px.pie(names=mean_all.index, values=mean_all.values, hole=0.5, color_discrete_sequence=px.colors.sequential.RdPu_r)
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=350, showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("#### 🔥 Hubungan Antar Produk (Heatmap)")
    col_h1, col_h2 = st.columns([1.5, 1])
    with col_h1:
        fig_heat = px.imshow(df[FITUR].corr(), text_auto=".2f", color_continuous_scale="RdPu", template="plotly_dark")
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)
    with col_h2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("💡 **Fakta Menarik:**\n\nKorelasi yang sangat cerah (tinggi) antara **Grocery** dan **Detergents_Paper** menunjukkan bahwa *Pelanggan yang membeli banyak sembako hampir pasti memborong sabun dan tisu.*\n\nIni adalah indikator kuat dari pola belanja **Toko Kelontong/Retail** yang menyetok barang untuk dijual kembali.")

    st.markdown("#### 📏 Rentang Belanja & Pelanggan 'Sultan'")
    st.write("Setiap titik di luar kotak (sebelah kanan) adalah anomali pelanggan dengan daya beli sangat ekstrem. Ini menjadi alasan utama kenapa kita butuh Machine Learning untuk memisahkan segmentasi ini dengan akurat.")
    fig_box = px.box(df, y=FITUR, color_discrete_sequence=['#38BDF8'], orientation='v')
    fig_box.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
    st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# TAB 3: KODE (NOTEBOOK RENDERER)
# ==========================================
with tab_kode:
    st.markdown("<h1>📓 Notebook Viewer</h1>", unsafe_allow_html=True)
    
    nb_path = os.path.join(os.path.dirname(__file__), "clustering.ipynb")
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        for i, cell in enumerate(nb.cells):
            tipe  = cell.cell_type
            label = "🟦 Markdown" if tipe == "markdown" else "🟩 Code"
            st.markdown(
                f"""<div style="background:rgba(244, 114, 182, 0.05); border-left:3px solid #F472B6;
                    padding:6px 14px; border-radius:6px; margin-bottom:4px;">
                    <span style="color:#F472B6; font-size:12px; font-weight:600;">Cell [{i+1}]</span>
                    <span style="color:rgba(255,255,255,0.4); font-size:12px;">&nbsp;·&nbsp;{label}</span>
                </div>""", unsafe_allow_html=True)

            if tipe == "markdown":
                st.markdown(cell.source)
            elif tipe == "code":
                st.code(cell.source, language="python")
                for output in cell.get("outputs", []):
                    if output.output_type == "stream":
                        teks = html.escape(output.text)
                        st.markdown(f"""<div class="code-output">{teks}</div>""", unsafe_allow_html=True)
                    elif output.output_type in ("display_data", "execute_result"):
                        if "image/png" in output.data:
                            import base64
                            img_data = output.data["image/png"]
                            st.markdown(f'<img src="data:image/png;base64,{img_data}" style="max-width:100%; border-radius:8px; margin-top:6px;">', unsafe_allow_html=True)
                        elif "text/html" in output.data:
                            st.markdown(output.data["text/html"], unsafe_allow_html=True)
                        elif "text/plain" in output.data:
                            teks = html.escape(output.data["text/plain"])
                            st.markdown(f"""<div class="code-output">{teks}</div>""", unsafe_allow_html=True)
            st.markdown("<hr style='border:0.5px solid rgba(244, 114, 182, 0.2); margin:12px 0'>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"File Notebook tidak ditemukan di: {nb_path}.")

# ==========================================
# TAB 4: ABOUT ME (PORTFOLIO STYLE)
# ==========================================
with tab_about:
    st.markdown("<br>", unsafe_allow_html=True)
    c_m1, c_m2 = st.columns([1, 3])
    
    with c_m1:
        if os.path.exists("prime.png"):
            st.image("prime.png", use_container_width=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #F472B6, #38BDF8); padding: 4px; border-radius: 50%; max-width: 200px; margin: 0 auto;">
                <img src="prime.png" style="border-radius: 50%; width: 100%; background: #0B0E14;">
            </div>
            """, unsafe_allow_html=True)
            
    with c_m2:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); padding: 30px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);">
            <h1 style='margin:0; font-size:2.5rem; color:#F472B6 !important; text-align:left;'>Sabdo Winarah</h1>
            <p style="color:#38BDF8; font-weight:600; font-size:1.1rem !important; margin-top:5px;">Pelajar SMK Negeri 1 Purbalingga | Full-Stack & ML Enthusiast</p>
            <p style="margin-top: 15px;">Saya sangat tertarik pada pengembangan solusi digital, mulai dari UI/UX desain hingga arsitektur Machine Learning. Proyek Groment ini adalah implementasi praktis bagaimana algoritma Clustering dapat diterapkan untuk menyelesaikan studi kasus segmentasi pasar di dunia nyata.</p>
            
        </div>
        <div style="margin-top: 20px;">
                <h4 style="color:#F472B6; margin-bottom:10px;">Tech Stack & Skills</h4>
                
                <span style="display:inline-block; background:rgba(244,114,182,0.1); color:#F472B6; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:4px 4px 4px 0; border:1px solid rgba(244,114,182,0.3);">Machine Learning (K-Means, CatBoost)</span>
                <span style="display:inline-block; background:rgba(244,114,182,0.1); color:#F472B6; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:4px 4px 4px 0; border:1px solid rgba(244,114,182,0.3);">Data Science (Pandas, Scikit-learn)</span>
                
                <span style="display:inline-block; background:rgba(56,189,248,0.1); color:#38BDF8; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:4px 4px 4px 0; border:1px solid rgba(56,189,248,0.3);">Python & Streamlit</span>
                <span style="display:inline-block; background:rgba(56,189,248,0.1); color:#38BDF8; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:4px 4px 4px 0; border:1px solid rgba(56,189,248,0.3);">Next.js, React, PHP</span>
                <span style="display:inline-block; background:rgba(56,189,248,0.1); color:#38BDF8; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:4px 4px 4px 0; border:1px solid rgba(56,189,248,0.3);">HTML, CSS, JS</span>
                <span style="display:inline-block; background:rgba(56,189,248,0.1); color:#38BDF8; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:4px 4px 4px 0; border:1px solid rgba(56,189,248,0.3);">Tailwind & Bootstrap</span>
                
                <span style="display:inline-block; background:rgba(244,114,182,0.1); color:#F472B6; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin:4px 4px 4px 0; border:1px solid rgba(244,114,182,0.3);">UI/UX (Figma, Canva)</span>
        </div>
        """, unsafe_allow_html=True)
