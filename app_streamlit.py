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

# ─── CUSTOM CSS (REVISI: SOFT COLORS, SMALL FONTS, LOGO) ──
st.markdown("""
<style>
    /* Main Theme: Modern Dark & Soft Pink */
    .stApp { 
        background-color: #0E1117; 
        color: #E2E8F0; 
        font-family: 'Inter', sans-serif;
    }
    
    /* Font Size Adjustments (Lebih Kecil & Rapi) */
    h1 { font-size: 1.5rem !important; color: #F472B6 !important; font-weight: 800; margin-bottom: 1rem; }
    h2 { font-size: 1.2rem !important; color: #F472B6 !important; font-weight: 700; }
    h3 { font-size: 1.0rem !important; color: #F472B6 !important; }
    p, li, label, div { font-size: 0.85rem !important; color: #CBD5E0; line-height: 1.4; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { 
        background-color: #161922; 
        border-right: 1px solid rgba(244, 114, 182, 0.1); 
    }
    
    /* Logo Header di Sidebar */
    .logo-container {
        text-align: center;
        padding: 10px 0 20px 0;
    }
    .logo-text-groment {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem !important;
        font-weight: 800;
        color: #F472B6;
        letter-spacing: -1px;
    }

    /* Result Card */
    .result-card {
        background: rgba(244, 114, 182, 0.05);
        border: 1px solid rgba(244, 114, 182, 0.3);
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }

    /* Notebook Style Cells */
    .notebook-cell {
        background-color: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #F472B6;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .code-output {
        background-color: #050505;
        padding: 8px;
        border-radius: 4px;
        color: #93C5FD;
        font-size: 0.75rem !important;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR: LOGO GROMENT & PETUNJUK ─────────────────────
with st.sidebar:
    # Menggunakan Link Image Logo yang telah dibuat
    st.image("http://googleusercontent.com/image_generation_content/1", use_container_width=True)
    st.markdown('<div class="logo-container"><span class="logo-text-groment">GROMENT</span></div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 Panduan Prediksi")
    st.info("""
    1. Masukkan nilai pengeluaran tahunan pada grid di sebelah kanan.
    2. Nilai harus berupa angka positif.
    3. Klik tombol **Analisis** untuk memproses.
    4. Perhatikan posisi data Anda pada grafik distribusi di bawah hasil.
    """)
    st.markdown("---")
    st.caption("Machine Learning Project by Sabdo Winarah")

# ─── MAIN CONTENT: TABS ───────────────────────────────────
tab_prediksi, tab_analisis, tab_kode, tab_about = st.tabs([
    "🔮 Prediksi", "📊 Analisis Data", "💻 Kode", "👤 About Me"
])

# ==========================================
# TAB 1: PREDIKSI (HALAMAN UTAMA)
# ==========================================
with tab_prediksi:
    st.markdown("<h1>Prediksi Segmentasi Pelanggan</h1>", unsafe_allow_html=True)
    
    # Input Grid Horizontal (3 Kolom x 2 Baris)
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
        
        cluster_info = {
            0: {"nama": "Restaurant / HoReCa", "desc": "Segmen ini didominasi oleh pengeluaran bahan segar harian.", "recom": "Berikan penawaran bahan baku segar volume besar."},
            1: {"nama": "Retail Store", "desc": "Segmen ini dominan pada produk Grocery dan kebutuhan rumah tangga.", "recom": "Tawarkan paket bundling sembako dan detergen."}
        }
        res = cluster_info[prediction]

        # Card Hasil Prediksi
        st.markdown(f"""
        <div class="result-card">
            <h2 style='margin:0;'>Klasifikasi: {res['nama']}</h2>
            <p>{res['desc']}</p>
            <p style='color:#F472B6;'><b>Rekomendasi:</b> {res['recom']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        # VISUALISASI PENDUKUNG (3 Visualisasi)
        st.markdown("### 📈 Visualisasi Pendukung Jawaban")
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            # 1. Radar Chart (Karakteristik Cluster)
            st.markdown("#### Perbandingan Karakteristik")
            avg_df = df.copy()
            avg_df['Cluster'] = model.predict(df[FITUR].values)
            c_mean = avg_df[avg_df['Cluster'] == prediction][FITUR].mean().values
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=input_data[0], theta=FITUR, fill='toself', name='Data Anda', line_color='#F472B6'))
            fig_radar.add_trace(go.Scatterpolar(r=c_mean, theta=FITUR, fill='toself', name='Rata-rata Cluster', line_color='rgba(255,255,255,0.2)'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_v2:
            # 2. Scatter Plot (Posisi Data)
            st.markdown("#### Posisi Anda di Distribusi Data")
            fig_pos = px.scatter(df, x="Fresh", y="Grocery", opacity=0.2, template="plotly_dark")
            fig_pos.add_trace(go.Scatter(x=[fresh], y=[grocery], mode='markers', marker=dict(size=12, color='#F472B6', symbol='x'), name='Posisi Anda'))
            fig_pos.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_pos, use_container_width=True)

        # 3. Bar Chart (Detail Perbandingan Global)
        st.markdown("#### Perbandingan Detail Terhadap Rata-rata Global")
        g_avg = df[FITUR].mean()
        comp_df = pd.DataFrame({'Kategori': FITUR, 'Input Anda': input_data[0], 'Rata-rata Global': g_avg.values})
        fig_bar = px.bar(comp_df, x='Kategori', y=['Input Anda', 'Rata-rata Global'], barmode='group', color_discrete_map={'Input Anda': '#F472B6', 'Rata-rata Global': '#4A5568'})
        fig_bar.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 2: ANALISIS DATA
# ==========================================
with tab_analisis:
    st.markdown("<h1>Analisis Dataset</h1>", unsafe_allow_html=True)
    st.write("Penjelasan dataset Wholesale Customers (UCI Repository).")

    f_c1, f_c2, f_c3 = st.columns(3) 
    with f_c1: 
        st.markdown('<div class="feature-card"><b>🥦 Fresh</b><br>Sayur-sayur an, buah-buah an, dan daging segar harian.</div>', unsafe_allow_html=True) 
        st.markdown('<br>', unsafe_allow_html=True) 
        st.markdown('<div class="feature-card"><b>🧊 Frozen</b><br>Makanan beku seperti nugget,sosis,Bakso,Patty,Fish Roll,dll.</div>', unsafe_allow_html=True) 
    with f_c2: 
        st.markdown('<div class="feature-card"><b>🥛 Milk</b><br>Produk olahan susu seperti keju dan yogurt.</div>', unsafe_allow_html=True) 
        st.markdown('<br>', unsafe_allow_html=True) 
        st.markdown('<div class="feature-card"><b>🧴 Detergents & Paper</b><br>Sabun, tisu, dan alat kebersihan.</div>', unsafe_allow_html=True) 
    with f_c3: 
        st.markdown('<div class="feature-card"><b>🛍️ Grocery</b><br>Sembako pokok seperti beras, minyak, dan tepung.</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card"><b>🧀 Delicassen</b><br>Daging olahan premium dan makanan siap saji.</div>', unsafe_allow_html=True)
    
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
    	fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_heat, use_container_width=True)
    with col_h2:
        st.markdown("### 💡 Fakta Menarik")
        st.write("""
        Jika Anda melihat kotak yang sangat cerah antara **Grocery** dan **Detergents_Paper**, itu artinya:
    
        *Pelanggan yang membeli banyak sembako hampir pasti juga membeli banyak sabun dan tisu.*
        
        Hal ini sangat logis karena Toko Kelontong biasanya menyetok kedua barang ini secara bersamaan untuk dijual kembali.
        """)

# ==========================================
# TAB 3: KODE (NOTEBOOK FULL)
# ==========================================
with tab_kode:
    st.markdown("<h1>Full Notebook Pipeline</h1>", unsafe_allow_html=True)
    
    steps = [
        {"title": "1. Import Library", "code": "import pandas as pd\nimport numpy as np\nfrom sklearn.cluster import KMeans", "out": "Libraries Loaded Successfully."},
        {"title": "2. Data Preprocessing", "code": "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(df[FITUR])", "out": "Data Normalized (StandardScaler)."},
        {"title": "3. K-Means Training", "code": "model = KMeans(n_clusters=2, random_state=42)\nmodel.fit(X_scaled)", "out": "Model Fit: Cluster Centers Calculated."}
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
        st.markdown(f"""
        <div style="background:rgba(244, 114, 182, 0.05); padding:20px; border-radius:10px; border:1px solid rgba(244, 114, 182, 0.2);">
            <h2 style='margin:0; color:#F472B6 !important;'>Sabdo Winarah</h2>
            <p><b>Profesi:</b> Pelajar / ML Developer</p>
            <p><b>Bio:</b> Fokus pada pengembangan solusi cerdas menggunakan Machine Learning. Project Groment ini adalah aplikasi segmentasi pasar berbasis data nyata.</p>
        </div>
        """, unsafe_allow_html=True)
