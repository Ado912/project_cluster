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

# ─── Config ───────────────────────────────────────────────
# Bagian load model
if not os.path.exists("cluster3.joblib"):
    df_train = pd.read_csv("Wholesale customers data.csv")
    FITUR_TRAIN = ["Fresh", "Milk", "Grocery", "Frozen", 
                   "Detergents_Paper", "Delicassen"]
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

st.set_page_config(
    page_title="Segmentasi Pelanggan Grosir",
    page_icon="🛒",
    layout="wide"
)
# 1. Set halaman default
if "halaman" not in st.session_state:
    st.session_state.halaman = "Project"


# ─── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #00033D; color: #FFCCF2; font-family: 'Syne', sans-serif;}
.css-card { 
        background-color: #001550; 
        padding: 20px; 
        border-radius: 15px; 
        border: 2px solid #FF007F; 
        font-family: 'DM Sans', sans-serif;
}
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 2rem;
    background: #00033D;
   
    border-radius: 12px;
    margin-bottom: 2rem;
    color : #FFCCF2;
}

.nav-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.95rem;
    color: #FFCCF2;
    margin bottom:0;
    
}
[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #FFCCF2 !important;
    border-radius: 20px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.35rem 1rem !important;
    width: 100% !important;
}

[data-testid="stButton"] > button:hover {
    background: #FFCCF2 !important;
    color: #00033D !important;
}
.hero-title h1{
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #FFCCF2;
}

.hero-sub {
    font-size: 0.88rem;
    color: #6B7280;
    margin-bottom: 2rem;
    line-height: 1.6;
    color:#FFCCF2;
    text-align:center;
}
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4F8EF7;
    margin-bottom: 0.4rem;
    margin-top: 2rem;
}
.project-title h1{
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 1000;
    color: #FFCCF2;
    text-align:center;
}
.project-box {
    background: #FFCCF2 ;
    border: 1px solid rgba(79,142,247,0.2);
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 1rem;
}

.project-box p {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.90rem;
    color: #00033D!important;
    line-height: 1.8;
    margin: 0;
   
    text-align:center;
}
</style>

<div class="navbar">
    <div class="nav-logo">K-Means Clustering</div>
</div>
""", unsafe_allow_html=True)
# 2. Tombol navbar pakai st.button
col1, col2, col3, col4 ,col5 = st.columns([3, 1, 1, 1,1])
with col2:
    if st.button("Project"):
        st.session_state.halaman = "Project"
with col3:
    if st.button("Data Set"):
        st.session_state.halaman = "Data Set"
with col4:
    if st.button("Prediksi"):
        st.session_state.halaman = "Prediksi"
with col5:
    if st.button("code"):
        st.session_state.halaman= "code"

# 3. Tampilkan konten sesuai halaman
if st.session_state.halaman == "Project":
    # isi halaman project di sini
    
    st.markdown(""" 
<div class="hero-title">
<h1 >Segmentasi Pelanggan Grosir</h1>
</div>
<p class=hero-sub>Project clustering oleh Sabdo Winarah — mengelompokkan pelanggan  
distributor grosir berdasarkan pola pembelian mereka</p>
            """,unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Pelanggan", "400")
    c2.metric("Jumlah Cluster", "2")
    c3.metric("Fitur Data", "6")
    c4.metric("Silhouette Score", "0.5483")
    st.markdown("""
    <div class="project-title">
    <h1>Apa Itu Project Ini?</h1>
    </div>
    <div class="project-box">
    <p>
        Project ini bertujuan untuk mengelompokkan pelanggan grosir 
        berdasarkan pola pembelian tahunan mereka menggunakan metode 
        K-Means Clustering.
        Dari  pada melayani semua pelanggan dengan cara yang sama, 
        kita kelompokkan mereka berdasarkan 
        apa yang sering mereka beli — sehingga strategi bisnis 
        bisa lebih tepat sasaran.
    </p>
        </div>
    """, unsafe_allow_html=True)
elif st.session_state.halaman == "Data Set":
    # isi halaman dataset di sini
    st.markdown("""      
                
        

        <h1  style="font-family:'Syne',sans-serif; font-weight:800;font-size: 2.2rem; 
        color:#FFCCF2">
        Mengenal Datanya
        </h1>     
        <p style="color:#FFCCF2; font-size:0.88rem;">
    Dataset berisi pembelian tahunan (satuan moneter) dari 440 pelanggan  distributor grosir di Portugal</p>    
    """,unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#FFCCF2; border-radius:12px; padding:1rem; margin-bottom:0.75rem;">
        <p style="color:#00033D; font-weight:700; margin:0;">🥦 Fresh</p>
        <p style="color:#00033D; font-size:0.82rem; margin:0.3rem 0 0 0;">
        Produk segar — sayur, buah, daging</p>
        </div>

        <div style="background:#FFCCF2; border-radius:12px; padding:1rem;">
            <p style="color:#00033D; font-weight:700; margin:0;">🥛 Milk</p>
            <p style="color:#00033D; font-size:0.82rem; margin:0.3rem 0 0 0;">
            Produk susu — susu, keju, yogurt</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
            st.markdown("""
        <div style="background:#FFCCF2;border-radius:12px; padding:1rem; margin-bottom:0.75rem;">
        <p style="color:#00033D; font-weight:700; margin:0;">🛍️ Grocery</p>
        <p style="color:#00033D; font-size:0.82rem; margin:0.3rem 0 0 0;">
        Kebutuhan pokok — beras, minyak, gula</p>
        </div>

        <div style="background:#FFCCF2; border-radius:12px; padding:1rem;">
        <p style="color:#00033D; font-weight:700; margin:0;">🧊 Frozen</p>
        <p style="color:#00033D; font-size:0.82rem; margin:0.3rem 0 0 0;">
        Produk beku — nugget, es krim</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
    <div style="background:#FFCCF2; border-radius:12px; padding:1rem; margin-bottom:0.75rem;">
        <p style="color:#00033D; font-weight:700; margin:0;">🧴 Detergents_Paper</p>
        <p style="color:#00033D; font-size:0.82rem; margin:0.3rem 0 0 0;">
        Detergen dan kertas — sabun, tisu</p>
    </div>

    <div style="background:#FFCCF2; border-radius:12px; padding:1rem;">
        <p style="color:#00033D; font-weight:700; margin:0;">🧀 Delicassen</p>
        <p style="color:#00033D; font-size:0.82rem; margin:0.3rem 0 0 0;">
        Makanan spesial — keju impor, deli</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ─── HEATMAP KORELASI ─────────────────────────────────────
    st.markdown("""
<h3 style="font-family:'Syne',sans-serif; font-weight:700; 
color:#FFCCF2; margin-bottom:0.5rem;">🔥 Heat Map</h3>
<p style="color:#FFCCF2; font-size:0.85rem; margin-bottom:1rem;">
Korelasi antar fitur dataset — semakin terang semakin kuat hubungannya.</p>
""", unsafe_allow_html=True)

    corr = df[FITUR].corr()
    fig_heatmap = px.imshow(
    corr,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    text_auto=".2f",
    template="plotly_dark"
)
    fig_heatmap.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0),
    coloraxis_colorbar=dict(tickfont=dict(color="#FFCCF2"))
)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ─── SCATTER PLOT PERSEBARAN CLUSTER ──────────────────────
    st.markdown("""
<h3 style="font-family:'Syne',sans-serif; font-weight:700; 
color:#FFCCF2; margin-bottom:0.5rem;">Persebaran Cluster</h3>
<p style="color:#FFCCF2; font-size:0.85rem; margin-bottom:1rem;">
Persebaran 440 pelanggan berdasarkan hasil clustering K-Means.</p>
""", unsafe_allow_html=True)

    df_vis = df[FITUR].copy()
    df_vis["Cluster"] = model.predict(df_vis.values)
    cluster_names = {
    0: "Retail Store",
    1: "Restaurant / HoReCa Besar"
}
    df_vis["Tipe Pelanggan"] = df_vis["Cluster"].map(cluster_names)

    color_map = {
    "Retail Store": "#4F8EF7",
    "Restaurant / HoReCa Besar": "#FF7043"
}

    fig_scatter = px.scatter(
    df_vis, x="Fresh", y="Grocery",
    color="Tipe Pelanggan",
    color_discrete_map=color_map,
    hover_data=["Milk", "Frozen"],
    title="Hasil Clustering Pelanggan (K=3)",
    template="plotly_dark"
)
    fig_scatter.update_traces(marker=dict(size=7, opacity=0.8))
    fig_scatter.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        font=dict(color="#FFCCF2")
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    title_font=dict(color="#FFCCF2")
)
    df_vis = df[FITUR].copy()
    df_vis["Cluster"] = model.predict(df_vis.values)
    st.write(df_vis["Cluster"].value_counts())
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ─── TABEL RATA-RATA PER CLUSTER ──────────────────────────
    st.markdown("""
<h3 style="font-family:'Syne',sans-serif; font-weight:700; 
color:#FFCCF2; margin-bottom:0.5rem;">📊 Rata-rata per Cluster</h3>
""", unsafe_allow_html=True)

    mean_df = df_vis.groupby("Cluster")[FITUR].mean().round(0)
    st.dataframe(mean_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ─── ELBOW CHART ──────────────────────────────────────────
    st.markdown("""
<h3 style="font-family:'Syne',sans-serif; font-weight:700; 
color:#FFCCF2; margin-bottom:0.5rem;">📉 Elbow Method</h3>
<p style="color:#FFCCF2; font-size:0.85rem; margin-bottom:1rem;">
Grafik ini menunjukkan kenapa kita memilih K=3 sebagai jumlah cluster optimal.</p>
""", unsafe_allow_html=True)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FITUR])

    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    fig_elbow = px.line(
        x=list(K_range),
        y=inertia,
        markers=True,
        labels={"x": "Jumlah Cluster (K)", "y": "Inertia"},
        title="Elbow Method — Cluster ada 3",
        template="plotly_dark"
)
    fig_elbow.add_vline(
        x=2,
        line_dash="dash",
        line_color="#FF007F",
        annotation_text="K=2 Optimal",
        annotation_font_color="#FF007F"
)
    fig_elbow.update_traces(
    line_color="#4F8EF7",
    marker=dict(color="#FF007F", size=8)
)
    fig_elbow.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=40, b=0),
    title_font=dict(color="#FFCCF2"),
    xaxis=dict(tickfont=dict(color="#FFCCF2")),
    yaxis=dict(tickfont=dict(color="#FFCCF2"))
)
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ─── PENJELASAN 3 CLUSTER ─────────────────────────────────
    st.markdown("""
<h3 style="font-family:'Syne',sans-serif; font-weight:700; 
color:#FFCCF2; margin-bottom:1rem;">💡 Penjelasan Hasil Cluster</h3>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
    <div style="background:#FFCCF2; border:1px solid 
    rgba(79,142,247,0.3); border-radius:12px; padding:1.25rem;">
        <p style="color:#00033D; font-weight:700; font-size:1rem; margin:0;">
        🏪 Retail Store</p>
        <p style="color:#00033D; font-size:0.82rem; margin:0.5rem 0 0 0; line-height:1.6;">
        Banyak membeli Grocery, Milk, dan Detergents. 
        Ciri khas toko retail atau minimarket yang menjual ke konsumen akhir.</p>
    </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
    <div style="background:#FFCCF2; border:1px solid 
    rgba(52,196,124,0.3); border-radius:12px; padding:1.25rem;">
        <p style="color:#00033D; font-weight:700; font-size:1rem; margin:0;">
        🍽️ HoReCa Besar</p>
        <p style="color:#00033D; font-size:0.82rem; margin:0.5rem 0 0 0; line-height:1.6;">
        Pembelian Fresh dominan dengan volume menengah. 
        Cocok untuk restoran kecil atau katering lokal.</p>
    </div>
    """, unsafe_allow_html=True)
elif st.session_state.halaman == "Prediksi":
    st.markdown("""
        <h1  style="font-family:'Syne',sans-serif; font-weight:800;font-size: 2.2rem; 
        color:#FFCCF2">
        Mengenal Datanya
        </h1>  
    """,unsafe_allow_html=True)
    col_form, col_hasil = st.columns([1, 1])
    with col_form:
        st.markdown("""
        <div style="background:#FFCCF2; border-radius:14px; padding:1.25rem;">
        <p style="color:#00033D; font-family:'Syne',sans-serif; 
        font-weight:700; font-size:1rem; margin-bottom:1rem;">📝 Input Data</p>
        </div>
        <style>
        [data-testid="columns"]:first-child > div:first-child {
        background: #FFCCF2;
        color:#00033D;
        border: 1px solid rgba(255,0,127,0.2);
        border-radius: 14px;
        padding: 1.25rem;
        }
                    
                    </style>
        """, unsafe_allow_html=True)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fresh      = st.number_input("🥦 Fresh",      0, 100_000, 1_000)
            milk       = st.number_input("🥛 Milk",       0, 100_000, 1_000)
            grocery    = st.number_input("🛍️ Grocery",    0, 100_000, 1_000)
        with col_f2:
            frozen     = st.number_input("🧊 Frozen",     0, 100_000, 1_000)
            detergents = st.number_input("🧴 Detergents", 0, 100_000, 1_000)
            delicassen = st.number_input("🧀 Delicassen", 0, 100_000, 1_000)

        submitted = st.button("🔮 Prediksi Sekarang", type="primary", 
                              use_container_width=True)

    with col_hasil:
        if submitted:
            # ✅ Urutan sesuai FITUR
            data_baru = np.array([[fresh, milk, grocery, frozen, 
                                   detergents, delicassen]])
            pred = model.predict(data_baru)[0]

            cluster_info = {
                0: {
                    "nama": "🍽️ Restaurant / HoReCa",
                    "warna": "#34C47C",
                    "desc": "Pelanggan ini kemungkinan adalah restoran atau hotel. Pola pembelian didominasi produk Fresh untuk kebutuhan memasak sehari-hari.",
                    "rekomendasi": "Cocok ditawarkan produk Fresh dalam jumlah besar dengan pengiriman rutin."
                },
                1: {
                    "nama": "🏪 Retail Store",
                    "warna": "#4F8EF7",
                    "desc": "Pelanggan ini kemungkinan adalah toko retail atau minimarket. Banyak membeli Grocery, Milk, dan Detergents untuk dijual kembali.",
                    "rekomendasi": "Cocok ditawarkan paket bundling Grocery + Detergents + Milk."
                }
            }

            info = cluster_info[pred]

            st.markdown(f"""
            <div style="background:#FFCCF2;  
            border-radius:14px; padding:1.5rem; margin-bottom:1rem;padding:5rem 3rem 5rem 3rem">
                <p style="color:#00033D; font-family:'Syne',sans-serif; 
                font-weight:800; font-size:1.5rem; margin:0;text-align:center">
                {info['nama']}</p>
                <p style="color:#00033D; font-family:'Syne',sans-serif; 
                font-weight:100; font-size:1rem; margin:0;text-align:center">
                {info['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

            

            st.markdown(f"""
            <div style="background:#FFCCF2;border-radius:12px; padding:1rem; margin-top:1rem;">
                <p style="color:#00033D; font-size:0.85rem; margin:0;">
                💡 <strong>Rekomendasi Bisnis:</strong><br>{info['rekomendasi']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Radar chart perbandingan
            input_vals = [fresh, milk, grocery, frozen, detergents, delicassen]
            df_vis2 = df[FITUR].copy()
            df_vis2["Cluster"] = model.predict(df_vis2.values)
            cluster_mean = df_vis2[df_vis2["Cluster"] == pred][FITUR].mean().round(0).tolist()

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=input_vals + [input_vals[0]],
                theta=FITUR + [FITUR[0]],
                fill='toself',
                name='Input Kamu',
                line_color=info["warna"]
            ))
            fig_r.add_trace(go.Scatterpolar(
                r=cluster_mean + [cluster_mean[0]],
                theta=FITUR + [FITUR[0]],
                fill='toself',
                name='Rata-rata Cluster',
                line_color="rgba(255,255,255,0.3)"
            ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, color="#555")),
                paper_bgcolor="rgba(0,0,0,0)",
                template="plotly_dark",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.25,
                    font=dict(color="#FFCCF2")
                ),
                margin=dict(l=0, r=0, t=10, b=0),
                height=350
            )
            st.plotly_chart(fig_r, use_container_width=True)

        else:
            st.markdown("""
            <div style="background:#FFCCF2; border-radius:14px; padding:3rem; 
            text-align:center;">
                <p style="font-size:2.5rem; margin-bottom:0.5rem;">🔮</p>
                <p style="color:#00033D; font-weight:600;">
                Isi form di sebelah kiri</p>
                <p style="color:#00033D; font-size:0.90rem;">
                Hasil prediksi akan muncul di sini</p>
            </div>
            """, unsafe_allow_html=True)

elif st.session_state.halaman == "code":
    
    st.markdown("""
        <h1 style="font-family:'Syne',sans-serif; font-weight:800;font-size: 2.2rem; color:#FFCCF2">
        Kode Pembuatan Model (Full)
        </h1>  
        <p style="color:#FFCCF2; font-size:0.88rem; margin-bottom:2rem;">
        Berikut adalah keseluruhan kode Python dari Jupyter Notebook yang mencakup Exploratory Data Analysis (EDA), deteksi outlier, evaluasi algoritma, hingga penyimpanan model akhir.
        </p>
    """, unsafe_allow_html=True)

    # ==========================================
    # IMPORT LIBRARY
    # ==========================================
    st.markdown("<h3 style='color:#FFCCF2;'>Import Library</h3>", unsafe_allow_html=True)
    st.code('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import joblib''', language='python')

    # ==========================================
    # 1. LOAD DATA & EKSPLORASI AWAL
    # ==========================================
    st.markdown("<h3 style='color:#FFCCF2;'>1. Load Data & Eksplorasi Awal</h3>", unsafe_allow_html=True)
    st.code('''df = pd.read_csv("Wholesale customers data.csv")
print(df.head())''', language='python')
    
    st.code('print(df.tail())', language='python')
    st.code('print(df.columns)', language='python')
    st.code("print(df['Region'].unique())", language='python')
    st.code('print(df.sample(5))', language='python')

    # ==========================================
    # 2. VISUALISASI DISTRIBUSI DATA
    # ==========================================
    st.markdown("<h3 style='color:#FFCCF2;'>2. Visualisasi Distribusi Data</h3>", unsafe_allow_html=True)
    st.code('''sns.histplot(df["Channel"], kde=True)
plt.title("Channel")
plt.show()''', language='python')

    st.code('''sns.histplot(df['Fresh'], kde=True)
plt.title("Fresh")
plt.show()''', language='python')

    st.code('''sns.histplot(df['Milk'], kde=True)
plt.title("Milk")
plt.show()''', language='python')

    st.code('''sns.histplot(df['Grocery'], kde=True)
plt.title("Grocery")
plt.show()''', language='python')

    st.code('''sns.histplot(df['Frozen'], kde=True)
plt.title("Frozen")
plt.show()''', language='python')

    # ==========================================
    # 3. KORELASI ANTAR FITUR (HEATMAP)
    # ==========================================
    st.markdown("<h3 style='color:#FFCCF2;'>3. Korelasi Antar Fitur (Heatmap)</h3>", unsafe_allow_html=True)
    st.code('''corr = df[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, fmt=".2f", cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.show()''', language='python')

    # ==========================================
    # 4. DETEKSI OUTLIER
    # ==========================================
    st.markdown("<h3 style='color:#FFCCF2;'>4. Deteksi Outlier</h3>", unsafe_allow_html=True)
    st.code('''fig, axes = plt.subplots(2, 3, figsize=(14, 8))
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

for i, feat in enumerate(features):
    ax = axes[i//3][i%3]
    ax.boxplot(df[feat])
    ax.set_title(feat)
    ax.grid(alpha=0.3)

plt.suptitle("Boxplot per Fitur — Deteksi Outlier", fontweight='bold')
plt.tight_layout()
plt.show()''', language='python')

    st.code('''# Hitung jumlah outlier per fitur pakai IQR
print("\\n=== Jumlah Outlier per Fitur (metode IQR) ===")
for feat in features:
    Q1 = df[feat].quantile(0.25)
    Q3 = df[feat].quantile(0.75)
    IQR = Q3 - Q1
    outlier = df[(df[feat] < Q1 - 1.5*IQR) | (df[feat] > Q3 + 1.5*IQR)]
    print(f"{feat:20s}: {len(outlier)} outlier")''', language='python')

    # ==========================================
    # 5. ELBOW METHOD UNTUK MENENTUKAN K
    # ==========================================
    st.markdown("<h3 style='color:#FFCCF2;'>5. Elbow Method</h3>", unsafe_allow_html=True)
    st.code('''X = df[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]
X_array = X.to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_array)

inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.style.use("dark_background")
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o', color='purple')
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method untuk Menentukan K Optimal")
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.show()''', language='python')

    # ==========================================
    # 6. PERBANDINGAN MODEL CLUSTERING
    # ==========================================
    st.markdown("<h3 style='color:#FFCCF2;'>6. Perbandingan Model Clustering</h3>", unsafe_allow_html=True)
    st.code('''# K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
label_kmeans = kmeans.fit_predict(X_scaled)
sil_kmeans = silhouette_score(X_scaled, label_kmeans)

# Agglomerative
agglo = AgglomerativeClustering(n_clusters=3)
label_agglo = agglo.fit_predict(X_scaled)
sil_agglo = silhouette_score(X_scaled, label_agglo)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
label_dbscan = dbscan.fit_predict(X_scaled)
sil_dbscan = silhouette_score(X_scaled, label_dbscan)

print("\\n=== Perbandingan Silhouette Score ===")
print(f"K-Means              : {sil_kmeans:.4f}")
print(f"Agglomerative        : {sil_agglo:.4f}")
print(f"DBSCAN               : {sil_dbscan:.4f}")''', language='python')

    st.code('''# Cek Rata-Rata Karakteristik per Metode
df['cluster_kmeans'] = label_kmeans
print("\\nK-Means Rata-Rata:")
print(df.groupby('cluster_kmeans')[features].mean().round(0))

df['cluster_agglo'] = label_agglo
print("\\nAgglomerative Rata-Rata:")
print(df.groupby('cluster_agglo')[features].mean().round(0))

df['cluster_dbscan'] = label_dbscan
print("\\nDBSCAN Rata-Rata:")
print(df.groupby('cluster_dbscan')[features].mean().round(0))''', language='python')

    # ==========================================
    # 7. PEMBUATAN PIPELINE FINAL
    # ==========================================
    st.markdown("<h3 style='color:#FFCCF2;'>7. Pembuatan Pipeline Final (K-Means)</h3>", unsafe_allow_html=True)
    st.code('''model = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=2, random_state=42, n_init=10))
])

# Fit pipeline
model.fit(X_array)
labels = model.named_steps['kmeans'].labels_
df['Cluster'] = labels''', language='python')

    st.code('''# Visualisasi Hasil Clustering
plt.style.use('dark_background')
plt.figure(figsize=(9, 6))
scatter = plt.scatter( 
    df['Fresh'],
    df['Grocery'], 
    c=labels,
    cmap='viridis',
    s=80,
    alpha=0.85
)
plt.xlabel("Fresh")
plt.ylabel("Grocery")
plt.title("Hasil Clustering Pelanggan (K=3)")
plt.colorbar(scatter, label='Cluster')
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()''', language='python')

    st.code('''# Evaluasi Silhouette Pipeline Akhir
X_scaled_final = model.named_steps['scaler'].transform(X_array)
sil_score = silhouette_score(X_scaled_final, labels)
print(f"\\nSilhouette Score Final: {sil_score:.4f}")''', language='python')

    # ==========================================
    # 8. SIMPAN MODEL
    # ==========================================
    st.markdown("<h3 style='color:#FFCCF2;'>8. Simpan Model</h3>", unsafe_allow_html=True)
    st.code('''joblib.dump(model, "cluster3.joblib")
print("Model berhasil disimpan sebagai cluster3.joblib")''', language='python')
    
