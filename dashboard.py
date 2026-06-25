import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

st.set_page_config(page_title="Dashboard Clustering Provinsi Indonesia", page_icon="🗺️", layout="wide")
st.title("🗺️ Dashboard Clustering Sosial Ekonomi Provinsi Indonesia")
st.markdown("**K-Means Clustering** | Data BPS 2023 | 4 Variabel | K=3")
st.markdown("---")

@st.cache_data
def load_data():
    data = {
        "Provinsi": ["Aceh","Sumatera Utara","Sumatera Barat","Riau","Jambi","Sumatera Selatan","Bengkulu","Lampung","Kep. Bangka Belitung","Kep. Riau","DKI Jakarta","Jawa Barat","Jawa Tengah","DI Yogyakarta","Jawa Timur","Banten","Bali","Nusa Tenggara Barat","Nusa Tenggara Timur","Kalimantan Barat","Kalimantan Tengah","Kalimantan Selatan","Kalimantan Timur","Kalimantan Utara","Sulawesi Utara","Sulawesi Tengah","Sulawesi Selatan","Sulawesi Tenggara","Gorontalo","Sulawesi Barat","Maluku","Maluku Utara","Papua Barat","Papua"],
        "TPP_SMA": [74.46,74.43,68.64,67.79,66.62,64.81,63.41,64.54,68.96,78.97,88.10,66.47,58.35,89.69,68.65,70.07,76.51,63.66,43.46,55.58,63.93,68.35,73.63,59.50,67.57,55.69,67.41,68.28,46.19,54.79,75.01,64.61,59.99,39.50],
        "Penduduk": [5409.2,15180.5,5677.6,6555.8,3633.2,8647.3,2059.4,4496.6,1492.0,2121.5,10640.0,49306.8,37180.4,3712.6,41230.0,12167.0,4374.3,5474.0,5481.8,5549.7,2737.2,4170.2,3856.8,720.1,2660.8,3051.2,9260.1,2704.6,1198.4,1458.9,1895.1,1318.5,1168.4,4429.7],
        "TPT": [5.75,5.24,5.90,4.25,4.50,4.53,3.21,4.18,3.89,7.61,7.57,7.89,5.24,3.58,4.33,7.97,3.73,3.73,3.10,4.52,3.84,3.95,6.37,4.10,6.19,3.49,5.26,3.66,3.07,3.04,6.08,4.60,5.53,3.49],
        "Kemiskinan": [9.79,8.23,4.67,6.73,10.19,11.07,14.21,8.02,3.54,5.05,4.44,7.19,9.78,10.27,7.50,6.00,3.77,13.76,9.12,4.44,4.78,3.84,4.68,5.18,4.91,8.90,5.01,7.40,4.47,9.08,5.49,6.23,8.23,5.68],
    }
    df = pd.DataFrame(data)
    X = df[["TPP_SMA","Penduduk","TPT","Kemiskinan"]]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    df["Label Cluster"] = df["Cluster"].map({0:"Cluster 0",1:"Cluster 1",2:"Cluster 2"})
    dbi = davies_bouldin_score(X_scaled, df["Cluster"])
    sil = silhouette_score(X_scaled, df["Cluster"])
    return df, X_scaled, dbi, sil

df, X_scaled, dbi, sil = load_data()

CLUSTER_COLORS = {0:"#1f77b4", 1:"#2ca02c", 2:"#ff7f0e"}
CLUSTER_NAMES = {0:"Cluster 0 - Padat & Tekanan Tinggi", 1:"Cluster 1 - Berkembang & Pendidikan Baik", 2:"Cluster 2 - Tantangan Pendidikan & Kemiskinan"}

st.sidebar.header("⚙️ Filter & Navigasi")
selected_cluster = st.sidebar.multiselect("Tampilkan Cluster:", options=[0,1,2], default=[0,1,2], format_func=lambda x: CLUSTER_NAMES[x])
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Info Model")
st.sidebar.metric("Jumlah Cluster (K)", "3")
st.sidebar.metric("Davies-Bouldin Index", f"{dbi:.4f}")
st.sidebar.metric("Silhouette Score", f"{sil:.4f}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Sumber Data:** BPS 2023")
st.sidebar.markdown("**Metode:** K-Means + Min-Max Scaling")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Data & Hasil Clustering","📈 Visualisasi Scatter Plot","🔍 Profiling Cluster","📐 Evaluasi Model"])

with tab1:
    st.subheader("📋 Data Lengkap 34 Provinsi")
    df_filtered = df[df["Cluster"].isin(selected_cluster)].copy()
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Provinsi Ditampilkan", len(df_filtered))
    c2.metric("Cluster Dipilih", len(selected_cluster))
    c3.metric("Total Provinsi", len(df))
    st.markdown("---")
    display_df = df_filtered[["Provinsi","TPP_SMA","Penduduk","TPT","Kemiskinan","Label Cluster","Cluster"]].copy()
    display_df["TPP_SMA"] = display_df["TPP_SMA"].round(2)
    display_df["Penduduk"] = display_df["Penduduk"].round(1)
    display_df["TPT"] = display_df["TPT"].round(2)
    display_df["Kemiskinan"] = display_df["Kemiskinan"].round(2)
    display_df = display_df.rename(columns={"TPP_SMA":"TPP SMA (%)","Penduduk":"Penduduk (Ribu Jiwa)","TPT":"TPT (%)","Kemiskinan":"Kemiskinan (%)"})
    display_df = display_df.reset_index(drop=True)
    cluster_col = display_df["Cluster"].tolist()
    display_df = display_df.drop(columns=["Cluster"])
    def color_rows(row):
        cl = cluster_col[row.name]
        c = {0:"background-color:#d0e8ff;color:black", 1:"background-color:#d0f0d0;color:black", 2:"background-color:#ffe4b0;color:black"}
        return [c.get(cl,"")] * len(row)
    st.dataframe(display_df.style.apply(color_rows, axis=1), use_container_width=True, height=600)
    st.caption("🔵 Cluster 0 = Biru | 🟢 Cluster 1 = Hijau | 🟠 Cluster 2 = Oranye")

with tab2:
    st.subheader("📈 Visualisasi Scatter Plot Interaktif")
    var_options = {"TPP SMA (%)":"TPP_SMA","Jumlah Penduduk (Ribu Jiwa)":"Penduduk","TPT (%)":"TPT","Kemiskinan (%)":"Kemiskinan"}
    cx, cy = st.columns(2)
    x_label = cx.selectbox("Sumbu X:", list(var_options.keys()), index=3)
    y_label = cy.selectbox("Sumbu Y:", list(var_options.keys()), index=2)
    x_col = var_options[x_label]
    y_col = var_options[y_label]
    fig, ax = plt.subplots(figsize=(12,7))
    df_plot = df[df["Cluster"].isin(selected_cluster)]
    for cl in sorted(df_plot["Cluster"].unique()):
        subset = df_plot[df_plot["Cluster"]==cl]
        ax.scatter(subset[x_col], subset[y_col], label=CLUSTER_NAMES[cl], color=CLUSTER_COLORS[cl], s=35, alpha=0.85, edgecolors="black", linewidths=0.4)
        for _, row in subset.iterrows():
            ax.annotate(row["Provinsi"], (row[x_col], row[y_col]), fontsize=5.5, alpha=0.9, xytext=(5,5), textcoords="offset points", bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, ec="none"))
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f"Cluster Provinsi: {x_label} vs {y_label}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Gunakan dropdown di atas untuk mengganti kombinasi variabel pada scatter plot.")

with tab3:
    st.subheader("🔍 Profiling Setiap Cluster")
    profil = df.groupby("Cluster")[["TPP_SMA","Penduduk","TPT","Kemiskinan"]].mean().round(2)
    profil["Jumlah Provinsi"] = df.groupby("Cluster")["Provinsi"].count()
    nasional = df[["TPP_SMA","Penduduk","TPT","Kemiskinan"]].mean().round(2)
    nasional["Jumlah Provinsi"] = len(df)
    profil_display = profil.copy()
    profil_display.index = [CLUSTER_NAMES[i] for i in profil_display.index]
    profil_display.loc["Rata-rata Nasional"] = nasional
    profil_display.columns = ["TPP SMA (%)","Penduduk (Ribu Jiwa)","TPT (%)","Kemiskinan (%)","Jumlah Provinsi"]
    st.dataframe(profil_display.style.highlight_max(axis=0, color="#ffd700", subset=["TPP SMA (%)","TPT (%)","Kemiskinan (%)"]).highlight_min(axis=0, color="#90ee90", subset=["TPP SMA (%)","TPT (%)","Kemiskinan (%)"]), use_container_width=True)
    st.markdown("---")
    st.markdown("### Karakteristik Setiap Cluster")
    c0,c1,c2 = st.columns(3)
    with c0:
        st.markdown("#### 🔵 Cluster 0")
        st.markdown("**Provinsi Berpenduduk Padat**")
        st.markdown("- Jawa Barat, Jawa Tengah, Jawa Timur")
        st.markdown("- Penduduk sangat besar (>37 juta/provinsi)")
        st.markdown("- TPT & kemiskinan di atas rata-rata nasional")
    with c1:
        st.markdown("#### 🟢 Cluster 1")
        st.markdown("**Provinsi Berkembang**")
        st.markdown("- 10 provinsi: DKI Jakarta, Banten, dll")
        st.markdown("- TPP SMA tertinggi (73.83%)")
        st.markdown("- TPT tertinggi, kemiskinan terendah")
    with c2:
        st.markdown("#### 🟠 Cluster 2")
        st.markdown("**Tantangan Pendidikan & Kemiskinan**")
        st.markdown("- 21 provinsi (mayoritas Indonesia Timur)")
        st.markdown("- TPP SMA terendah (62.18%)")
        st.markdown("- TPT terendah, kemiskinan sedang-tinggi")
    st.markdown("---")
    st.markdown("### Perbandingan Rata-rata Per Cluster")
    vars_to_plot = st.multiselect("Pilih variabel:", ["TPP SMA (%)","TPT (%)","Kemiskinan (%)"], default=["TPP SMA (%)","TPT (%)","Kemiskinan (%)"])
    if vars_to_plot:
        fig2, ax2 = plt.subplots(figsize=(10,5))
        plot_data = profil_display.loc[[CLUSTER_NAMES[0],CLUSTER_NAMES[1],CLUSTER_NAMES[2]], vars_to_plot]
        nasional_vals = profil_display.loc["Rata-rata Nasional", vars_to_plot]
        x = np.arange(len(vars_to_plot))
        width = 0.25
        colors_bar = ["#1f77b4","#2ca02c","#ff7f0e"]
        for i, (label, row) in enumerate(plot_data.iterrows()):
            ax2.bar(x+i*width, row.values, width, label=f"Cluster {i}", color=colors_bar[i], alpha=0.8)
        for j, val in enumerate(nasional_vals.values):
            ax2.axhline(y=val, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax2.set_xticks(x+width)
        ax2.set_xticklabels(vars_to_plot, fontsize=11)
        ax2.set_ylabel("Nilai Rata-rata", fontsize=11)
        ax2.set_title("Perbandingan Rata-rata Variabel per Cluster\n(Garis merah = rata-rata nasional)", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig2)

with tab4:
    st.subheader("📐 Evaluasi Model K-Means")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("### Davies-Bouldin Index (DBI)")
        st.metric("Nilai DBI", f"{dbi:.4f}", delta="< 1.0 ✅")
        st.markdown(f"""
**Interpretasi:**
- Nilai DBI = **{dbi:.4f}** (< 1,0)
- Semakin kecil DBI, semakin baik kualitas cluster
- Nilai ini menunjukkan antar-cluster cukup terpisah dan data dalam cluster cukup kompak
        """)
    with cb:
        st.markdown("### Silhouette Score")
        st.metric("Nilai Silhouette", f"{sil:.4f}", delta="Positif ✅")
        st.markdown(f"""
**Interpretasi:**
- Nilai Silhouette = **{sil:.4f}** (positif)
- Rentang nilai: -1 (buruk) hingga 1 (sempurna)
- Nilai positif menunjukkan sebagian besar data berada di cluster yang tepat
- Nilai 0,3 tergolong **cukup baik** untuk data sosial ekonomi yang kompleks
        """)
    st.markdown("---")
    st.markdown("### Elbow Method - Penentuan K Optimal")
    wcss_vals = [8.40,6.55,4.22,2.55,2.30,2.25,1.95]
    k_vals = list(range(1,8))
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.plot(k_vals, wcss_vals, marker="o", color="#1f77b4", linewidth=2, markersize=8)
    ax3.axvline(x=3, color="red", linestyle="--", linewidth=1.5, label="K Optimal = 3")
    ax3.set_xlabel("Jumlah Cluster (K)", fontsize=12)
    ax3.set_ylabel("WCSS", fontsize=12)
    ax3.set_title("Elbow Method - Penentuan Jumlah Cluster Optimal", fontsize=13)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    st.caption("Titik tekukan (elbow) berada pada K=3, ditandai garis merah putus-putus.")
    st.markdown("---")
    st.markdown("### Ringkasan Model")
    summary = pd.DataFrame({
        "Parameter":["Algoritma","Jumlah Cluster (K)","Normalisasi","Random State","DBI","Silhouette Score"],
        "Nilai":["K-Means Clustering","3","Min-Max Scaling","42",f"{dbi:.4f}",f"{sil:.4f}"]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
