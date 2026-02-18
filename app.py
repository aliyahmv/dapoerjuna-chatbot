import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from agent import build_agent
from memory import memory, remember
from tools import df as df_tools
import random

# ==================== PAGE CONFIG & DATA LOADING ====================
st.set_page_config("DAPOERJUNA", "🍳", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("df_resep_cleaned.csv")

df = load_data()

# ==================== SIDEBAR ====================
st.sidebar.title("")
page = st.sidebar.radio("Pilih Halaman:", ["🤖 Chatbot", "📊 Dashboard Analisis", "📚 Informasi UMKM"])
st.sidebar.markdown("---")

if page == "🤖 Chatbot":
    st.sidebar.header("Mood Chef Juna")
    chef_mood = st.sidebar.selectbox("Pilih Mood:", ["Chef Juna Mengayomi 😇", "Chef Juna Galak 😈", "Random Mood 🎭"])
    if st.sidebar.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.session_state.memory = memory
        st.rerun()

elif page == "📊 Dashboard Analisis":
    st.sidebar.header("🔍 Filter Dashboard")
    selected_difficulty = st.sidebar.selectbox("Pilih Tingkat Kesulitan:", ["Semua Tingkat", "Mudah", "Sedang", "Cukup Rumit"], key="dashboard_filter")

# ==================== MAIN CONTENT ====================
if page == "🤖 Chatbot":
    st.markdown("""
    <div style='font-size:40px;font-weight:700;margin-bottom:0.3em'>
    🍳 DAPOERJUNA – Masakan Indonesia Gak Perlu Ribet
    </div>
    <p style='font-size:22px;margin-top:-10px'>
    👨‍🍳 Saya <strong>Chef Juna</strong>. Gak bisa masak? Sini gue marahin—eh, ajarin maksudnya.
    </p><p style='font-size:17px;color:#444'>
    Tanya resep biar masakan enak, bingung mau masak apa, atau biar dimarahin Chef Juna? Semuanya bisa di sini!
    <br><br>
    📊 <strong>Analisis tren resep favorit untuk inspirasi menu dan ide kuliner kreatif serta membantu keputusan bisnis kulinermu!</strong><br>
    📚 <strong>Dilengkapi informasi tambahan untuk UMKM kuliner.</strong>
    """, unsafe_allow_html=True)

    with st.expander("🤯 Bingung? Coba tanya kayak gini dulu:"):
        st.markdown("""
        * _"Gimana sih cara bikin ayam geprek yang kriuk di luar, juicy di dalam?"_
        * _"Berikan saya resep yang paling banyak disukai."_
        * _"Chef Juna, aku mau resep makanan yang mudah dan cocok untuk diet vegan."_
        """)
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_recipes_blob" not in st.session_state:
        st.session_state.last_recipes_blob = ""

    chat_container = st.container(height=500)
    with chat_container:
        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar="👨‍🍳" if m["role"] == "assistant" else "👨"):
                st.markdown(m["content"])

    if prompt := st.chat_input("Mau masak apa hari ini? Ketik aja…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            for m in st.session_state.messages:
                with st.chat_message(m["role"], avatar="👨‍🍳" if m["role"] == "assistant" else "👨"):
                    st.markdown(m["content"])

        with st.chat_message("assistant", avatar="👨‍🍳"):
            with st.spinner("Chef Juna sedang mikir…"):
                remember("user", prompt)
                agent = build_agent()
                mood_map = {
                    "Chef Juna Mengayomi 😇": "baik",
                    "Chef Juna Galak 😈": "galak",
                    "Random Mood 🎭": "random"
                }
                init_state = {
                    "messages": [f"User: {prompt}"],
                    "steps": 0,
                    "attitude": mood_map[chef_mood],
                }
                out = agent.invoke(init_state, config={"max_loops": 6})
                reply = out["messages"][-1].split("</tool>")[-1].strip()
                remember("ai", reply)

                if "Langkah:" in reply:
                    st.session_state.last_recipes_blob = reply
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

elif page == "📊 Dashboard Analisis":
    st.title("📊 Dashboard Analisis Resep Masakan")
    if selected_difficulty == "Semua Tingkat":
        df_filtered = df.copy()
    else:
        df_filtered = df[df["difficulty"] == selected_difficulty]

    col1, col2 = st.columns(2)
    col1.metric("Total Resep", len(df_filtered))
    col2.metric("Total Kategori", df_filtered["category"].nunique())
    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📌 Tingkat Kesulitan Resep")
        difficulty_counts = df_filtered["difficulty"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.pie(difficulty_counts, labels=difficulty_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(difficulty_counts)))
        ax1.axis("equal")
        st.pyplot(fig1)

    with col4:
        st.subheader("🥦 Preferensi Konsumsi")
        diet_counts = df_filtered["diet"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.pie(diet_counts, labels=diet_counts.index, autopct="%1.1f%%", startangle=140, colors=["#ffcc99", "#c2c2f0"])
        ax2.axis("equal")
        st.pyplot(fig2)

    st.markdown("---")
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("🍽️ Kategori Resep")
        kategori_counts = df_filtered["category"].value_counts().nlargest(10).sort_values(ascending=True)
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        kategori_counts.plot(kind='barh', color=sns.light_palette("navy", n_colors=len(kategori_counts), reverse=False), ax=ax3)
        ax3.set_xlabel("Jumlah Resep")
        ax3.set_ylabel("Kategori")
        st.pyplot(fig3)

    with col6:
        st.subheader("🍛 Jenis Makanan")
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        df_filtered["meal_weight"].value_counts().plot(kind="bar", color="mediumseagreen", ax=ax4)
        ax4.set_xlabel("Jenis Makanan")
        ax4.set_ylabel("Jumlah Resep")
        plt.xticks(rotation=45)
        st.pyplot(fig4)

    st.markdown("---")
    st.subheader("🔥 Top 3 Resep Terpopuler")
    top3 = df_filtered.sort_values(by='loves', ascending=False).head(3)
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='loves', y='title', data=top3, palette='magma', ax=ax5, orient='h')
    ax5.set_xlabel("Jumlah Loves", fontsize=12)
    ax5.set_ylabel("Nama Resep", fontsize=12)
    st.pyplot(fig5)

elif page == "📚 Informasi UMKM":
    st.title("📚 Informasi UMKM Kuliner")

    st.subheader("🚀 Tips & Trik Usaha Kuliner")
    st.markdown("""
    - [Cara Menentukan Harga Jual Makanan untuk UMKM](https://pageid.borong.com/blog/cara-menentukan-harga-jual-makanan/)
    - [Strategi Pemasaran Digital untuk Bisnis Kuliner yang Sukses](https://news.ralali.com/strategi-pemasaran-digital-untuk-bisnis-kuliner-yang-sukses/)
    - [8 Tips Menjaga Konsistensi Kualitas Produk dalam Bisnis Kuliner](https://sokoguru.id/soko%20kreatif/8-tips-menjaga-konsistensi-kualitas-produk-dalam-bisnis-kuliner-agar-pelanggan-loyal)
    """)

    st.subheader("🛒 Marketplace untuk Jualan")
    st.markdown("""
    - [GoFood](https://gofood.co.id/)
    - [GrabFood](https://grab.com/id/)
    - [ShopeeFood](https://shopee.co.id/m/shopeefood)
    - [Tokopedia](https://www.tokopedia.com/)
    """)

    st.subheader("📰 Tren Kuliner Lokal Terbaru")
    st.markdown("""
    - [Tren Kuliner Indonesia 2025: Makanan yang Sedang Naik Daun](https://www.cheon.co.id/tren-kuliner-indonesia-2025-makanan-yang-sedang-naik-daun/)
    - [Intip Tren Kuliner Anak Muda Indonesia: Hobi Jajan Namun Tetap Hemat](https://goodstats.id/article/intip-tren-kuliner-anak-muda-yang-hobi-jajan-namun-tetap-hemat-mmp99)
    - [5 Jajanan Tradisional yang Kini Kembali Populer di Kalangan Anak Muda](https://radarpena.disway.id/read/205697/5-jajanan-tradisional-yang-kini-kembali-populer-di-kalangan-anak-muda-bikin-nostalgia)
    """)
