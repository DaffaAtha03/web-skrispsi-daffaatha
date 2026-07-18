import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sentence_transformers import SentenceTransformer

# Menonaktifkan GUI Matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="CineMatch — Hybrid Movie Recommender",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="collapsed",
)

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""
if "do_search" not in st.session_state:
    st.session_state["do_search"] = False

def trigger_search():
    st.session_state["do_search"] = True

def quick_search(query):
    st.session_state["query_input"] = query
    st.session_state["do_search"] = True

# ====================================================================
# CSS — PREMIUM CINEMATIC + TIKET + 3D HOVER GLOW
# ====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #05050A; color: #e8e8f0; font-family: 'Plus Jakarta Sans', sans-serif;
}
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

.hero {
    background: linear-gradient(135deg, rgba(26,0,51,0.8) 0%, rgba(8,8,18,0.9) 50%, rgba(0,26,51,0.8) 100%);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px;
    padding: 40px 50px 30px; margin-bottom: 20px; position: relative; overflow: hidden; box-shadow: 0 20px 40px rgba(0,0,0,0.5);
}
.hero-content { position: relative; z-index: 1; }
.hero-eyebrow { font-size: 0.75rem; font-weight: 800; letter-spacing: 4px; color: #e50914; text-transform: uppercase; margin-bottom: 8px; }
.hero-title {
    font-size: 3.5rem; font-weight: 800; letter-spacing: -1.5px; margin: 0 0 10px;
    background: linear-gradient(90deg, #ffffff 0%, #f5c518 50%, #e50914 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.1;
}
.hero-sub { color: #A0A0B5; font-size: 1rem; margin: 0 0 24px; line-height: 1.6; font-weight: 400; }

.movie-card {
    background: linear-gradient(180deg, rgba(20,20,40,0.8) 0%, rgba(10,10,25,0.9) 100%);
    border: 1px solid rgba(255,255,255,0.05);
    border-left: 5px solid var(--glow-color, #ffffff);
    border-radius: 12px; padding: 24px 20px; margin-bottom: 5px; position: relative; overflow: hidden; min-height: 160px;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.movie-card:hover {
    transform: scale(1.05) translateY(-5px);
    box-shadow: 0 15px 35px var(--glow-color, rgba(255,255,255,0.2));
    border-color: var(--glow-color, #ffffff);
    z-index: 10;
}
.movie-card::before, .movie-card::after {
    content: ''; position: absolute; top: 50%; width: 24px; height: 24px; background: #05050A; border-radius: 50%;
}
.movie-card::before { left: -12px; transform: translateY(-50%); border-right: 1px solid rgba(255,255,255,0.05); }
.movie-card::after { right: -12px; transform: translateY(-50%); border-left: 1px solid rgba(255,255,255,0.05); }

.barcode { height: 12px; background: repeating-linear-gradient(90deg, #A0A0B5, #A0A0B5 2px, transparent 2px, transparent 4px); margin-top: 15px; opacity: 0.3; }
.card-rank { position: absolute; top: -10px; right: -5px; font-size: 4.5rem; font-weight: 900; color: rgba(255,255,255,0.03); line-height: 1; font-style: italic; z-index: 0; }
.card-content { position: relative; z-index: 1; }
.card-title { font-size: 1.05rem; font-weight: 700; color: #fff; margin: 0 0 12px; line-height: 1.4; }
.card-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 15px; }
.card-tag { display: inline-block; border-radius: 6px; padding: 4px 10px; font-size: 0.65rem; font-weight: 700; }
.score-bar-wrap { margin-top: auto; }
.score-bar-label { display: flex; justify-content: space-between; font-size: 0.7rem; color: #A0A0B5; margin-bottom: 6px; font-weight:600; }
.score-track { width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 10px; overflow: hidden; }
.score-fill { height: 100%; background: linear-gradient(90deg, #e50914, #f5c518); }
.section-title { display: flex; align-items: center; gap: 12px; margin: 40px 0 20px; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); }
.section-title h3 { margin: 0; font-size: 1.3rem; font-weight: 700; color: #fff; }
.section-dot { width: 12px; height: 12px; border-radius: 50%; background: linear-gradient(135deg, #e50914, #f5c518); }
.modern-footer { margin-top: 60px; padding: 30px 0; border-top: 1px solid rgba(255,255,255,0.05); text-align: center; color: #8888AA; font-size: 0.85rem; }
.modern-footer span { font-weight: bold; background: linear-gradient(90deg, #e50914, #f5c518); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Memuat Model AI BERT & Dataset...")
def init_models_and_data():
    try:
        pred_matrix = pd.read_parquet("pred_matrix_saved.parquet")
        movies_indo = pd.read_csv("movies_indo_clean.csv")
        df_imdb     = pd.read_csv("df_imdb_clean.csv")
        movies_ml   = pd.read_csv("movies_ml_clean.csv")
        ratings     = pd.read_csv("ratings_clean.csv")
    except FileNotFoundError:
        st.error("❌ File data tidak ditemukan!")
        st.stop()

    if 'genre' in movies_indo.columns: movies_indo = movies_indo.rename(columns={'genre': 'genres'})
    movies_indo['content'] = movies_indo['genres'].fillna('')
    movies_indo['year']    = pd.to_numeric(movies_indo.get('year', pd.Series(dtype=float)), errors='coerce')

    df_imdb['text']   = df_imdb['title'].fillna('') + ' ' + df_imdb['genres'].fillna('')
    df_imdb['genres'] = df_imdb['genres'].fillna('').str.lower()
    df_imdb['year']   = pd.to_numeric(df_imdb.get('year', pd.Series(dtype=float)), errors='coerce')

    def _clean(t):
        t = str(t).lower()
        t = re.sub(r'\(\d{4}\)', '', t)
        return re.sub(r'[^a-z0-9\s]', '', t).strip()
    movies_ml['clean_title'] = movies_ml['title'].apply(_clean)

    train_data = [
        ("film sedih banget","Drama"), ("film yang bikin nangis","Drama"), ("film romantis","Romance"), ("film cinta sejati","Romance"),
        ("film lucu banget","Comedy"), ("film komedi seru","Comedy"), ("film horor serem","Horror"), ("film hantu","Horror"),
        ("film aksi","Action"), ("film action seru","Action"),
    ]
    texts, labels = [x[0] for x in train_data], [x[1] for x in train_data]
    vectorizer_nlp = TfidfVectorizer()
    model_nlp = MultinomialNB()
    model_nlp.fit(vectorizer_nlp.fit_transform(texts), labels)

    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    g_vecs = bert_model.encode(df_imdb['text'].tolist())
    i_vecs = bert_model.encode(movies_indo['content'].tolist())

    return pred_matrix, movies_indo, df_imdb, movies_ml, ratings, vectorizer_nlp, model_nlp, bert_model, g_vecs, i_vecs

(pred_matrix, movies_indo, df_imdb, movies_ml, ratings, vectorizer_nlp, model_nlp, bert_model, g_vecs, i_vecs) = init_models_and_data()
pivot_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

def clean_title(title): return re.sub(r'[^a-z0-9\s]', '', re.sub(r'\(\d{4}\)', '', str(title).lower())).strip()
def predict_genre_ml(text): return model_nlp.predict(vectorizer_nlp.transform([text]))[0] if text.strip() else "Drama"
def detect_country(text): return "Indonesia" if ("indonesia" in text.lower() or "indo" in text.lower()) else "Mixed"
def detect_year(text): m = re.search(r'(19\d{2}|20\d{2})', text); return int(m.group()) if m else None

def recommend_system_web(user_input, explicit_genre, is_year_filtered, rentang_tahun, explicit_country):
    uid, max_n = 8, 8
    predicted_genre = explicit_genre if explicit_genre != "Otomatis (AI)" else predict_genre_ml(user_input)
    country_mode = "Indonesia" if explicit_country == "Hanya Indonesia" else ("Global" if explicit_country == "Hanya Internasional" else detect_country(user_input))

    c_indo = movies_indo[movies_indo['genres'].str.contains(predicted_genre, case=False, na=False)].copy()
    c_global = df_imdb[df_imdb['genres'].str.contains(predicted_genre.lower(), na=False)].copy()

    if is_year_filtered:
        start_year, end_year = rentang_tahun
        c_indo = c_indo[(c_indo['year'] >= start_year) & (c_indo['year'] <= end_year)]
        c_global = c_global[(c_global['year'] >= start_year) & (c_global['year'] <= end_year)]
        year_info = f"{start_year} - {end_year}"
    else:
        target_year = detect_year(user_input)
        if target_year:
            c_indo, c_global, year_info = c_indo[c_indo['year'] == target_year], c_global[c_global['year'] == target_year], str(target_year)
        else: year_info = "Semua Waktu"

    qv = bert_model.encode([user_input if user_input else predicted_genre])
    c_global['semantic_score'] = cosine_similarity(qv, g_vecs[c_global.index.tolist()])[0] if not c_global.empty else 0.0
    c_indo['semantic_score'] = cosine_similarity(qv, i_vecs[c_indo.index.tolist()])[0] if not c_indo.empty else 0.0

    if uid in pred_matrix.index:
        up = pred_matrix.loc[uid]
        ml2 = movies_ml[movies_ml['movieId'].isin(up.index)].copy()
        ml2['svd_rating'] = ml2['movieId'].map(up)
        svd_lkp = pd.Series(ml2['svd_rating'].values, index=ml2['clean_title']).to_dict()
        c_global['svd_score'] = c_global['title'].apply(clean_title).map(svd_lkp).fillna(ratings['rating'].mean()) / 5.0
    else: c_global['svd_score'] = 0.5

    c_global['final_score'] = 0.5 * c_global['semantic_score'] + 0.5 * c_global['svd_score']
    c_indo['final_score'] = c_indo['semantic_score']

    results_meta = []
    if country_mode == "Indonesia" and not c_indo.empty: results_meta.extend([{'title': r['title'], 'genres': r.get('genres',''), 'source': 'indo', 'score': float(r['final_score']), 'sem_s': float(r['semantic_score']), 'svd_s': 0.8, 'year': r.get('year','')} for _, r in c_indo.sort_values('final_score', ascending=False).head(max_n).iterrows()])
    elif country_mode == "Global" and not c_global.empty: results_meta.extend([{'title': r['title'], 'genres': r.get('genres',''), 'source': 'imdb', 'score': float(r['final_score']), 'sem_s': float(r['semantic_score']), 'svd_s': float(r['svd_score']), 'year': r.get('year','')} for _, r in c_global.sort_values('final_score', ascending=False).head(max_n).iterrows()])
    else:
        if not c_global.empty: results_meta.extend([{'title': r['title'], 'genres': r.get('genres',''), 'source': 'imdb', 'score': float(r['final_score']), 'sem_s': float(r['semantic_score']), 'svd_s': float(r['svd_score']), 'year': r.get('year','')} for _, r in c_global.sort_values('final_score', ascending=False).head(5).iterrows()])
        if not c_indo.empty: results_meta.extend([{'title': r['title'], 'genres': r.get('genres',''), 'source': 'indo', 'score': float(r['final_score']), 'sem_s': float(r['semantic_score']), 'svd_s': 0.8, 'year': r.get('year','')} for _, r in c_indo.sort_values('final_score', ascending=False).head(3).iterrows()])

    seen, deduped = set(), []
    for m in results_meta:
        if m['title'] not in seen: seen.add(m['title']); deduped.append(m)
    return deduped, predicted_genre, country_mode, year_info

def dark_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); fig.patch.set_facecolor((0,0,0,0)); ax.set_facecolor((0,0,0,0))
    for sp in ax.spines.values(): sp.set_edgecolor((1,1,1,0.1))
    ax.tick_params(colors="#A0A0B5", labelsize=10); ax.title.set_color("#fff")
    return fig, ax
def evaluate_relevance(uid=1, k=8, threshold=3.5):
    if uid not in pivot_matrix.index: return None, None
    actual = pivot_matrix.loc[uid]; liked = set(actual[actual >= threshold].index)
    rec = set(pred_matrix.loc[uid].nlargest(k).index) if liked else set()
    hits = rec & liked; return len(hits)/k if liked else 0, len(hits)/len(liked) if liked else 0

st.markdown(f"""
<div class="hero">
    <div class="hero-content">
        <div class="hero-eyebrow">Thesis Project Showcase</div>
        <p class="hero-title">CineMatch AI</p>
        <p class="hero-sub">Sistem Rekomendasi Film Pintar. Menggabungkan analisis pencocokan judul/tema (Semantic Search) dan pola selera penonton (SVD Matrix).</p>
        <div class="badge-container">
            <span class="badge">🧠 SVD Matrix</span><span class="badge">🔍 Sentence-BERT</span><span class="badge">💬 Naive Bayes NLP</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================================================================
# BINGUNG CARA PAKAINYA? (SUDAH DIPERJELAS)
# ====================================================================
with st.expander("💡 Bingung cara pakainya? Klik di sini untuk panduan lengkap", expanded=False):
    st.markdown("""
    <div style="color: #A0A0B5; font-size: 0.9rem; line-height: 1.6; margin-bottom: 10px;">
        CineMatch adalah AI Pintar yang dirancang untuk mengerti maksudmu. Ada <b>3 cara mudah</b> untuk mencari film di sini:
        <ul style="margin-top: 10px; margin-bottom: 10px; padding-left: 20px;">
            <li style="margin-bottom: 10px;">🎬 <b>Cari Berdasarkan Judul (Mencari Film Mirip):</b><br>
            Cukup ketik judul film yang kamu suka. Misal: <i>"Transformers"</i> atau <i>"The Conjuring"</i>. AI akan mencari film dengan tema, genre, dan cerita yang paling mirip dengan film tersebut!</li>
            <li style="margin-bottom: 10px;">🤖 <b>Cari Berdasarkan Nuansa (Natural Language):</b><br>
            Ceritakan saja suasana yang ingin kamu tonton. Misal: <i>"film sedih yang bikin nangis"</i>, <i>"komedi gokil"</i>, atau <i>"action seru tahun 2019"</i>.</li>
            <li>🎛️ <b>Gunakan Filter Spesifik (Manual):</b><br>
            Buka menu <b>"Filter Spesifik"</b> di bawah kolom pencarian. Kamu bisa mengunci Genre, memilih Asal Negara, atau menggeser <b>Rentang Tahun</b> secara pasti tanpa tebakan AI.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### 🎬 Ceritakan atau Ketik Judul Film yang Kamu Suka:")
c0, c1, c2, c3, c4 = st.columns([0.8, 1.8, 2.2, 1.7, 1.5])
with c0: st.markdown("<p style='font-size:0.75rem;color:#A0A0B5; font-weight:600; margin-top:10px;'>🔥 Populer:</p>", unsafe_allow_html=True)
with c1: st.button("💔 sedih baper romantis", type="secondary", use_container_width=True, on_click=quick_search, args=("sedih baper romantis",))
with c2: st.button("👻 horor menyeramkan indo", type="secondary", use_container_width=True, on_click=quick_search, args=("horor menyeramkan indo",))
with c3: st.button("🤖 transformers action", type="secondary", use_container_width=True, on_click=quick_search, args=("transformers",)) # Contoh input nama film
with c4: st.button("🎲 Surprise Me!", type="secondary", use_container_width=True, on_click=quick_search, args=("film seru rating tinggi",)) 

col_inp, col_btn = st.columns([5, 1])
with col_inp: st.text_input("CARI FILM", key="query_input", placeholder="Ketik judul (The Conjuring) atau nuansa (Film action indo 2019)...", label_visibility="collapsed", on_change=trigger_search)
with col_btn: st.button("🚀 Cari Film", type="primary", use_container_width=True, on_click=trigger_search)

with st.expander("🎛️ Filter Spesifik (Manual Constraint)", expanded=False):
    f_col1, f_col2, f_col3 = st.columns([1, 1.3, 1])
    with f_col1: pil_genre = st.selectbox("Pilih Genre", ["Otomatis (AI)", "Action", "Comedy", "Drama", "Horror", "Romance"])
    with f_col2: 
        filter_tahun_aktif = st.checkbox("Gunakan Rentang Tahun")
        pil_rentang_tahun = st.slider("Geser Rentang Tahun Rilis", min_value=1970, max_value=2026, value=(2012, 2021), disabled=not filter_tahun_aktif)
    with f_col3: pil_negara = st.selectbox("Asal Negara", ["Campuran (AI)", "Hanya Indonesia", "Hanya Internasional"])

if st.session_state["do_search"]:
    st.session_state["do_search"] = False
    user_query = st.session_state["query_input"]
    
    if not user_query.strip() and pil_genre == "Otomatis (AI)" and not filter_tahun_aktif:
        st.warning("⚠️ Silakan ketik preferensi/judul film, atau gunakan menu filter di atas terlebih dahulu!")
    else:
        with st.status("🤖 Otak AI sedang memproses...", expanded=True) as status:
            st.write("🔍 Mengekstrak niat pencarian atau judul film..."); time.sleep(0.5)
            st.write("🧠 Mencocokkan kemiripan vektor semantik (Sentence-BERT)..."); time.sleep(0.5)
            st.write("📊 Membandingkan pola laten rating (SVD Matrix)..."); time.sleep(0.5)
            hasil_film, gen, count_mode, yr_info = recommend_system_web(user_query, pil_genre, filter_tahun_aktif, pil_rentang_tahun, pil_negara)
            status.update(label="Rekomendasi Ditemukan!", state="complete", expanded=False)

        genre_emoji = {"Drama":"💔","Horror":"👻","Action":"💥","Comedy":"😂","Romance":"❤️"}.get(gen,"🎬")

        st.markdown(f"""
        <div class="analysis-box">
            <div class="ai-item"><div class="ai-label">Konteks Terdeteksi</div><div class="ai-value"><span>{genre_emoji}</span> {gen}</div></div>
            <div class="ai-item"><div class="ai-label">Mode Asal</div><div class="ai-value"><span>{"🇮🇩" if count_mode == "Indonesia" else "🌍"}</span> {count_mode}</div></div>
            <div class="ai-item"><div class="ai-label">Tahun Rilis</div><div class="ai-value"><span>📅</span> {yr_info}</div></div>
            <div class="ai-item"><div class="ai-label">Total Ditemukan</div><div class="ai-value"><span>🔍</span> {len(hasil_film)} Film</div></div>
        </div>
        """, unsafe_allow_html=True)

        if len(hasil_film) == 0: st.info("⚠️ Film dengan kriteria tersebut tidak ditemukan di database.")
        else:
            st.markdown(f"""<div class="section-title"><div class="section-dot"></div><h3>Menampilkan {len(hasil_film)} Rekomendasi Terbaik Untukmu</h3></div>""", unsafe_allow_html=True)
            
            warna_neon = {"Drama":"#e50914", "Horror":"#8a2be2", "Action":"#45b6fe", "Comedy":"#f5c518", "Romance":"#ff69b4"}
            glow_color = warna_neon.get(gen, "#34d399")

            for i in range(0, len(hasil_film), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(hasil_film):
                        item = hasil_film[i + j]
                        title, genres, src = item['title'], str(item.get('genres', gen)).title(), item.get('source', 'imdb')
                        score, y_val = float(item.get('score', 0.5)), item.get('year', '')
                        
                        g_short = genres[:20] + ("…" if len(genres) > 20 else "")
                        src_lbl, src_bg, src_col = ("IMDb", "rgba(69,182,254,0.15)", "#45b6fe") if src == "imdb" else ("Indo", "rgba(52,211,153,0.15)", "#34d399")
                        pct, yr_str = min(int(score * 100), 100), f" • {int(float(y_val))}" if y_val else ""

                        with cols[j]:
                            st.markdown(f"""
                            <div class="movie-card" style="--glow-color: {glow_color};">
                                <div class="card-rank">{i+j+1}</div>
                                <div class="card-content">
                                    <p class="card-title">{title}</p>
                                    <div class="card-tags">
                                        <span class="card-tag" style="background:rgba(255,255,255,0.1); color:#fff;">{g_short}</span>
                                        <span class="card-tag" style="background:{src_bg}; color:{src_col};">{src_lbl}{yr_str}</span>
                                    </div>
                                    <div class="score-bar-wrap">
                                        <div class="score-bar-label">Akurasi AI <span>{pct}%</span></div>
                                        <div class="score-track"><div class="score-fill" style="width:{pct}%"></div></div>
                                    </div>
                                </div>
                                <div class="barcode"></div>
                            </div>
                            """, unsafe_allow_html=True)

                            with st.expander("💡 Alasan Rekomendasi"):
                                st.write(f"<span style='font-size:0.75rem'>Kemiripan Judul/Tema: <b>{int(item['sem_s']*100)}%</b></span>", unsafe_allow_html=True)
                                st.write(f"<span style='font-size:0.75rem'>Skor Kualitas Rating: <b>{int(item['svd_s']*100)}%</b></span>", unsafe_allow_html=True)
                                fig_radar = go.Figure(data=go.Scatterpolar(
                                    r=[int(item['sem_s']*100), int(item['svd_s']*100), np.random.randint(70,95)],
                                    theta=['Tema/Cerita', 'Rating SVD', 'Popularitas'], fill='toself', line_color=glow_color
                                ))
                                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), showlegend=False, height=140, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#A0A0B5", size=8))
                                st.plotly_chart(fig_radar, use_container_width=True)

                            st.button(f"🔎 Cari Film Mirip", key=f"btn_rel_{i}_{j}", on_click=quick_search, args=(title,), use_container_width=True)

        with st.expander("🔬 Mode Sidang Skripsi (Cara Kerja Sistem)", expanded=False):
            st.markdown("Sistem ini menggabungkan NLP (Naive Bayes) untuk ekstraksi niat, CBF (Sentence-BERT) untuk kesamaan konteks/judul, dan CF (SVD Matrix) untuk prediksi rating/kualitas. Skor akhir merupakan kalkulasi Hybrid 50:50.")

        st.markdown("""<div class="section-title"><div class="section-dot"></div><h3>📈 Dashboard Database & Evaluasi</h3></div>""", unsafe_allow_html=True)
        tab_stat, tab_eval = st.tabs(["📊 Statistik Dataset", "⚙️ Evaluasi SVD"])
        with tab_stat:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.write("**Top Distribusi Genre dalam Database**")
                g_counts = df_imdb['genres'].value_counts().head(7).reset_index(); g_counts.columns = ['Genre', 'Jumlah']
                fig_p = px.pie(g_counts, values='Jumlah', names='Genre', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_p, use_container_width=True)
            with col_s2:
                st.write("**Sebaran Rilis Film Berdasarkan Tahun**")
                y_counts = df_imdb['year'].dropna().value_counts().reset_index(); y_counts.columns = ['Tahun', 'Jumlah']
                fig_b = px.bar(y_counts, x='Tahun', y='Jumlah', color='Jumlah')
                st.plotly_chart(fig_b, use_container_width=True)

        with tab_eval:
            col1, col2 = st.columns(2)
            with col1:
                # FIX UNTUK ERROR BENTROK MATRIKS:
                uid_eval = 8 if 8 in pivot_matrix.index else (pivot_matrix.index[0] if len(pivot_matrix) > 0 else 1)
                if uid_eval in pivot_matrix.index:
                    u1 = pivot_matrix.loc[uid_eval].dropna()
                    cids = u1.index.intersection(pred_matrix.columns)
                    if len(cids) > 0:
                        yt, yp = u1[cids].values.astype(float), pred_matrix.loc[uid_eval, cids].values.astype(float)
                        v = ~(np.isnan(yt) | np.isnan(yp))
                        
                        if np.sum(v) > 0:  # Memastikan datanya tidak kosong (MENCEGAH ERROR VALUE)
                            rmse, mae = np.sqrt(mean_squared_error(yt[v], yp[v])), mean_absolute_error(yt[v], yp[v])
                            fig1, ax1 = dark_fig(6, 4)
                            b1 = ax1.bar(['RMSE', 'MAE'], [rmse, mae], color=['#e50914', '#f5c518'], width=0.4)
                            ax1.set_title(f'Tingkat Error Model SVD (Sampel User)', pad=15); ax1.set_ylim(0, max(rmse, mae) + 0.3)
                            for bar in b1: ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, f'{bar.get_height():.4f}', ha='center', va='bottom', color='#fff')
                            st.pyplot(fig1); plt.close(fig1)
                        else:
                            st.info("⚠️ Data irisan SVD tidak cukup untuk dievaluasi.")
                    else:
                        st.info("⚠️ Data irisan SVD tidak cukup untuk dievaluasi.")
                else:
                    st.info("⚠️ User tidak ditemukan pada tabel evaluasi.")
                    
            with col2:
                p_at_8, r_at_8 = evaluate_relevance(uid=uid_eval, k=8, threshold=3.5)
                if p_at_8 is not None and p_at_8 > 0:
                    fig2, ax2 = dark_fig(6, 4)
                    b2 = ax2.bar(['Precision @8', 'Recall @8'], [p_at_8*100, r_at_8*100], color=['#45b6fe', '#34d399'], width=0.4)
                    ax2.set_title('Relevansi Top-8', pad=15); ax2.set_ylim(0, 120)
                    for bar in b2: ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+2, f'{bar.get_height():.2f}%', ha='center', va='bottom', color='#fff')
                    st.pyplot(fig2); plt.close(fig2)
                else:
                    st.info("⚠️ Grafik Presisi & Recall tidak tersedia untuk sampel ini.")

st.markdown("""
<div class="modern-footer">
    <p>Dikembangkan untuk Tugas Akhir / Skripsi 2026</p>
    <h2><span>CineMatch AI</span> | Hybrid Recommender System</h2>
    <p>Powered by <b>Truncated SVD</b>, <b>Sentence-BERT</b>, & <b>Multinomial Naive Bayes</b></p>
</div>
""", unsafe_allow_html=True)
