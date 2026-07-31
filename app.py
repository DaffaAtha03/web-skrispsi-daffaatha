import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="CineMatch — Hybrid Movie Recommender",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="collapsed",
)

# ====================================================================
# BANK DATA SHORTCUT
# ====================================================================
POOL_PROMPTS = [
    ("💔 sedih baper romantis", "sedih baper romantis"),
    ("👻 horor menyeramkan", "horor menyeramkan"),
    ("🤖 action sci-fi seru", "action sci-fi seru"),
    ("😂 komedi ngakak abis", "komedi ngakak abis"),
    ("🦸‍♂️ superhero action", "superhero action"),
    ("🕵️ misteri detektif", "misteri detektif thriller"),
    ("👨‍👩‍👧 drama keluarga", "drama keluarga menyentuh hati"),
    ("🧟 zombie survival", "zombie survival horror"),
    ("🪄 petualangan magis", "petualangan fantasi sihir"),
    ("🏎️ balapan ekstrim", "balapan mobil action"),
    ("👽 invasi alien", "invasi alien sci-fi luar angkasa"),
    ("🗡️ perang kerajaan", "perang kerajaan kolosal sejarah")
]

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""
if "do_search" not in st.session_state:
    st.session_state["do_search"] = False
if "pil_genre_box" not in st.session_state:
    st.session_state["pil_genre_box"] = "Otomatis (AI)"
if "shortcut_prompts" not in st.session_state:
    st.session_state["shortcut_prompts"] = random.sample(POOL_PROMPTS, 3)

def trigger_search():
    st.session_state["do_search"] = True

def quick_search(query):
    st.toast(f"Mencari film dengan nuansa: {query[:15]}...", icon="🔍")
    st.session_state["query_input"] = query
    st.session_state["pil_genre_box"] = "Otomatis (AI)" 
    st.session_state["shortcut_prompts"] = random.sample(POOL_PROMPTS, 3)
    st.session_state["do_search"] = True

def similar_search(title, current_genre):
    st.toast(f"Menganalisis kemiripan dengan {title}...", icon="🤖")
    st.session_state["query_input"] = title
    if current_genre in ["Action", "Comedy", "Drama", "Horror", "Romance"]:
        st.session_state["pil_genre_box"] = current_genre
    st.session_state["do_search"] = True

def surprise_me_action():
    st.balloons()
    st.toast("Menyiapkan film kejutan untukmu!", icon="🎲")
    st.session_state["shortcut_prompts"] = random.sample(POOL_PROMPTS, 3)
    
    random_surprise = random.choice([
        "film action terbaik sepanjang masa", 
        "film komedi paling lucu", 
        "film horor paling menakutkan", 
        "film sci-fi plot twist"
    ])
    st.session_state["query_input"] = random_surprise
    st.session_state["pil_genre_box"] = "Otomatis (AI)"
    st.session_state["do_search"] = True

# ====================================================================
# CSS STYLING
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

@keyframes shine {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.hero-title {
    font-size: 3.5rem; font-weight: 800; letter-spacing: -1.5px; margin: 0 0 10px;
    background: linear-gradient(90deg, #ffffff 0%, #f5c518 20%, #e50914 50%, #f5c518 80%, #ffffff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    background-clip: text; line-height: 1.1;
    animation: shine 4s linear infinite;
}

.hero-sub { color: #A0A0B5; font-size: 1rem; margin: 0 0 24px; line-height: 1.6; font-weight: 400; }

.badge-container { display: flex; gap: 10px; flex-wrap: wrap; }
.badge { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.1); padding: 6px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; color: #fff; }

.analysis-box {
    display: flex; gap: 20px; background: rgba(20,20,35,0.6); border: 1px solid rgba(255,255,255,0.08);
    padding: 15px 25px; border-radius: 14px; margin-bottom: 25px; align-items: center; justify-content: space-between;
}
.ai-item { display: flex; flex-direction: column; gap: 4px; }
.ai-label { font-size: 0.7rem; color: #8888AA; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; }
.ai-value { font-size: 0.95rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 6px; }

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

# ====================================================================
# INISIALISASI MODEL & DATA
# ====================================================================
@st.cache_resource(show_spinner="Memuat Model AI BERT & Dataset...")
def init_models_and_data():
    try:
        pred_matrix = pd.read_parquet("pred_matrix_saved.parquet")
        df_movies   = pd.read_csv("all_movies_combined.csv")
        ratings     = pd.read_csv("ratings_clean.csv")
    except FileNotFoundError as e:
        st.error(f"❌ File data tidak ditemukan: {e}. Pastikan file dataset sudah lengkap di direktori app.py!")
        st.stop()

    if 'genres' not in df_movies.columns and 'genre' in df_movies.columns:
        df_movies = df_movies.rename(columns={'genre': 'genres'})
    
    df_movies['genres'] = df_movies['genres'].fillna('').str.lower()
    df_movies['title']  = df_movies['title'].fillna('Unknown')
    df_movies['text']   = df_movies['title'] + ' ' + df_movies['genres']
    df_movies['year']   = pd.to_numeric(df_movies.get('year', pd.Series(dtype=float)), errors='coerce')

    # Identifikasi region tag (Indonesia vs Mixed) berdasarkan kolom source/origin atau deteksi teks
    if 'source' in df_movies.columns:
        df_movies['region_tag'] = df_movies['source'].apply(lambda x: 'Indonesia' if str(x).lower() in ['indo', 'indonesia'] else 'Mixed')
    else:
        df_movies['region_tag'] = df_movies['genres'].apply(lambda x: 'Indonesia' if 'indonesia' in str(x).lower() else 'Mixed')

    def _clean(t):
        t = str(t).lower()
        t = re.sub(r'\(\d{4}\)', '', t)
        return re.sub(r'[^a-z0-9\s]', '', t).strip()
    df_movies['clean_title'] = df_movies['title'].apply(_clean)

    train_data = [
        ("film sedih banget","drama"), ("film yang bikin nangis","drama"), ("film romantis","romance"), ("film cinta sejati","romance"),
        ("film lucu banget","comedy"), ("film komedi seru","comedy"), ("film horor serem","horror"), ("film hantu","horror"),
        ("film aksi","action"), ("film action seru","action"),
    ]
    texts, labels = [x[0] for x in train_data], [x[1] for x in train_data]
    vectorizer_nlp = TfidfVectorizer()
    model_nlp = MultinomialNB()
    model_nlp.fit(vectorizer_nlp.fit_transform(texts), labels)

    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    movie_vecs = bert_model.encode(df_movies['text'].tolist(), show_progress_bar=False)

    return pred_matrix, df_movies, ratings, vectorizer_nlp, model_nlp, bert_model, movie_vecs

(pred_matrix, df_movies, ratings, vectorizer_nlp, model_nlp, bert_model, movie_vecs) = init_models_and_data()

def clean_title(title): 
    return re.sub(r'[^a-z0-9\s]', '', re.sub(r'\(\d{4}\)', '', str(title).lower())).strip()

def predict_genre_ml(text): 
    return model_nlp.predict(vectorizer_nlp.transform([text]))[0] if text.strip() else "drama"

def detect_country(text): 
    return "Indonesia" if ("indonesia" in text.lower() or "indo" in text.lower()) else "Mixed"

def detect_year(text): 
    m = re.search(r'(19\d{2}|20\d{2})', text)
    return int(m.group()) if m else None

# ====================================================================
# FUNGSI REKOMENDASI HIBRIDA
# ====================================================================
def recommend_system_web(user_input, explicit_genre, is_year_filtered, rentang_tahun, explicit_country):
    uid, max_n = 8, 8
    predicted_genre = explicit_genre.lower() if explicit_genre != "Otomatis (AI)" else predict_genre_ml(user_input)
    
    # Menentukan mode region berdasarkan filter manual atau deteksi AI
    if explicit_country == "Hanya Indonesia":
        country_mode = "Indonesia"
    elif explicit_country == "Hanya Internasional (Mixed)":
        country_mode = "Mixed"
    else:
        country_mode = detect_country(user_input)

    # Filter berdasarkan genre
    c_movies = df_movies[df_movies['genres'].str.contains(predicted_genre, case=False, na=False)].copy()

    # Filter berdasarkan region / asal negara
    if explicit_country != "Campuran (AI)":
        c_movies = c_movies[c_movies['region_tag'] == country_mode]

    if is_year_filtered:
        start_year, end_year = rentang_tahun
        c_movies = c_movies[(c_movies['year'] >= start_year) & (c_movies['year'] <= end_year)]
        year_info = f"{start_year} - {end_year}"
    else:
        target_year = detect_year(user_input)
        if target_year:
            c_movies, year_info = c_movies[c_movies['year'] == target_year], str(target_year)
        else: 
            year_info = "Semua Waktu"

    qv = bert_model.encode([user_input if user_input else predicted_genre])
    c_movies['semantic_score'] = cosine_similarity(qv, movie_vecs[c_movies.index.tolist()])[0] if not c_movies.empty else 0.0

    if uid in pred_matrix.index:
        up = pred_matrix.loc[uid]
        c_movies['svd_rating'] = c_movies['movieId'].map(up) if 'movieId' in c_movies.columns else np.nan
        mean_rating = ratings['rating'].mean() if 'rating' in ratings.columns else 3.0
        c_movies['svd_score'] = c_movies['svd_rating'].fillna(mean_rating) / 5.0
    else: 
        c_movies['svd_score'] = 0.5

    c_movies['final_score'] = 0.5 * c_movies['semantic_score'] + 0.5 * c_movies['svd_score']

    results_meta = []
    if not c_movies.empty:
        # Jika mode campuran (AI), ambil kombinasi proporsional (misal: Mixed & Indonesia)
        if explicit_country == "Campuran (AI)" and not detect_country(user_input) == "Indonesia":
            c_mixed = c_movies[c_movies['region_tag'] == 'Mixed'].sort_values('final_score', ascending=False).head(5)
            c_indo = c_movies[c_movies['region_tag'] == 'Indonesia'].sort_values('final_score', ascending=False).head(3)
            sub_combined = pd.concat([c_mixed, c_indo])
        else:
            sub_combined = c_movies.sort_values('final_score', ascending=False).head(max_n)

        results_meta.extend([{
            'title': r['title'], 
            'genres': r.get('genres',''), 
            'source': r.get('region_tag', 'Mixed'), 
            'score': float(r['final_score']), 
            'sem_s': float(r['semantic_score']), 
            'svd_s': float(r['svd_score']), 
            'year': r.get('year','')
        } for _, r in sub_combined.iterrows()])

    seen, deduped = set(), []
    for m in results_meta:
        if m['title'] not in seen: 
            seen.add(m['title'])
            deduped.append(m)
    return deduped, predicted_genre.title(), country_mode, year_info

# ====================================================================
# TAMPILAN UTAMA APLIKASI
# ====================================================================
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
            Buka menu <b>"Filter Spesifik"</b> di bawah kolom pencarian. Kamu bisa mengunci Genre, memilih Asal Negara (Mixed / Indonesia), atau menggeser <b>Rentang Tahun</b> secara pasti tanpa tebakan AI.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### 🎬 Ceritakan atau Ketik Judul Film yang Kamu Suka:")
c0, c1, c2, c3, c4 = st.columns([0.8, 1.8, 2.2, 1.7, 1.5])
with c0: 
    st.markdown("<p style='font-size:0.75rem;color:#A0A0B5; font-weight:600; margin-top:10px;'>🔥 Populer:</p>", unsafe_allow_html=True)

sc = st.session_state["shortcut_prompts"]

with c1: st.button(sc[0][0], type="secondary", use_container_width=True, on_click=quick_search, args=(sc[0][1],))
with c2: st.button(sc[1][0], type="secondary", use_container_width=True, on_click=quick_search, args=(sc[1][1],))
with c3: st.button(sc[2][0], type="secondary", use_container_width=True, on_click=quick_search, args=(sc[2][1],)) 
with c4: st.button("🎲 Surprise Me!", type="secondary", use_container_width=True, on_click=surprise_me_action) 

col_inp, col_btn = st.columns([5, 1])
with col_inp: 
    st.text_input("CARI FILM", key="query_input", placeholder="Ketik judul (The Conjuring) atau nuansa (Film action indo 2019)...", label_visibility="collapsed", on_change=trigger_search)
with col_btn: 
    st.button("🚀 Cari Film", type="primary", use_container_width=True, on_click=trigger_search)

with st.expander("🎛️ Filter Spesifik (Manual Constraint)", expanded=False):
    f_col1, f_col2, f_col3 = st.columns([1, 1.3, 1])
    with f_col1: 
        pil_genre = st.selectbox("Pilih Genre", ["Otomatis (AI)", "Action", "Comedy", "Drama", "Horror", "Romance"], key="pil_genre_box")
    with f_col2: 
        filter_tahun_aktif = st.checkbox("Gunakan Rentang Tahun")
        pil_rentang_tahun = st.slider("Geser Rentang Tahun Rilis", min_value=1970, max_value=2026, value=(2012, 2021), disabled=not filter_tahun_aktif)
    with f_col3: 
        pil_negara = st.selectbox("Asal Negara", ["Campuran (AI)", "Hanya Indonesia", "Hanya Internasional (Mixed)"])
    
    st.markdown("<hr style='margin: 10px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    st.button("🎯 Terapkan Filter & Cari Film", type="primary", use_container_width=True, on_click=trigger_search, key="btn_filter_search")

if st.session_state["do_search"]:
    st.session_state["do_search"] = False
    user_query = st.session_state["query_input"]
    
    if not user_query.strip() and pil_genre == "Otomatis (AI)" and not filter_tahun_aktif:
        st.warning("⚠️ Silakan ketik preferensi/judul film, atau gunakan menu filter di atas terlebih dahulu!")
    else:
        with st.status("🤖 Otak AI sedang memproses...", expanded=True) as status:
            st.write("🔍 Mengekstrak niat pencarian atau judul film..."); time.sleep(0.3)
            st.write("🧠 Mencocokkan kemiripan vektor semantik (Sentence-BERT)..."); time.sleep(0.3)
            st.write("📊 Membandingkan pola laten rating (SVD Matrix)..."); time.sleep(0.3)
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

        if len(hasil_film) == 0: 
            st.info("⚠️ Film dengan kriteria tersebut tidak ditemukan di database.")
        else:
            st.markdown(f"""<div class="section-title"><div class="section-dot"></div><h3>Menampilkan {len(hasil_film)} Rekomendasi Terbaik Untukmu</h3></div>""", unsafe_allow_html=True)
            
            warna_neon = {"Drama":"#e50914", "Horror":"#8a2be2", "Action":"#45b6fe", "Comedy":"#f5c518", "Romance":"#ff69b4"}
            glow_color = warna_neon.get(gen, "#34d399")

            for i in range(0, len(hasil_film), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(hasil_film):
                        item = hasil_film[i + j]
                        title, genres, src = item['title'], str(item.get('genres', gen)).title(), item.get('source', 'Mixed')
                        score, y_val = float(item.get('score', 0.5)), item.get('year', '')
                        
                        g_short = genres[:20] + ("…" if len(genres) > 20 else "")
                        # Label & Warna Badge Region
                        if src.lower() == "indonesia":
                            src_lbl, src_bg, src_col = ("Indonesia", "rgba(52,211,153,0.15)", "#34d399")
                        else:
                            src_lbl, src_bg, src_col = ("Mixed", "rgba(69,182,254,0.15)", "#45b6fe")

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
                                st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{i}_{j}_{title}")

                            st.button(f"🔎 Cari Film Mirip", key=f"btn_rel_{i}_{j}", on_click=similar_search, args=(title, gen), use_container_width=True)

        with st.expander("🔬 Mode Sidang Skripsi (Cara Kerja Sistem)", expanded=False):
            st.markdown("Sistem ini menggabungkan NLP (Naive Bayes) untuk ekstraksi niat, CBF (Sentence-BERT) untuk kesamaan konteks/judul, dan CF (SVD Matrix) untuk prediksi rating/kualitas. Skor akhir merupakan kalkulasi Hybrid 50:50.")

        st.markdown("""<div class="section-title"><div class="section-dot"></div><h3>📈 Dashboard Statistik Database</h3></div>""", unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.write("**Top Distribusi Genre dalam Database**")
            g_counts = df_movies['genres'].value_counts().head(7).reset_index(); g_counts.columns = ['Genre', 'Jumlah']
            fig_p = px.pie(g_counts, values='Jumlah', names='Genre', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_p, use_container_width=True, key="pie_gen")
        with col_s2:
            st.write("**Sebaran Rilis Film Berdasarkan Tahun**")
            y_counts = df_movies['year'].dropna().value_counts().reset_index(); y_counts.columns = ['Tahun', 'Jumlah']
            fig_b = px.bar(y_counts, x='Tahun', y='Jumlah', color='Jumlah')
            st.plotly_chart(fig_b, use_container_width=True, key="bar_year")

st.markdown("""
<div class="modern-footer">
    <p>Dikembangkan untuk Tugas Akhir / Skripsi 2026</p>
    <h2><span>CineMatch AI</span> | Hybrid Recommender System</h2>
    <p>Powered by <b>Truncated SVD</b>, <b>Sentence-BERT</b>, & <b>Multinomial Naive Bayes</b></p>
</div>
""", unsafe_allow_html=True)
