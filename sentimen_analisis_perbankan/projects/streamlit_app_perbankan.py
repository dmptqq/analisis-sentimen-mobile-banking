# streamlit_app.py
# APLIKASI SENTIMEN MOBILE BANKING - VERSI FINAL (4 BANK UTAMA + 15 BANK DIGITAL)

import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')
import os
from io import BytesIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Untuk scraping Google Play
try:
    from google_play_scraper import reviews, Sort, app
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Sentiment Analysis - Mobile Banking",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto"
)

# ============================================
# INISIALISASI SESSION STATE
# ============================================
if 'tfidf_clicked' not in st.session_state:
    st.session_state.tfidf_clicked = False
if 'tfidf_fig' not in st.session_state:
    st.session_state.tfidf_fig = None
if 'tfidf_df' not in st.session_state:
    st.session_state.tfidf_df = None
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'current_bank' not in st.session_state:
    st.session_state.current_bank = None
if 'is_multi_bank' not in st.session_state:
    st.session_state.is_multi_bank = False
if 'scraping_error' not in st.session_state:
    st.session_state.scraping_error = None
if 'selected_banks_list' not in st.session_state:
    st.session_state.selected_banks_list = []

# ============================================
# DATABASE BANK (4 BANK UTAMA + 15 BANK DIGITAL)
# ============================================
BANKS = {
    # Bank Utama (4)
    "BCA": {"app_id": "com.bca.mybca.omni.android", "name": "BCA Mobile"},
    "BRI": {"app_id": "id.co.bri.brimo", "name": "BRImo"},
    "BTN": {"app_id": "id.co.btn.mobilebanking.android", "name": "bale by BTN"},
    "MANDIRI": {"app_id": "id.bmri.livin", "name": "Livin' by Mandiri"},
    
    # Bank Digital (15)
    "BNC (Neobank)": {"app_id": "com.bnc.finance", "name": "Neobank by BNC"},
    "Superbank": {"app_id": "id.co.bankfama.android", "name": "Superbank"},
    "SeaBank": {"app_id": "id.co.bankbkemobile.digitalbank", "name": "SeaBank"},
    "blu by BCA Digital": {"app_id": "com.bcadigital.blu", "name": "blu by BCA Digital"},
    "wondr by BNI": {"app_id": "id.bni.wondr", "name": "wondr by BNI"},
    "Bank Jago": {"app_id": "com.jago.digitalBanking", "name": "Bank Jago"},
    "Krom Bank Digital": {"app_id": "com.krom.android", "name": "Krom"},
    "Allo Bank": {"app_id": "com.alloapp.yump", "name": "Allo Bank"},
    "DBS digibank": {"app_id": "com.dbs.id.pt.digitalbank", "name": "DBS digibank"},
    "OCTO by CIMB Niaga": {"app_id": "id.co.cimbniaga.mobile.android", "name": "OCTO Mobile"},
    "Bank Saqu": {"app_id": "bjj.bank.digital.indo.prod", "name": "Bank Saqu"},
    "Raya Digital Bank": {"app_id": "id.co.bankraya.apps", "name": "Raya Digital Bank"},
    "Aladin Bank Syariah": {"app_id": "id.aladinbank.mobile", "name": "Aladin Bank"},
    "Amar Bank": {"app_id": "com.senyumkubank.rekeningonline", "name": "Amar Bank"},
    "M-Smile by Bank Mega": {"app_id": "com.msmile.bankmega", "name": "M-Smile"}
}

# ============================================
# LOAD INSET LEXICON FAJRI KOTO
# ============================================

# Cara paling aman untuk Streamlit Cloud
# Karena file TSV berada di FOLDER YANG SAMA dengan script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSITIVE_FILE = os.path.join(SCRIPT_DIR, "positive.tsv")
NEGATIVE_FILE = os.path.join(SCRIPT_DIR, "negative.tsv")

positive_lexicon = {}
negative_lexicon = {}

def load_inset_lexicon():
    global positive_lexicon, negative_lexicon
    
    if os.path.exists(POSITIVE_FILE):
        try:
            df_pos = pd.read_csv(POSITIVE_FILE, sep='\t')
            df_pos = df_pos[df_pos['weight'] >= 1]
            for _, row in df_pos.iterrows():
                word = str(row['word']).lower().strip()
                weight = int(row['weight'])
                if word and word != 'nan':
                    positive_lexicon[word] = weight
            st.success(f"✅ Loaded {len(positive_lexicon)} positive words")
        except Exception as e:
            st.error(f"Gagal load positive.tsv: {e}")
    else:
        st.warning(f"File {POSITIVE_FILE} tidak ditemukan!")
    
    if os.path.exists(NEGATIVE_FILE):
        try:
            df_neg = pd.read_csv(NEGATIVE_FILE, sep='\t')
            df_neg = df_neg[df_neg['weight'] <= -1]
            for _, row in df_neg.iterrows():
                word = str(row['word']).lower().strip()
                weight = abs(int(row['weight']))
                if word and word != 'nan':
                    negative_lexicon[word] = weight
            st.success(f"✅ Loaded {len(negative_lexicon)} negative words")
        except Exception as e:
            st.error(f"Gagal load negative.tsv: {e}")
    else:
        st.warning(f"File {NEGATIVE_FILE} tidak ditemukan!")

load_inset_lexicon()

# ============================================
# FALLBACK LEXICON (JIKA FILE TIDAK DITEMUKAN)
# ============================================

FALLBACK_POSITIVE = {
    'bagus', 'baik', 'mantap', 'keren', 'hebat', 'puas', 'terbaik', 'sempurna',
    'cepat', 'mudah', 'lengkap', 'berhasil', 'sukses', 'aman', 'nyaman', 
    'membantu', 'top', 'oke', 'senang', 'suka', 'recommended', 'lancar', 'stabil'
}

FALLBACK_NEGATIVE = {
    'error', 'bug', 'lemot', 'lambat', 'macet', 'gagal', 'masalah', 'jelek',
    'buruk', 'kecewa', 'kesal', 'marah', 'ribet', 'crash', 'freeze', 'down',
    'force close', 'tidak bisa', 'lemot banget', 'error terus', 'frustasi'
}

if not positive_lexicon:
    positive_lexicon = {word: 3 for word in FALLBACK_POSITIVE}
    st.info("Menggunakan fallback positive lexicon")

if not negative_lexicon:
    negative_lexicon = {word: 3 for word in FALLBACK_NEGATIVE}
    st.info("Menggunakan fallback negative lexicon")

POSITIVE_WORDS = set(positive_lexicon.keys())
NEGATIVE_WORDS = set(negative_lexicon.keys())

# ============================================
# STOPWORDS
# ============================================
STOPWORDS = {
    'yang', 'dan', 'di', 'dari', 'ini', 'itu', 'untuk', 'dengan', 'pada', 'ke',
    'dalam', 'seperti', 'juga', 'adalah', 'mereka', 'kita', 'kami', 'anda', 'aku',
    'saya', 'kamu', 'dia', 'ia', 'tersebut', 'akan', 'bisa', 'dapat', 'telah',
    'sudah', 'sedang', 'masih', 'lagi', 'saja', 'pun', 'sangat', 'cukup', 'kurang',
    'yg', 'udah', 'udh', 'gak', 'ga', 'tdk', 'nggak', 'nya', 'sih', 'dong', 'deh',
    'kok', 'lho', 'nih', 'banget', 'bgt', 'aja', 'si', 'gue', 'lo', 'lu'
}

# ============================================
# FUNGSI CLEAN APP ID
# ============================================

def clean_app_id(app_id):
    if not app_id:
        return app_id
    app_id = app_id.strip()
    if '&' in app_id:
        app_id = app_id.split('&')[0]
    if '?' in app_id:
        app_id = app_id.split('?')[0]
    return app_id

# ============================================
# FUNGSI SCRAPING GOOGLE PLAY
# ============================================

def is_valid_review(review_text):
    if not review_text or not isinstance(review_text, str):
        return False
    
    text = review_text.strip()
    
    if len(text) < 10:
        return False
    if len(text) > 5000:
        return False
    
    has_letter = any(c.isalpha() for c in text)
    if not has_letter:
        return False
    
    has_digit = any(c.isdigit() for c in text)
    only_numbers = all(c.isdigit() or c.isspace() for c in text)
    if only_numbers:
        return False
    
    spam_patterns = [
        r'buka.*link', r'klik.*disini', r'daftar.*sekarang',
        r'dapatkan.*uang', r'bonus', r'promo.*terbatas',
        r'whatsapp.*\d+', r'wa.*\d+', r'telegram.*\d+',
        r'ikut.*grup', r'join.*grup', r'follow.*instagram'
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, text.lower()):
            return False
    
    words = text.split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return False
    
    return True

def scrape_google_play_reviews(app_id, count=4000, lang='id', country='id'):
    if not SCRAPER_AVAILABLE:
        return pd.DataFrame(), "google-play-scraper tidak terinstall."
    
    try:
        actual_count = min(count, 4000)
        all_reviews = []
        continuation_token = None
        
        progress_bar = st.progress(0)
        
        while len(all_reviews) < actual_count:
            # Hitung sisa yang dibutuhkan
            remaining = actual_count - len(all_reviews)
            # Ambil minimal antara 250 dan sisa yang dibutuhkan
            batch_size = min(250, remaining)
            
            result, token = reviews(
                app_id,
                lang='id',
                country='id',
                sort=Sort.NEWEST,
                count=batch_size,
                continuation_token=continuation_token
            )
            
            if not result: 
                break
            
            # Potong jika melebihi batas
            if len(all_reviews) + len(result) > actual_count:
                result = result[:actual_count - len(all_reviews)]
            
            all_reviews.extend(result)
            continuation_token = token
            
            progress = min(len(all_reviews) / actual_count, 1.0)
            progress_bar.progress(progress)
            
            time.sleep(1)
            
            if not continuation_token or len(all_reviews) >= actual_count: 
                break
        
        progress_bar.empty()
        
        if not all_reviews:
            return pd.DataFrame(), f"Tidak ada ulasan untuk aplikasi '{app_id}'"
        
        reviews_data = []
        for review in all_reviews:
            if review.get('content') and review['content'].strip():
                reviews_data.append({
                    'user_name': review.get('userName', 'Unknown'),
                    'content': review['content'],
                    'rating': review.get('score', 0),
                    'at': review.get('at'),
                    'thumbs_up_count': review.get('thumbsUpCount', 0),
                    'review_id': review.get('reviewId', '')
                })
        
        df_result = pd.DataFrame(reviews_data)
        return df_result, None
        
    except Exception as e:
        return pd.DataFrame(), f"Gagal mengambil data: {str(e)[:200]}"

def scrape_all_banks(selected_banks, review_count=4000):
    all_reviews = []
    errors = []
    progress_bar = st.progress(0)
    
    for idx, bank_name in enumerate(selected_banks):
        app_id = BANKS[bank_name]["app_id"]
        st.info(f"📱 Mengambil data {bank_name} ({app_id})...")
        
        df_reviews, error = scrape_google_play_reviews(app_id, count=review_count)
        
        if error:
            errors.append(f"{bank_name}: {error}")
            st.error(f"❌ {bank_name}: Gagal mengambil data - {error[:100]}")
        elif not df_reviews.empty:
            df_reviews['bank_name'] = bank_name
            all_reviews.append(df_reviews)
            st.success(f"✅ {bank_name}: {len(df_reviews)} ulasan")
        else:
            errors.append(f"{bank_name}: Tidak ada data")
            st.warning(f"⚠️ {bank_name}: Tidak ada ulasan yang ditemukan")
        
        progress_bar.progress((idx + 1) / len(selected_banks))
        time.sleep(1.5)
    
    progress_bar.empty()
    
    if errors:
        st.session_state.scraping_error = "\n".join(errors)
    
    if all_reviews:
        return pd.concat(all_reviews, ignore_index=True)
    return pd.DataFrame()

# ============================================
# FUNGSI PREPROCESSING
# ============================================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words)

def hitung_score_dengan_bobot(text):
    if not isinstance(text, str):
        return 0, 0
    
    text_lower = text.lower()
    pos_total = 0
    neg_total = 0
    
    words = text_lower.split()
    for word in words:
        if word in positive_lexicon:
            pos_total += positive_lexicon[word]
        if word in negative_lexicon:
            neg_total += negative_lexicon[word]
    
    for phrase, weight in positive_lexicon.items():
        if ' ' in phrase and phrase in text_lower:
            pos_total += weight
    
    for phrase, weight in negative_lexicon.items():
        if ' ' in phrase and phrase in text_lower:
            neg_total += weight
    
    return pos_total, neg_total

def klasifikasi_5_kategori(pos, neg, rating=None):
    rating_boost = 0
    if rating is not None and rating in [1, 2, 3, 4, 5]:
        if rating >= 4:
            rating_boost = 2
        elif rating <= 2:
            rating_boost = -2
    
    net_score = pos - neg
    
    if net_score >= 3:
        return 'Puas'
    elif net_score == 2:
        return 'Puas'
    elif net_score == 1:
        return 'Ragu-ragu (Cenderung Puas)'
    elif net_score == 0:
        return 'Ragu-ragu (Tetap Netral)'
    elif net_score == -1:
        return 'Ragu-ragu (Cenderung Tidak Puas)'
    elif net_score == -2:
        return 'Ragu-ragu (Cenderung Tidak Puas)'
    elif net_score <= -3:
        return 'Tidak Puas'
    return 'Ragu-ragu (Tetap Netral)'

def agregasi_ke_3_kategori(kategori_5):
    if kategori_5 == 'Puas' or kategori_5 == 'Ragu-ragu (Cenderung Puas)':
        return 'Puas'
    elif kategori_5 == 'Ragu-ragu (Tetap Netral)':
        return 'Ragu-ragu'
    return 'Tidak Puas'

def preprocess_dataframe(df):
    df = df.copy()
    
    df['content'] = df['content'].fillna("").astype(str)
    df = df[df['content'].str.len() >= 5].copy()
    
    if len(df) == 0:
        return df
    
    df['cleaned'] = df['content'].apply(clean_text)
    df['no_stopwords'] = df['cleaned'].apply(remove_stopwords)
    df['processed'] = df['no_stopwords']
    df = df[df['processed'].str.len() > 3].copy()
    
    if len(df) > 0:
        pos_neg = df['processed'].apply(hitung_score_dengan_bobot)
        df['pos_count'] = pos_neg.apply(lambda x: x[0])
        df['neg_count'] = pos_neg.apply(lambda x: x[1])
        
        df['final_sentiment'] = df.apply(
            lambda row: klasifikasi_5_kategori(row['pos_count'], row['neg_count'], row['rating'] if 'rating' in row else None),
            axis=1
        )
        
        df['sentiment_3class'] = df['final_sentiment'].apply(agregasi_ke_3_kategori)
    
    return df

# ============================================
# FUNGSI VISUALISASI
# ============================================

def plot_chart_5_kategori(df, title="Analisis Sentimen Perbankan"):

    colors = [
        '#4A90E2',
        '#7FB3D5',
        '#F7DC6F',
        '#F1948A',
        '#E35D5D'
    ]

    order = [
        'Puas',
        'Ragu-ragu (Cenderung Puas)',
        'Ragu-ragu (Tetap Netral)',
        'Ragu-ragu (Cenderung Tidak Puas)',
        'Tidak Puas'
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        x='bank_name',
        hue='final_sentiment',
        data=df,
        hue_order=order,
        palette=colors,
        edgecolor='white',
        linewidth=0.8,
        ax=ax
    )

    for p in ax.patches:
        height = p.get_height()

        if height > 0:
            ax.annotate(
                f'{int(height)}',
                (
                    p.get_x() + p.get_width()/2,
                    height
                ),
                ha='center',
                va='bottom',
                fontsize=8
            )

    max_height = max(
        [p.get_height() for p in ax.patches]
    )

    ax.set_ylim(0, max_height * 1.25)

    ax.set_title(
        title,
        fontsize=13,
        fontweight='bold'
    )

    ax.set_xlabel("Bank", fontsize=10)
    ax.set_ylabel("Jumlah Ulasan", fontsize=10)

    ax.legend(
        title='Kategori Detail',
        fontsize=8,
        title_fontsize=9,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False
    )

    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()

    return fig

def plot_chart_3_kategori(df, title="Ringkasan Sentimen"):

    colors = {
        'Puas': '#4A90E2',
        'Ragu-ragu': '#F7DC6F',
        'Tidak Puas': '#E35D5D'
    }

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7, 3.2)
    )

    sentiment_counts = (
        df['sentiment_3class']
        .value_counts()
        .reindex(
            ['Puas', 'Ragu-ragu', 'Tidak Puas'],
            fill_value=0
        )
    )

    bars = ax1.bar(
        sentiment_counts.index,
        sentiment_counts.values,
        color=[
            colors.get(x)
            for x in sentiment_counts.index
        ]
    )

    max_val = sentiment_counts.max()

    ax1.set_ylim(0, max_val * 1.30)

    ax1.set_title(
        'Jumlah Sentimen',
        fontsize=11,
        fontweight='bold'
    )

    ax1.set_ylabel(
        'Jumlah',
        fontsize=9
    )

    ax1.tick_params(labelsize=9)

    for bar, val in zip(
        bars,
        sentiment_counts.values
    ):
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            val + max_val * 0.03,
            str(val),
            ha='center',
            fontsize=9,
            fontweight='bold'
        )

    ax2.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        textprops={'fontsize': 9},
        radius=0.85,
        colors=[
            colors.get(x)
            for x in sentiment_counts.index
        ]
    )

    ax2.set_title(
        'Persentase',
        fontsize=11,
        fontweight='bold'
    )

    plt.tight_layout()

    return fig

def plot_tfidf_top_words(vectorizer, tfidf_matrix, top_n=20, title=""):
    feature_names = vectorizer.get_feature_names_out()
    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    actual_top_n = min(top_n, len(feature_names))
    
    if actual_top_n == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Tidak cukup data untuk TF-IDF", ha='center', va='center')
        return fig
    
    top_indices = mean_tfidf.argsort()[-actual_top_n:][::-1]
    top_words = feature_names[top_indices]
    top_scores = mean_tfidf[top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(actual_top_n), top_scores, color='#3498db')
    ax.set_yticks(range(actual_top_n))
    ax.set_yticklabels(top_words)
    ax.set_xlabel('Skor TF-IDF')
    ax.set_title(f'Top {actual_top_n} Kata - {title}')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def plot_confusion_matrix_heatmap(cm, labels, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={'size': 10, 'weight': 'bold'})
    ax.set_xlabel('Prediksi Lexicon', fontsize=10)
    ax.set_ylabel('Rating (Ground Truth)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    return fig

def display_lexicon_stats():
    st.markdown("### 📚 InSet Lexicon Fajri Koto")
    st.caption("Referensi: Koto, F., & Rahmaningtyas, G. Y. (2018)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📖 Kata Positif", f"{len(positive_lexicon):,}")
        if positive_lexicon:
            weights_pos = list(positive_lexicon.values())
            st.caption(f"Bobot: 1-{max(weights_pos)}")
    with col2:
        st.metric("📖 Kata Negatif", f"{len(negative_lexicon):,}")
        if negative_lexicon:
            weights_neg = list(negative_lexicon.values())
            st.caption(f"Bobot: 1-{max(weights_neg)}")
    
    with st.expander("🔍 Lihat Contoh Kata dari InSet Lexicon"):
        pos_examples = list(positive_lexicon.items())[:15]
        neg_examples = list(negative_lexicon.items())[:15]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**✅ Kata Positif (dengan bobot):**")
            for word, weight in pos_examples:
                if ' ' in word:
                    st.write(f"- \"{word}\" (bobot: {weight})")
                else:
                    st.write(f"- {word} (bobot: {weight})")
        
        with col2:
            st.write("**❌ Kata Negatif (dengan bobot):**")
            for word, weight in neg_examples:
                if ' ' in word:
                    st.write(f"- \"{word}\" (bobot: {weight})")
                else:
                    st.write(f"- {word} (bobot: {weight})")

def display_process_tab(df_raw, df_processed, selected_bank_name, is_multi_bank):
    st.markdown("### 🔄 Alur Proses Analisis Sentimen")
    st.markdown("---")
    
    st.info("📌 **Sumber Data:** Scraping Real dari Google Play Store")
    display_lexicon_stats()
    st.markdown("---")
    
    # STEP 1: Data Mentah
    st.markdown("### 📁 STEP 1: DATA MENTAH")
    st.write(f"**Jumlah data awal:** {len(df_raw)} ulasan")
    st.write(f"**Kolom yang tersedia:** {', '.join(df_raw.columns.tolist())}")
    with st.expander("📄 Lihat Preview Data Mentah"):
        st.dataframe(df_raw.head(10), use_container_width=True)
    
    # STEP 2: Entity Relation
    if is_multi_bank:
        st.markdown("### 🔗 STEP 2: ENTITY RELATION (PENGGABUNGAN DATA)")
        bank_counts = df_raw['bank_name'].value_counts()
        st.write(f"**Data berasal dari {len(bank_counts)} bank:** {', '.join(bank_counts.index.tolist())}")
        st.write("**Distribusi data per bank:**")
        st.dataframe(pd.DataFrame({'Bank': bank_counts.index, 'Jumlah': bank_counts.values}), use_container_width=True)
        
        if 'rating' in df_raw.columns:
            st.write(f"**Rating range:** {df_raw['rating'].min()} - {df_raw['rating'].max()}")
    else:
        st.markdown("### 🔗 STEP 2: ENTITY RELATION")
        st.info("📌 Mode 1 Bank: Analisis single bank.")
        st.write(f"**Bank yang dianalisis:** {selected_bank_name}")
    
    # STEP 3: Cleaning Text
    st.markdown("### 🧹 STEP 3: CLEANING TEXT")
    st.write("Membersihkan teks: lowercase, hapus URL, hapus tanda baca, hapus angka, hapus spasi berlebih")
    
    df_temp = df_raw.copy()
    df_temp['cleaned_example'] = df_temp['content'].apply(clean_text)
    with st.expander("📄 Lihat Contoh Hasil Cleaning"):
        sample_clean = df_temp[['content', 'cleaned_example']].head(5)
        sample_clean.columns = ['Original (Sebelum)', 'Cleaned (Sesudah)']
        st.dataframe(sample_clean, use_container_width=True)
    
    # STEP 4: Stopword Removal
    st.markdown("### 🚫 STEP 4: STOPWORD REMOVAL")
    st.write(f"**Jumlah stopwords:** {len(STOPWORDS)} kata")
    st.write("Contoh stopwords: yang, dan, di, dari, untuk, dengan, pada, ke, dll")
    
    df_temp['no_stopwords_example'] = df_temp['cleaned_example'].apply(remove_stopwords)
    with st.expander("📄 Lihat Contoh Hasil Stopword Removal"):
        sample_stop = df_temp[['cleaned_example', 'no_stopwords_example']].head(5)
        sample_stop.columns = ['After Cleaning', 'After Stopword Removal']
        st.dataframe(sample_stop, use_container_width=True)
    
    # STEP 5: Text Processing (Hasil Akhir)
    st.markdown("### ✅ STEP 5: TEXT PROCESSING (HASIL AKHIR)")
    st.write(f"**Jumlah data setelah preprocessing:** {len(df_processed)} ulasan")
    st.write(f"**Data yang terfilter:** {len(df_processed)} dari {len(df_raw)} data awal")
    
    with st.expander("📄 Lihat Contoh Data Setelah Processing"):
        sample_processed = df_processed[['content', 'processed', 'pos_count', 'neg_count', 'final_sentiment']].head(10)
        st.dataframe(sample_processed, use_container_width=True)
    
    # STEP 6: Apply InSet Lexicon
    st.markdown("### 📝 STEP 6: APPLY INSET LEXICON (SCORING)")
    st.write(f"**Kata positif:** {len(positive_lexicon)} kata dengan bobot 1-5")
    st.write(f"**Kata negatif:** {len(negative_lexicon)} kata dengan bobot 1-5")
    
    with st.expander("📄 Lihat Contoh Perhitungan Score dengan Bobot"):
        score_sample = df_processed[['processed', 'pos_count', 'neg_count']].head(10).copy()
        score_sample['net_score'] = score_sample['pos_count'] - score_sample['neg_count']
        st.dataframe(score_sample, use_container_width=True)
    
    st.write("**Hasil labeling 5 kategori:**")
    sentiment_counts = df_processed['final_sentiment'].value_counts()
    for sent, count in sentiment_counts.items():
        st.write(f"- {sent}: {count} ({count/len(df_processed)*100:.1f}%)")
    
    # Visualisasi Distribusi Sentimen 5 Kategori
    st.markdown("### 📊 Visualisasi Distribusi Sentimen (5 Kategori)")
    colors = ['#4A90E2', '#7FB3D5', '#F7DC6F', '#F1948A', '#E35D5D']
    order = ['Puas', 'Ragu-ragu (Cenderung Puas)', 'Ragu-ragu (Tetap Netral)',
             'Ragu-ragu (Cenderung Tidak Puas)', 'Tidak Puas']
    
    sentiment_counts = df_processed['final_sentiment'].value_counts()
    sentiment_counts = sentiment_counts.reindex(order, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(sentiment_counts)), sentiment_counts.values, color=colors)
    ax.set_xticks(range(len(sentiment_counts)))
    ax.set_xticklabels(sentiment_counts.index, rotation=45, ha='right', fontsize=9)
    ax.set_title(f'Distribusi 5 Kategori Sentimen - {selected_bank_name}')
    ax.set_ylabel('Jumlah Ulasan')
    ax.set_xlabel('Kategori Sentimen')
    for bar, val in zip(bars, sentiment_counts.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(int(val)), 
                    ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    
    # Visualisasi Perbandingan 3 Kategori
    st.markdown("### 📊 Visualisasi Distribusi Sentimen (3 Kategori - Agregasi)")
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sent_3class = df_processed['sentiment_3class'].value_counts()
    order_3class = ['Puas', 'Ragu-ragu', 'Tidak Puas']
    sent_3class = sent_3class.reindex(order_3class, fill_value=0)
    
    bars2 = ax1.bar(sent_3class.index, sent_3class.values, color=['#4A90E2', '#F7DC6F', '#E35D5D'])
    ax1.set_title('Jumlah per Sentimen (3 Kategori)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Jumlah Ulasan')
    for bar, val in zip(bars2, sent_3class.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontweight='bold')
    
    ax2.pie(sent_3class.values, labels=sent_3class.index, autopct='%1.1f%%',
            colors=['#4A90E2', '#F7DC6F', '#E35D5D'])
    ax2.set_title('Persentase Sentimen (3 Kategori)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    
    st.markdown("""
    <div class="aggregation-box">
        <b>📌 Keterangan Agregasi ke 3 Kategori:</b><br>
        🔵 Puas = Puas + Ragu-ragu (Cenderung Puas)<br>
        🟡 Ragu-ragu = Ragu-ragu (Tetap Netral)<br>
        🔴 Tidak Puas = Tidak Puas + Ragu-ragu (Cenderung Tidak Puas)
    </div>
    """, unsafe_allow_html=True)
    
    st.success("✅ **PROSES SELESAI!**")

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f2b3d 0%, #1a4a6f 100%);
        padding: 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 600;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
        font-size: 1rem;
    }
    
    .stat-card {
        background: white;
        margin-bottom:18px;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    
    .stat-card .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a4a6f;
        margin: 0.5rem 0;
    }
    
    .stat-card .stat-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1a2a 0%, #0f2b3d 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255,255,255,0.1);
        border-radius: 12px;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        background-color: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 8px;
    }
    
    div[data-testid="stHorizontalRadio"]{
    display:flex !important;
    flex-wrap:nowrap !important;
    overflow-x:auto !important;
    gap:10px !important;
    padding-bottom:6px;
    }

    div[data-testid="stHorizontalRadio"] > div{
        display:flex !important;
        flex-wrap:nowrap !important;
        width:max-content !important;
    }

    div[data-testid="stHorizontalRadio"] label{
        min-width:max-content !important;
        white-space:nowrap !important;
        border-radius:12px !important;
        padding:10px 18px !important;
        border:1px solid #dbe2ea !important;
        background:white !important;
        transition:all .2s ease;
    }

    div[data-testid="stHorizontalRadio"] label:hover{
        border-color:#1a4a6f !important;
    }
    
    div[data-testid="stHorizontalRadio"] label[data-baseweb="radio"] {
        margin: 0 4px;
    }
    
    [data-testid="stMetric"] {
        background-color: #1a4a6f !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
        border: none !important;
    }
    
    [data-testid="stMetric"] label {
        color: #f39c12 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }
    
    [data-testid="stMetric"] .stMetricValue {
        color: white !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    
    .aggregation-box {
        background: linear-gradient(135deg, #1a4a6f 0%, #0f2b3d 100%);
        padding: 15px;
        border-radius: 12px;
        margin: 15px 0;
        color: white;
    }
    
    .aggregation-box b {
        color: #f39c12;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
    
    
/* =========================================
   RESPONSIVE MOBILE & TABLET
========================================= */

.block-container{
    padding-top:2.5rem;
    padding-bottom:1rem;
}

@media (max-width:1024px){

    .main-header{
        padding:1.2rem;
        border-radius:14px;
    }

    .main-header h1{
        font-size:1.5rem !important;
        line-height:1.3;
    }

    .main-header p{
        font-size:0.85rem !important;
    }

    h1{
        font-size:1.7rem !important;
    }

    h2{
        font-size:1.4rem !important;
    }

    h3{
        font-size:1.2rem !important;
    }

    .stat-card .stat-value{
        font-size:1.4rem;
    }

    .stat-card .stat-label{
        font-size:.75rem;
    }

    [data-testid="stMetric"] .stMetricValue{
        font-size:1.4rem !important;
    }
}

@media (max-width:768px){

    html, body{
        font-size:14px !important;
    }

    .block-container{
        padding-top:2rem !important;
        padding-left:0.8rem !important;
        padding-right:0.8rem !important;
        padding-bottom:1rem !important;
    }

    .main-header{
        padding:0.9rem;
        text-align:center;
    }

    .main-header h1{
        font-size:1.15rem !important;
        line-height:1.35;
    }

    .main-header p{
        font-size:0.75rem !important;
        line-height:1.4;
    }

    h1{
        font-size:1.4rem !important;
    }

    h2{
        font-size:1.2rem !important;
    }

    h3{
        font-size:1.05rem !important;
    }

    .stat-card{
        margin-bottom:20px;
        padding:.8rem;
    }

    .stat-card .stat-value{
        font-size:1.2rem;
    }

    .stat-card .stat-label{
        font-size:.65rem;
    }

    [data-testid="stMetric"]{
        padding:.6rem !important;
    }

    [data-testid="stMetric"] .stMetricValue{
        font-size:1rem !important;
    }

    [data-testid="stMetric"] label{
        font-size:.7rem !important;
    }

    .aggregation-box{
        font-size:0.8rem;
    }

    div[data-testid="column"]{
        width:100% !important;
        flex:1 1 100% !important;
    }

    .stButton button{
        width:100%;
    }
    
    @media (max-width:768px){

    div[data-testid="stHorizontalRadio"]{
        overflow-x:auto !important;
        flex-wrap:nowrap !important;
        scrollbar-width:none;
    }

    div[data-testid="stHorizontalRadio"]::-webkit-scrollbar{
        display:none;
    }

    div[data-testid="stHorizontalRadio"] label{
        min-width:max-content !important;
        font-size:13px !important;
        padding:10px 14px !important;
    }
}

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Sentiment Analysis Dashboard</h1>
        <p>Mobile Banking Application | 4 Bank Utama + 15 Bank Digital</p>
        <p style="font-size: 0.85rem; margin-top: 0.5rem;">🔍 Scraping Real dari Google Play Store + InSet Lexicon Fajri Koto</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan")
        st.markdown("---")
        
        # Mode Analisis
        analysis_mode = st.radio(
            "📋 Mode Analisis",
            ["🏦 Analisis 4 Bank Utama (BCA, BRI, BTN, MANDIRI)", "🏦 Pilih Bank Tertentu"]
        )
        
        selected_banks = []
        is_multi_bank = False
        selected_bank_name = ""
        
        if analysis_mode == "🏦 Analisis 4 Bank Utama (BCA, BRI, BTN, MANDIRI)":
            selected_banks = ["BCA", "BRI", "BTN", "MANDIRI"]
            selected_bank_name = "BCA, BRI, BTN, MANDIRI"
            is_multi_bank = True
            st.info("📊 Menganalisis 4 Bank Utama: BCA, BRI, BTN, MANDIRI")
            
        elif analysis_mode == "🏦 Pilih Bank Tertentu":
            selected_bank = st.selectbox(
                "Pilih Bank", 
                list(BANKS.keys()),
                help="Pilih bank yang ingin dianalisis"
            )
            selected_banks = [selected_bank]
            selected_bank_name = selected_bank
            is_multi_bank = False
            app_id = BANKS[selected_bank]["app_id"]
            st.caption(f"📱 App ID: `{app_id}`")
        
        st.markdown("---")
        
        # Jumlah data per bank - MAKSIMAL 4000
        jumlah_data = st.slider(
            "📊 Jumlah Data per Bank", 
            min_value=100, 
            max_value=4000, 
            value=2000, 
            step=100,
            help="Jumlah review yang akan di-scrape per bank. MAKSIMAL 4000 review. Semakin banyak data, semakin lama proses scraping."
        )
        
        st.markdown("---")
        
        st.markdown("### 📚 InSet Lexicon")
        st.markdown(f"**Positif:** {len(positive_lexicon):,} kata")
        st.markdown(f"**Negatif:** {len(negative_lexicon):,} kata")
        st.caption("Fajri Koto (2018)")
        
        process_button = st.button("🚀 Proses Data", use_container_width=True)
    
    # PROSES DATA
    if process_button:
        if not selected_banks:
            st.error("❌ Silakan pilih bank terlebih dahulu!")
            st.stop()
        
        st.session_state.tfidf_clicked = False
        st.session_state.tfidf_fig = None
        st.session_state.tfidf_df = None
        st.session_state.scraping_error = None
        
        if not SCRAPER_AVAILABLE:
            st.error("❌ google-play-scraper tidak terinstall. Jalankan: pip install google-play-scraper")
            st.stop()
        
        with st.spinner(f"Memproses data dari Google Play Store (maksimal {jumlah_data} per bank)..."):
            df_raw = scrape_all_banks(selected_banks, jumlah_data)
            
            if df_raw is not None and len(df_raw) > 0:
                df_processed = preprocess_dataframe(df_raw.copy())
                if len(df_processed) > 0:
                    st.session_state.df_raw = df_raw
                    st.session_state.df_processed = df_processed
                    st.session_state.processed = True
                    st.session_state.current_bank = selected_bank_name
                    st.session_state.is_multi_bank = is_multi_bank
                    st.session_state.selected_banks_list = selected_banks
                    st.success(f"✅ Berhasil mengambil {len(df_raw)} ulasan dari {len(selected_banks)} bank! Setelah preprocessing: {len(df_processed)} ulasan.")
                else:
                    st.error("❌ Tidak ada data yang valid setelah preprocessing! Pastikan ulasan memiliki teks yang bermakna.")
            else:
                st.error("❌ Gagal mengambil data dari Google Play Store.")
                if st.session_state.scraping_error:
                    with st.expander("📋 Detail Error"):
                        st.code(st.session_state.scraping_error)
    
    # TAMPILKAN HASIL
    if st.session_state.processed and st.session_state.df_processed is not None:
        df_raw = st.session_state.df_raw
        df = st.session_state.df_processed
        selected_banks = st.session_state.selected_banks_list
        
        # Stats cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">TOTAL REVIEW</div>
                <div class="stat-value">{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            avg_rating = df['rating'].mean() if 'rating' in df.columns else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">RATING RATA-RATA</div>
                <div class="stat-value">{avg_rating:.2f} ★</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            unique_banks = df['bank_name'].nunique()
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">JUMLAH BANK</div>
                <div class="stat-value">{unique_banks}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # NAVIGASI TAB MENGGUNAKAN HORIZONTAL RADIO BUTTON
        st.markdown("### 📑 Navigasi")
        
        tab_choice = st.radio(
            "",
            [
                "📊 Sentimen",
                "🔍 TF-IDF",
                "📈 Matrix",
                "🔄 Proses",
                "📝 Data"
            ],
            horizontal=True,
            label_visibility="collapsed",
            key="tab_radio"
        )
        
        st.markdown("---")
        
        # TAB 1: Distribusi Sentimen
        if tab_choice == "📊 Sentimen":
            st.markdown("#### 📊 Analisis Sentimen (5 Kategori)")
            fig1 = plot_chart_5_kategori(df, "Analisis Sentimen Perbankan: Dari Puas hingga Tidak Puas")
            st.pyplot(fig1, use_container_width=False)
            
            st.markdown("""
            <div class="aggregation-box">
                <b>📌 Keterangan Agregasi ke 3 Kategori:</b><br>
                🔵 Puas = Puas + Ragu-ragu (Cenderung Puas)<br>
                🟡 Ragu-ragu = Ragu-ragu (Tetap Netral)<br>
                🔴 Tidak Puas = Tidak Puas + Ragu-ragu (Cenderung Tidak Puas)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📊 Sentimen Utama (3 Kategori)")
            fig2 = plot_chart_3_kategori(df, "Sentimen Utama")
            st.pyplot(fig2, use_container_width=False)
        
        # TAB 2: TF-IDF Analysis
        elif tab_choice == "🔍 TF-IDF":
            st.markdown("### 🔍 Analisis TF-IDF")
            st.caption("Analisis kata-kata yang paling berpengaruh dalam ulasan menggunakan metode TF-IDF")
            
            col1, col2 = st.columns(2)
            with col1:
                max_features = st.selectbox("Max Features", [500, 1000, 1500], index=1, key="tfidf_max_features")
            with col2:
                top_n = st.slider("Top N Words", 10, 30, 20, key="tfidf_top_n")
            
            if st.button("📊 Hitung TF-IDF", key="tfidf_btn"):
                with st.spinner("Menghitung TF-IDF..."):
                    df_tfidf = df[df['processed'].str.len() > 3].copy()
                    
                    if len(df_tfidf) > 50:
                        try:
                            vectorizer = TfidfVectorizer(max_features=max_features, min_df=2, max_df=0.95)
                            tfidf_matrix = vectorizer.fit_transform(df_tfidf['processed'])
                            
                            if tfidf_matrix.shape[1] > 0:
                                fig = plot_tfidf_top_words(vectorizer, tfidf_matrix, top_n, st.session_state.current_bank)
                                
                                feature_names = vectorizer.get_feature_names_out()
                                mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
                                tfidf_df_result = pd.DataFrame({
                                    'Term': feature_names,
                                    'Score': mean_tfidf
                                }).sort_values('Score', ascending=False).head(50)
                                
                                st.session_state.tfidf_clicked = True
                                st.session_state.tfidf_fig = fig
                                st.session_state.tfidf_df = tfidf_df_result
                                
                                st.pyplot(fig, use_container_width=True)
                                st.markdown("### Top 50 Kata")
                                st.dataframe(tfidf_df_result, use_container_width=True)
                            else:
                                st.warning("TF-IDF matrix kosong")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    else:
                        st.warning(f"Data tidak cukup (minimal 50 review)")
            
            elif st.session_state.get('tfidf_clicked', False):
                if st.session_state.tfidf_fig is not None:
                    st.pyplot(st.session_state.tfidf_fig)
                    st.markdown("### Top 50 Kata")
                    st.dataframe(st.session_state.tfidf_df, use_container_width=True)
                    st.info("Klik tombol 'Hitung TF-IDF' di atas untuk menghitung ulang dengan parameter yang berbeda.")
            else:
                st.info("Klik tombol 'Hitung TF-IDF' untuk memulai analisis kata-kata yang paling sering muncul dalam ulasan.")
        
        # TAB 3: Confusion Matrix
        elif tab_choice == "📈 Matrix":
            st.markdown("### 📈 Confusion Matrix (Evaluasi Lexicon vs Rating)")
            
            if 'rating' in df.columns and len(df) > 0:
                # Jika multi bank, pilih bank untuk confusion matrix
                if len(selected_banks) > 1:
                    selected_bank_cm = st.selectbox("Pilih Bank untuk Confusion Matrix", selected_banks, key="cm_bank_select")
                    df_cm = df[df['bank_name'] == selected_bank_cm].copy()
                else:
                    df_cm = df.copy()
                    selected_bank_cm = selected_banks[0] if selected_banks else "Bank"
                
                if len(df_cm) > 0:
                    def rating_to_sentiment(rating):
                        if rating >= 4:
                            return 'Puas'
                        elif rating <= 2:
                            return 'Tidak Puas'
                        return 'Ragu-ragu'
                    
                    df_cm['true_sentiment'] = df_cm['rating'].apply(rating_to_sentiment)
                    labels = ['Puas', 'Ragu-ragu', 'Tidak Puas']
                    cm = confusion_matrix(df_cm['true_sentiment'], df_cm['sentiment_3class'], labels=labels)
                    
                    accuracy = accuracy_score(df_cm['true_sentiment'], df_cm['sentiment_3class'])
                    precision = precision_score(df_cm['true_sentiment'], df_cm['sentiment_3class'], average='weighted', zero_division=0)
                    recall = recall_score(df_cm['true_sentiment'], df_cm['sentiment_3class'], average='weighted', zero_division=0)
                    f1 = f1_score(df_cm['true_sentiment'], df_cm['sentiment_3class'], average='weighted', zero_division=0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("AKURASI", f"{accuracy:.2%}")
                    with col2:
                        st.metric("PRECISION", f"{precision:.3f}")
                    with col3:
                        st.metric("RECALL", f"{recall:.3f}")
                    with col4:
                        st.metric("F1-SCORE", f"{f1:.3f}")
                    
                    fig = plot_confusion_matrix_heatmap(cm, labels, f"Confusion Matrix - {selected_bank_cm}")
                    st.pyplot(fig, use_container_width=True)
                    
                    total = len(df_cm)
                    correct = (df_cm['true_sentiment'] == df_cm['sentiment_3class']).sum()
                    st.info(f"**Interpretasi:** Dari {total:,} ulasan {selected_bank_cm}, model lexicon berhasil memprediksi {correct:,} ulasan dengan benar ({correct/total:.1%}).")
                else:
                    st.warning(f"Tidak cukup data untuk {selected_bank_cm}")
            else:
                st.warning("Data tidak memiliki kolom rating")
        
        # TAB 4: Proses & Metodologi
        elif tab_choice == "🔄 Proses":
            display_process_tab(df_raw, df, st.session_state.current_bank, st.session_state.is_multi_bank)
        
        # TAB 5: Data & Download
        elif tab_choice == "📝 Data":
            st.markdown("### 📝 Data Mentah (Sebelum Preprocessing)")
            st.write(f"Total data mentah: {len(df_raw):,} review")
            
            with st.expander("📄 Lihat Preview Data Mentah"):
                st.dataframe(df_raw.head(20), use_container_width=True)
            
            st.markdown("### 📊 Data Bersih (Setelah Preprocessing & Labeling)")
            st.write(f"Total data bersih: {len(df):,} review")
            
            with st.expander("📄 Lihat Preview Data Bersih"):
                display_cols = ['bank_name', 'content', 'rating', 'final_sentiment', 'sentiment_3class', 'pos_count', 'neg_count', 'processed']
                available_cols = [col for col in display_cols if col in df.columns]
                st.dataframe(df[available_cols].head(20), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 💾 Download Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📁 Data Mentah (Raw)")
                
                # Pilih format untuk data mentah
                format_raw = st.radio("Format Data Mentah", ["CSV", "Excel"], key="format_raw", horizontal=True)
                
                if format_raw == "CSV":
                    csv_raw = df_raw.to_csv(index=False).encode('utf-8')
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        "📥 Download CSV", 
                        csv_raw, 
                        f"data_mentah_{st.session_state.current_bank}_{timestamp}.csv", 
                        "text/csv", 
                        use_container_width=True
                    )
                else:
                    # Excel untuk data mentah
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_raw.to_excel(writer, sheet_name='Data_Mentah', index=False)
                    excel_data = output.getvalue()
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        "📥 Download Excel", 
                        excel_data, 
                        f"data_mentah_{st.session_state.current_bank}_{timestamp}.xlsx", 
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                        use_container_width=True
                    )
            
            with col2:
                st.markdown("#### ✅ Data Bersih (Processed)")
                
                # Pilih format untuk data bersih
                format_clean = st.radio("Format Data Bersih", ["CSV", "Excel"], key="format_clean", horizontal=True)
                
                if format_clean == "CSV":
                    csv_clean = df.to_csv(index=False).encode('utf-8')
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        "📥 Download CSV", 
                        csv_clean, 
                        f"data_bersih_{st.session_state.current_bank}_{timestamp}.csv", 
                        "text/csv", 
                        use_container_width=True
                    )
                else:
                    # Excel untuk data bersih dengan multiple sheets
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Data_Bersih', index=False)
                        
                        # Tambahkan sheet ringkasan
                        summary_data = []
                        for bank in df['bank_name'].unique():
                            bank_df = df[df['bank_name'] == bank]
                            summary_data.append({
                                'Bank': bank,
                                'Total Ulasan': len(bank_df),
                                'Puas': len(bank_df[bank_df['sentiment_3class'] == 'Puas']),
                                'Ragu-ragu': len(bank_df[bank_df['sentiment_3class'] == 'Ragu-ragu']),
                                'Tidak Puas': len(bank_df[bank_df['sentiment_3class'] == 'Tidak Puas']),
                                'Rata-rata Rating': round(bank_df['rating'].mean(), 2)
                            })
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Ringkasan', index=False)
                    
                    excel_data = output.getvalue()
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        "📥 Download Excel", 
                        excel_data, 
                        f"data_bersih_{st.session_state.current_bank}_{timestamp}.xlsx", 
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                        use_container_width=True
                    )
            
            st.info("💡 **Informasi:** File Excel untuk Data Bersih mencakup 2 sheet: 'Data_Bersih' (data lengkap) dan 'Ringkasan' (statistik per bank).")
    
    else:
        st.markdown(f"""
        <div style="background: #e8f4f8; border-radius: 16px; padding: 3rem; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👈</div>
            <h3 style="color: #1a4a6f;">Selamat Datang!</h3>
            <p style="color: #4a5568;">Silakan pilih mode analisis di sidebar dan klik <strong>"Proses Data"</strong> untuk memulai.</p>
            <p style="color: #4a5568; margin-top: 10px;">Tersedia <strong>19 Bank</strong> yang dapat dianalisis: 4 Bank Utama + 15 Bank Digital.</p>
            <p style="color: #4a5568; margin-top: 10px;">Data diambil secara <strong>real-time dari Google Play Store</strong> menggunakan ID aplikasi resmi.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # FOOTER
    st.markdown(f"""
    <div class="footer">
        <p>Sentiment Analysis Dashboard | 19 Mobile Banking Applications</p>
        <p style="font-size: 12px;">© 2026 - 4 Bank Utama (BCA • BRI • BTN • MANDIRI) + 15 Bank Digital</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()