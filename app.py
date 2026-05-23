import streamlit as st
import pandas as pd
import joblib
import re

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# PAGE CONFIG
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="🧠",
    layout="centered"
)

# LOAD MODEL
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# STOPWORDS
factory = StopWordRemoverFactory()
stop_words = factory.get_stop_words()

# HISTORY SESSION
if "history" not in st.session_state:

    st.session_state.history = []

# CLEANING
def clean_text(text):

    text = str(text).lower()

    # hapus URL
    text = re.sub(r"http\S+", "", text)

    # hapus mention
    text = re.sub(r"@\w+", "", text)

    # hapus simbol & angka
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # tokenisasi
    words = text.split()

    # hapus stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# EMOJI
emotion_emoji = {

    "sadness": "😢",

    "happy": "😄",

    "anger": "😡",

    "fear": "😨",

    "love": "❤️"
}

# DESCRIPTION
emotion_description = {

    "sadness": "Teks menunjukkan kesedihan atau tekanan emosional.",

    "happy": "Teks menunjukkan kebahagiaan atau rasa senang.",

    "anger": "Teks mengandung kemarahan atau frustrasi.",

    "fear": "Teks menunjukkan rasa takut atau kecemasan.",

    "love": "Teks mengandung kasih sayang atau rasa cinta."
}

# COLOR
emotion_color = {

    "sadness": "#4dabf7",

    "happy": "#ffd43b",

    "anger": "#ff6b6b",

    "fear": "#9775fa",

    "love": "#ff4d6d"
}

# CUSTOM CSS
st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 10px;
    color: white;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 40px;
}

.result-card {
    padding: 25px;
    border-radius: 20px;
    margin-top: 20px;
    color: white;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.description {
    font-size: 17px;
    margin-top: 10px;
    font-weight: normal;
}

textarea {
    border-radius: 15px !important;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    background-color: #2563eb;
    color: white;
    border: none;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

.history-card {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown(
    "<div class='title'>🧠 Emotion Detection AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Klasifikasi Emosi Tweet Bahasa Indonesia menggunakan TF-IDF dan Logistic Regression</div>",
    unsafe_allow_html=True
)

# INPUT
user_input = st.text_area(
    "Masukkan Tweet",
    height=150,
    placeholder="Contoh: aku takut gagal ujian besok..."
)

# BUTTON
if st.button("🔍 Prediksi Emosi"):

    if user_input.strip() == "":

        st.warning("Masukkan teks terlebih dahulu.")

    else:

        # CLEAN TEXT
        cleaned = clean_text(user_input)

        # VECTORIZER
        vectorized = vectorizer.transform([cleaned])

        # PREDICTION
        prediction = model.predict(vectorized)[0]

        probabilities = model.predict_proba(vectorized)[0]

        # UI DATA
        color = emotion_color[prediction]

        emoji = emotion_emoji[prediction]

        description = emotion_description[prediction]

        # RESULT CARD
        st.markdown(
            f"""
            <div class='result-card' style='background-color:{color};'>
                {emoji} Emosi Terdeteksi: {prediction.upper()}
                <div class='description'>
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # PREPROCESSING
        with st.expander("🧹 Lihat Hasil Preprocessing"):

            st.code(cleaned)

        # PROBABILITY DATAFRAME
        prob_df = pd.DataFrame({

            'Emotion': model.classes_,

            'Probability': probabilities

        })

        # SORT DESC
        prob_df = prob_df.sort_values(
            by='Probability',
            ascending=False
        )

        # CHART
        st.subheader("📊 Probabilitas Prediksi")

        st.bar_chart(
            prob_df.set_index('Emotion')
        )

        # TOP CONFIDENCE
        top_probability = round(
            max(probabilities) * 100,
            2
        )

        st.info(
            f"Tingkat keyakinan model: {top_probability}%"
        )

        # SAVE HISTORY
        st.session_state.history.append({

            "text": user_input,

            "emotion": prediction,

            "confidence": top_probability

        })

# HISTORY
if len(st.session_state.history) > 0:

    st.subheader("🕘 Riwayat Prediksi")

    for item in reversed(
        st.session_state.history[-5:]
    ):

        st.markdown(f"""

        <div class='history-card'>

        <b>Teks:</b><br>
        {item['text']}

        <br><br>

        <b>Emosi:</b>
        {item['emotion']}

        <br><br>

        <b>Confidence:</b>
        {item['confidence']}%

        </div>

        """, unsafe_allow_html=True)

    # CLEAR HISTORY
    if st.button("🗑️ Clear History"):

        st.session_state.history = []

        st.rerun()

# FOOTER
st.markdown("""
<br>
<hr>
<center>
dibuat dengan penderitaan, kopi, dan machine learning ☕
</center>
""", unsafe_allow_html=True)