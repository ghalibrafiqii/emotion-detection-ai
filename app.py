import streamlit as st
import pandas as pd
import joblib
import re

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="🧠",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# =========================
# STOPWORDS
# =========================

factory = StopWordRemoverFactory()
stop_words = factory.get_stop_words()

# =========================
# SESSION HISTORY
# =========================

if "history" not in st.session_state:

    st.session_state.history = []

# =========================
# CLEANING FUNCTION
# =========================

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

# =========================
# EMOTION DATA
# =========================

emotion_emoji = {

    "sadness": "😢",

    "happy": "😄",

    "anger": "😡",

    "fear": "😨",

    "love": "❤️"
}

emotion_description = {

    "sadness": "Tweet menunjukkan kesedihan atau tekanan emosional.",

    "happy": "Tweet menunjukkan kebahagiaan atau rasa senang.",

    "anger": "Tweet mengandung kemarahan atau frustrasi.",

    "fear": "Tweet menunjukkan rasa takut atau kecemasan.",

    "love": "Tweet mengandung kasih sayang atau rasa cinta."
}

emotion_color = {

    "sadness": "#38bdf8",

    "happy": "#facc15",

    "anger": "#ef4444",

    "fear": "#8b5cf6",

    "love": "#ec4899"
}

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {

    font-family: 'Poppins', sans-serif;
}

.stApp {

    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );

    color: white;
}

.main-title {

    font-size: 52px;

    font-weight: 700;

    text-align: center;

    margin-top: 20px;

    background: linear-gradient(
        to right,
        #60a5fa,
        #c084fc
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.subtitle {

    text-align: center;

    color: #cbd5e1;

    margin-bottom: 40px;

    font-size: 18px;
}

.result-card {

    padding: 30px;

    border-radius: 25px;

    text-align: center;

    color: white;

    margin-top: 20px;

    box-shadow: 0px 10px 35px rgba(0,0,0,0.35);

    animation: fadeIn 0.5s ease-in-out;
}

.metric-card {

    background: rgba(255,255,255,0.08);

    padding: 20px;

    border-radius: 20px;

    text-align: center;

    backdrop-filter: blur(10px);

    border: 1px solid rgba(255,255,255,0.1);

    margin-bottom: 15px;
}

.history-card {

    background: rgba(255,255,255,0.08);

    padding: 14px 16px;

    border-radius: 16px;

    margin-bottom: 12px;

    backdrop-filter: blur(10px);

    border: 1px solid rgba(255,255,255,0.08);

    line-height: 1.5;

    word-wrap: break-word;
}

textarea {

    border-radius: 20px !important;

    background-color: rgba(255,255,255,0.05) !important;

    color: white !important;
}

div.stButton > button {

    width: 100%;

    height: 55px;

    border-radius: 16px;

    font-size: 18px;

    font-weight: 600;

    border: none;

    background: linear-gradient(
        to right,
        #3b82f6,
        #8b5cf6
    );

    color: white;

    transition: 0.3s;
}

div.stButton > button:hover {

    transform: scale(1.02);

    opacity: 0.9;
}

@keyframes fadeIn {

    from {

        opacity: 0;

        transform: translateY(10px);
    }

    to {

        opacity: 1;

        transform: translateY(0);
    }
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown(
    "<div class='main-title'>🧠 Emotion Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Klasifikasi Emosi Tweet Bahasa Indonesia menggunakan TF-IDF dan Logistic Regression</div>",
    unsafe_allow_html=True
)

# =========================
# LAYOUT
# =========================

left_col, right_col = st.columns([2,1])

# =========================
# LEFT SIDE
# =========================

with left_col:

    user_input = st.text_area(
        "Masukkan Tweet",
        height=220,
        placeholder="contoh: aku capek banget sama semuanya hari ini..."
    )

    if st.button("🔍 Prediksi Emosi"):

        if user_input.strip() == "":

            st.warning("Masukkan teks terlebih dahulu.")

        else:

            # CLEAN TEXT
            cleaned = clean_text(user_input)

            # TF-IDF
            vectorized = vectorizer.transform([cleaned])

            # PREDICT
            prediction = model.predict(vectorized)[0]

            probabilities = model.predict_proba(vectorized)[0]

            # UI DATA
            color = emotion_color[prediction]

            emoji = emotion_emoji[prediction]

            description = emotion_description[prediction]

            top_probability = round(
                max(probabilities) * 100,
                2
            )

            # RESULT CARD

            st.markdown(
                f"""
                <div class='result-card'
                style='background:{color};'>

                <h1 style='font-size:60px;'>
                {emoji}
                </h1>

                <h2>
                {prediction.upper()}
                </h2>

                <p style='font-size:18px;'>
                {description}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            # METRICS

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    f"""
                    <div class='metric-card'>

                    <h3>Confidence</h3>

                    <h1>{top_probability}%</h1>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    f"""
                    <div class='metric-card'>

                    <h3>Emotion</h3>

                    <h1>{emoji}</h1>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # PROBABILITY DATAFRAME

            prob_df = pd.DataFrame({

                "Emotion": model.classes_,

                "Probability": probabilities
            })

            prob_df = prob_df.sort_values(
                by="Probability",
                ascending=False
            )

            # CHART

            st.subheader("📊 Probability Distribution")

            st.bar_chart(
                prob_df.set_index("Emotion")
            )

            # PREPROCESSING

            with st.expander("🧹 Lihat Hasil Preprocessing"):

                st.code(cleaned)

            # SAVE HISTORY

            st.session_state.history.append({

                "text": user_input,

                "emotion": prediction,

                "confidence": top_probability
            })

# =========================
# RIGHT SIDE
# =========================

with right_col:

    st.subheader("🕘 Prediction History")

    if len(st.session_state.history) == 0:

        st.info("Belum ada history prediksi.")

    else:

        for item in reversed(
            st.session_state.history[-5:]
        ):

            emoji = emotion_emoji[item["emotion"]]

            st.markdown(
                f"""
                <div class='history-card'>

                <div style='font-size:18px;font-weight:600;'>

                {emoji} {item['emotion'].upper()}

                </div>

                <div style='margin-top:8px;font-size:15px;'>

                {item['text']}

                </div>

                <div style='margin-top:10px;color:#cbd5e1;font-size:14px;'>

                Confidence:
                <b>{item['confidence']}%</b>

                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        if st.button("🗑️ Clear History"):

            st.session_state.history = []

            st.rerun()

# =========================
# FOOTER
# =========================

st.markdown("""
<br><br>
<hr>

<center>

dibuat dengan machine learning,
kopi,
dan debugging yang tidak manusiawi ☕

</center>
""", unsafe_allow_html=True)
