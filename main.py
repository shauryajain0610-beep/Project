import streamlit as st
import pickle


st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Choose input type and analyze whether the content seems Real or Fake.")

# -----------------------------
# LOAD PRE-TRAINED MODEL AND VECTORIZER
# -----------------------------
# Ensure you have these files in the same folder:
# - model.pkl : trained classifier (e.g., LogisticRegression, RandomForest)
# - vectorizer.pkl : TfidfVectorizer used during training
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# STREAMLIT INPUT AREA
# -----------------------------
st.header("📝 Select Input Type")
choice = st.radio("Select what you want to enter:", ["Headline Only", "Full Article"])

headline = ""
article = ""

if choice == "Headline Only":
    headline = st.text_input("Enter News Headline")
else:
    headline = st.text_input("Headline (optional)")
    article = st.text_area("Enter Full Article Text", height=180)

# -----------------------------
# PREDICTION LOGIC
# -----------------------------
if st.button("Analyze"):
    combined_text = (headline + " " + article).strip()

    if not combined_text:
        st.warning("⚠ Please enter some text first.")
    else:
        # Transform text using the same vectorizer
        text_vector = vectorizer.transform([combined_text])
        prediction = model.predict(text_vector)[0]  # "FAKE" or "REAL"
        prediction_proba = model.predict_proba(text_vector)[0]  # Optional probability

        # -----------------------------
        # DYNAMIC REASONING & ADVICE
        # -----------------------------
        if prediction.upper() == "FAKE":
            reasoning = "The ML model predicts this news as FAKE based on patterns learned from past datasets."
            advice = """
            ❌ **Advice if Fake:**  
            - Immediately verify using trusted fact-checking sites  
            - Do NOT share this information unless verified  
            - Look for official government or credible news sources  
            """
            sources = """
            - [Alt News](https://www.altnews.in/)  
            - [BOOM Fact Check](https://www.boomlive.in/)  
            - [Factly](https://factly.in/)  
            """
            st.error("❌ FAKE NEWS")

        else:
            reasoning = "The ML model predicts this news as REAL based on learned authentic news patterns."
            advice = """
            ✔ **Advice if Real:**  
            - Still check original source for any updates  
            - Share responsibly from official/reputed outlets  
            - Verify facts from authentic government or national agencies  
            """
            sources = """
            - [Reuters Official News](https://www.reuters.com/)  
            - [BBC News](https://www.bbc.com/)  
            - [The Hindu](https://www.thehindu.com/)  
            """
            st.success("✔ REAL NEWS")

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        st.subheader("🧠 Reasoning")
        st.write(reasoning)

        st.subheader("💡 Smart Advice")
        st.write(advice)

        st.subheader("🔗 Trusted Verification Sources")
        st.markdown(sources)

        st.info("This ML-based model is trained on a Kaggle Fake vs Real News dataset for better accuracy.")
