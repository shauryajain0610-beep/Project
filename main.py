import streamlit as st
import pickle
import pandas as pd
import webbrowser

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# -------------------- LOAD TRAINED MODEL --------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------- CLASSIFIER FUNCTION --------------------
def predict_news(text):
    prediction = model(text)[0]
    return prediction

def reasoning(prediction):
    if prediction == "Fake":
        return "This article shows patterns commonly found in misinformation such as exaggerated claims, emotional tone, and lack of credible sources."
    else:
        return "This content contains strong factual indicators and writing style similar to trusted news sources."

def advice(prediction):
    if prediction == "Fake":
        return "Check multiple fact-checking websites, confirm author identities, and verify references before sharing."
    else:
        return "Still verify with trusted sources before spreading. Stay aware and think critically."

def external_links(topic):
    return [
        f"https://www.google.com/search?q={topic}+news",
        f"https://www.bbc.com/search?q={topic}",
        f"https://www.reuters.com/site-search/?query={topic}",
        f"https://timesofindia.indiatimes.com/topic/{topic}"
    ]

# -------------------- STREAMLIT UI --------------------
st.title("📰 Fake News Detection System")
st.write("Enter news content and analyze if it's Real or Fake with explainable reasoning.")

input_type = st.radio("Choose Input Type:", ["Headline", "Full Article"])

user_text = st.text_area(f"Enter {input_type} here:")

topic = st.text_input("Enter Topic / Keyword for external links (optional):")

if st.button("Analyze Now"):
    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        prediction = predict_news(user_text)
        st.subheader("🔍 Prediction Result")
        st.success(f"**This news appears to be: {prediction}**")

        st.subheader("🧠 Reasoning")
        st.write(reasoning(prediction))

        st.subheader("💡 Advice")
        st.info(advice(prediction))

        if topic != "":
            st.subheader("🔗 External Verified Sources")
            for link in external_links(topic):
                st.markdown(f"[Open Link]({link})")

        st.markdown("---")
        st.markdown("### 🙏 Thank you for using the Fake News Detection App!")
