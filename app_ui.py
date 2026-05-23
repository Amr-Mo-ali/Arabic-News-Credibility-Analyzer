import streamlit as st
import requests
import time

# Page config
st.set_page_config(
    page_title="Arabic News Credibility Analyzer",
    page_icon="🔍",
    layout="centered"
)

# Header
st.title("🔍 Arabic News Credibility Analyzer")
st.markdown("**محلل مصداقية الأخبار العربية** — Powered by AraBERT")
st.divider()

# Input
user_input = st.text_area(
    "أدخل نص الخبر العربي هنا:",
    height=200,
    placeholder="مثال: أكد المسؤولون أن..."
)

# Helper function with retry
def call_api(text, retries=5, delay=3):
    for i in range(retries):
        try:
            response = requests.post(
                "http://api:8000/predict/",
                json={"text": text},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP Error: {e}")

# Button
if st.button("🔎 تحليل الخبر", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ من فضلك أدخل نص الخبر أولاً.")
    else:
        with st.spinner("جاري التحليل..."):
            try:
                result = call_api(user_input)

                st.divider()
                st.subheader("📊 نتيجة التحليل:")

                if result['label'] == 'fake':
                    st.error("🔴 الخبر: **مزيف (Fake)**")
                else:
                    st.success("🟢 الخبر: **حقيقي (Real)**")

                st.metric(
                    label="نسبة الثقة",
                    value=f"{result['confidence']}%"
                )

            except requests.exceptions.ConnectionError:
                st.error("❌ تعذر الاتصال بالـ API — تأكد إن الـ API container شغال.")
            except Exception as e:
                st.error(f"❌ خطأ: {e}")