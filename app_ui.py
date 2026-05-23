import streamlit as st
import requests

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

# Button
if st.button("🔎 تحليل الخبر", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ من فضلك أدخل نص الخبر أولاً.")
    else:
        with st.spinner("جاري التحليل..."):
            try:
                response = requests.post(
                    "http://api:8000/predict/",
                    json={"text": user_input}
                )
                result = response.json()

                st.divider()
                st.subheader("📊 نتيجة التحليل:")

                if result['label'] == 'fake':
                    st.error(f"🔴 الخبر: **مزيف (Fake)**")
                else:
                    st.success(f"🟢 الخبر: **حقيقي (Real)**")

                st.metric(
                    label="نسبة الثقة",
                    value=f"{result['confidence']}%"
                )

            except Exception as e:
                st.error(f"❌ خطأ في الاتصال بالـ API: {e}")