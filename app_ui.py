import streamlit as st


st.title("Arabic News Credibility Analyzer")
st.write("Enter an Arabic news article to analyze its credibility.")
user_input = st.text_area("Enter the news article here:")
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a news article to analyze.")
    else:
        from src.predict import predict
        result = predict(user_input)
        st.subheader("Prediction Result:")
        st.write(f"**Label:** {result['label'].capitalize()}")
        st.write(f"**Confidence:** {result['confidence']}%")