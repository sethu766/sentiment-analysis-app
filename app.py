import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


st.set_page_config(page_title="AI Mood Detector", page_icon="🤖")


st.title(" AI Mood Detector ")
st.markdown("### Built using Machine Learning & NLP")
st.write("Enter a sentence and find out its sentiment!")


user_input = st.text_area(" Enter your text below:")


if st.button("Check Mood"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        confidence = model.predict_proba(transformed)[0]

        # Display result
        if prediction == 1:
            st.success(" Positive Sentiment")
            st.write(f"Confidence: {round(confidence[1]*100, 2)}%")
            st.balloons()
        else:
            st.error(" Negative Sentiment")
            st.write(f"Confidence: {round(confidence[0]*100, 2)}%")