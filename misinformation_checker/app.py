import streamlit as st
from rag_checker import classify_article


# Page title
st.set_page_config(page_title="VERITAS-AI", page_icon="🔍")


st.title("VERITAS-AI")
st.subheader("AI-Powered Misinformation Checker")


st.write("""
Paste an article or claim below. The system will compare it against
trusted sources and determine whether it appears reliable,
questionable, or false.
""")


# User input
article = st.text_area(
   "Enter an article or claim:",
   height=250
)


# Button
if st.button("Check Credibility"):


   if article.strip() == "":
       st.warning("Please enter some text.")
   else:


       with st.spinner("Analyzing article..."):


           classification, evidence = classify_article(article)


       st.success("Analysis Complete")


       st.header("Credibility Assessment")


       st.write(f"**Result:** {classification}")


       st.header("Supporting Evidence")


       for i, item in enumerate(evidence, start=1):


           st.subheader(f"Evidence {i}")


           st.write(f"**Source:** {item['source']}")


           st.write(item["text"])


           st.divider()

