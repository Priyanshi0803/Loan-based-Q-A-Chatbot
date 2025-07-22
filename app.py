import streamlit as st
from rag import retrieve, generate_answer

st.set_page_config(page_title="LoanBot: Your Loan Query Assistant")
st.title(" Loan based Q&A Chatbot")

user_question = st.text_input("Ask a question:")

if user_question:
    with st.spinner("Searching knowledge base..."):
        relevant_chunks = retrieve(user_question)
        context = "\n".join(relevant_chunks)

    with st.spinner("Generating response..."):
        answer = generate_answer(context, user_question)

    st.markdown("### Answer:")
    st.write(answer)

    with st.expander("📂 Retrieved context"):
        st.write(context)
