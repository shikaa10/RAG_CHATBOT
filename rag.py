import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

#  Use updated huggingface embeddings/LLM
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import tempfile
import os

st.set_page_config(page_title=" RAG Chatbot", layout="wide")
st.title(" RAG Chatbot using FLAN-T5")

uploaded_files = st.file_uploader("Upload 1 to 3 text files", type=["txt"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) > 3:
    st.warning("Please upload up to 3 files only.")
    st.stop()

if st.button(" Build RAG Bot") and uploaded_files:
    with st.spinner("Processing..."):
        all_docs = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            loader = TextLoader(tmp_path, encoding="ISO-8859-1")
            all_docs.extend(loader.load())
            os.remove(tmp_path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)

        #  Use up-to-date embedding class
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Load FLAN-T5 model
        model_id = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,         # Increase maximum response length
    do_sample=True,         # Enable sampling for more natural responses
    temperature=0.7,        # Controls randomness; lower = more deterministic
    top_p=0.9,              # Nucleus sampling
    repetition_penalty=1.2  # Discourage repeating phrases
)

        llm = HuggingFacePipeline(pipeline=pipe)

        # Save the chain in session_state
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=vectorstore.as_retriever()
        )

    st.success("RAG Bot is ready! Ask your question below ")

#  Ask a question
if "qa_chain" in st.session_state:
    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Answering..."):
            formatted_q = f"Answer the following question in a detailed paragraph:\n\n{question}"
            answer = st.session_state.qa_chain.run(formatted_q)
            st.markdown(f"*Answer:* {answer}")
else:
    st.info("Upload 1–3 files and click 'Build RAG Bot' to activate the chatbot.")