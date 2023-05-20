from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Your PDF")
    st.header("Ask Your PDF ")

    # Uploading PDF
    pdf = st.file_uploader("Upload your pdf", type="pdf")

    # Extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)	
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # splitting into chunks 
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)

        #create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_questions = st.text_input("Ask a questions about your PDF:")
        if user_questions:
            docs = knowledge_base.similarity_search(user_questions)

            llm= OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb: 
                response = chain.run(input_documents=docs, question=user_questions)
                print(cb)
            st.write(response)

if __name__ =='__main__':
    main()