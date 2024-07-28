import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import streamlit as st
import openai



prompt = ChatPromptTemplate.from_template(
    '''Try to answer the question based on provided context and try to be accurate.

    <context>
    {context}
    <context>
    Question: {input}
    '''
)

open_ai_api_key = st.sidebar.text_input("Enter your open ai api key",help="You can find it on https://platform.openai.com/api-keys and it will be used for creating embeddings for your pdf file",type="password")
groq_api_key = st.sidebar.text_input("Enter your groq api key",help="It's free and you can find it on https://console.groq.com/keys",type="password")




    

models = st.sidebar.selectbox("Select the model",options=["llama3-8b-8192","gemma2-7b-it"])




if groq_api_key and open_ai_api_key:
    try : 
        llm = ChatGroq(model="gemma-7b-it", groq_api_key=groq_api_key)
        openai.api_key = open_ai_api_key

        if "embedded" not in st.session_state:
            st.session_state.embedded = False

        def create_vector_store():
            try:
                if not open_ai_api_key :
                    return "Please provide your open ai api key"
                if "vectors" not in st.session_state:
                    st.session_state.loader = PyPDFDirectoryLoader(path="files")
                    st.session_state.documents = st.session_state.loader.load()
                    st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    st.session_state.docs = st.session_state.splitter.split_documents(st.session_state.documents)
                    st.session_state.embeddings = OpenAIEmbeddings()
                    st.session_state.vectors = FAISS.from_documents(st.session_state.docs, embedding=st.session_state.embeddings)
                    st.session_state.embedded = True
                    return "Your document is ready for query."
                else:
                    return "Your document is already ready for query."
            except Exception as e:
                print(e)
                return "Error in processing your document."

        pdf = st.file_uploader(label="Upload your PDF file", type=["pdf"])

        if pdf:
            if not os.path.exists("files"):
                os.makedirs("files")
            file_path = os.path.join("files", pdf.name)
            with open(file_path, mode="wb") as f:
                f.write(pdf.getbuffer())
            st.success(f"File uploaded successfully: {pdf.name}")

        if st.button("Ready Document"):
            st.write(create_vector_store())

        user_input = st.text_input("Enter your query from provided PDF")

        if user_input and st.session_state.embedded:
            docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, docs_chain)
            response = retrieval_chain.invoke({"input": user_input})
            st.write(response["answer"])

            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write("---------------------")
        elif user_input and not st.session_state.embedded:
            st.write("Please provide the document and make it ready for the query.")
    except Exception as e:
        st.error("Invalid api key")
