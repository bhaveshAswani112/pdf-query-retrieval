import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import streamlit as st
import openai
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# API Key Inputs
open_ai_api_key = st.sidebar.text_input(
    "Enter your open ai api key",
    help="You can find it on https://platform.openai.com/api-keys and it will be used for creating embeddings for your pdf file",
    type="password"
)
groq_api_key = st.sidebar.text_input(
    "Enter your groq api key",
    help="It's free and you can find it on https://console.groq.com/keys",
    type="password"
)

# Model Selection
models = st.sidebar.selectbox("Select the model", options=["llama3-8b-8192", "gemma2-7b-it"])

if groq_api_key and open_ai_api_key:
    try:
        llm = ChatGroq(model="gemma-7b-it", groq_api_key=groq_api_key)
        if "embedded" not in st.session_state:
            st.session_state.embedded = False

        if "store" not in st.session_state:
            st.session_state.store = {}

        session_id = st.text_input("Session Id", value="default-session")

        def create_vector_store():
            try:
                if not open_ai_api_key:
                    return "Please provide your open ai api key"
                if "vectors" not in st.session_state:
                    st.session_state.loader = PyPDFDirectoryLoader(path="files")
                    st.session_state.documents = st.session_state.loader.load()
                    st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
                    st.session_state.docs = st.session_state.splitter.split_documents(st.session_state.documents)
                    st.session_state.embeddings = OpenAIEmbeddings(openai_api_key=open_ai_api_key)
                    st.session_state.vectors = FAISS.from_documents(st.session_state.docs, embedding=st.session_state.embeddings)
                    st.session_state.embedded = True
                    st.session_state.retriever = st.session_state.vectors.as_retriever()
                    return "Your document is ready for query."
                else:
                    return "Your document is already ready for query."
            except Exception as e:
                print("Error in create_vector_store:", e)
                return "Error in processing your document."

        pdf = st.file_uploader(label="Upload your PDF file", type=["pdf"])

        if pdf:
            try:
                if not os.path.exists("files"):
                    os.makedirs("files")
                file_path = os.path.join("files", pdf.name)
                with open(file_path, mode="wb") as f:
                    f.write(pdf.getbuffer())
                st.success(f"File uploaded successfully: {pdf.name}")
            except Exception as e:
                print("Error uploading file:", e)
                st.error("Error uploading file")

        if st.button("Ready Document"):
            st.write(create_vector_store())

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        user_input = st.text_input("Enter your query from provided PDF")

        if st.session_state.embedded:
            try:
                question_prompt_content = '''Given the chat history and latest user question which reference context in chat history, formulate a standalone question which can be understood without chat history. Do not answer the question, just reformulate it if needed otherwise return as it is.               
                '''

                question_prompt = ChatPromptTemplate.from_messages([
                    ("system", question_prompt_content),
                    MessagesPlaceholder("history"),
                    ("human", "{input}")
                ])
                history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, question_prompt)
                # print("No error after history_aware_retriever")

                system_prompt_content = '''You are assistant for question answer tasks , try to answer the question based on following pieces of retrieved context. 
                    {context}
                '''
                system_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt_content),
                    MessagesPlaceholder("history"),
                    ("human", "{input}")
                ])
                qa_chain = create_stuff_documents_chain(llm, system_prompt)
                # print("No error after create_stuff_documents_chain")
                rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
                # print("No error after create_retrieval_chain")
                retrieval_chain = RunnableWithMessageHistory(rag_chain, get_session_history, input_messages_key="input", history_messages_key="history", output_messages_key="answer")
                # print("No error after RunnableWithMessageHistory")
            except Exception as e:
                # print("Error in retrieval chain setup:", e)
                st.error("Error in retrieval chain setup")

        if user_input and st.session_state.embedded:
            try:
                response = retrieval_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
                st.write(response["answer"])

                with st.expander("Document Similarity Search"):
                    for i, doc in enumerate(response['context']):
                        st.write(doc.page_content)
                        st.write("---------------------")
            except Exception as e:
                print("Error processing query:", e)
                st.error("Error processing query")
        elif user_input and not st.session_state.embedded:
            st.write("Please provide the document and make it ready for the query.")
    except Exception as e:
        print("General error:", e)
        st.error("Invalid API key")
