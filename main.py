import os 
from dotenv import load_dotenv
load_dotenv() #load all env variables 
import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser  


#load the nvidia api key 
os.environ["NVIDIA_API_KEY"]=os.getenv("NVIDIA_API_KEY")    

#Call the llm model
llm=ChatNVIDIA(model="meta/llama-3.3-70b-instruct")  #NVIDIA NIM inferencing

#vector store embeddings
def vector_embeddings():
    if 'vectors' not in st.session_state:        
        #initialize the loader
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs=st.session_state.loader.load()
        #initialize the splitter
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        #intialize the embeddings and vector_store
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

# ----------- DESIGN STREAMLIT APP  ----------------# 

st.set_page_config("AskAwayApp")
st.title("Nvidia Nim Askaway PDF Analyzer")

#design prompt for LLM model
prompt_template=ChatPromptTemplate.from_template(
"""
Answer the question based on the provided context only.
Please provide the most accurate response based on the question.
(context)
{context}
(/context)
Question:{question}

"""
)
prompt=st.text_input("Enter any question from the documents")

if st.button("Document Embedding"):
    vector_embeddings()
    st.write("FAISS Vector Store created successfully using Nvidia Embeddings")

import time

if prompt:
    # Retrieve relevant documents
    retrieved_docs = st.session_state.vectors.as_retriever().get_relevant_documents(prompt)
    retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])  # Combine docs into text
    
    # Create the retrieval chain correctly
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(st.session_state.vectors.as_retriever(), document_chain)

    # Start the timer
    start = time.process_time()

    # Pass input in the correct format
    response = retrieval_chain.invoke({"input": {"context": retrieved_context, "question": prompt}})

    # Print response time
    print("Response Time:", time.process_time() - start)

    # Display LLM response
    st.write(response.get('answer', "No answer returned."))

    # Display document similarity search
    with st.expander("Document Similarity Search"):
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                st.write(f"**Document {i+1}:**")
                st.write(doc.page_content)  # Display content
                st.write("-------------------------------------")
        else:
            st.warning("No relevant documents found for this query.")




    



        

        
    