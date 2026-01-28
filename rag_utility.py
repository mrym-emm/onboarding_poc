# to access the path of directories
import os

# for accessing the environment var:- apikeys etc
from dotenv import load_dotenv

# from langchain_community.document_loaders import (
#     UnstructuredPDFLoader,
# )  # read the files the user is loading

from langchain_community.document_loaders import PyPDFLoader

# used to split the texts into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# the embedding model tht ill be using to embbed both user query & the pdf
from langchain_huggingface import HuggingFaceEmbeddings

# vector embeddings stored in chromadb
from langchain_chroma import Chroma

# the llm ill be using
from langchain_groq import ChatGroq

# the chain to help send the qa to the llm
from langchain_classic.chains import RetrievalQA

# load environment variables from .env file
load_dotenv()

# getting the absolute path of the file thts gonna be uplaoded, this alter we concatanate with the pdf to get full path
working_dir = os.path.dirname(os.path.abspath((__file__)))


# load the embedding model & llm tht is going to be used
emebedding = HuggingFaceEmbeddings()

# using llama-3.3-70B from groq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)


# function for document ingestion
def process_document_to_chroma_db(file_name):
    # load the pdf document using UnstructuredPDFLoader
    # this will save the pdf in the work dir
    loader = PyPDFLoader(
        f"{working_dir}/{file_name}",
    )
    documents = loader.load()

    # split texts into embeddings, member ni returns an object
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # gets the splitted texts vector embeddings
    texts = text_splitter.split_documents(documents)

    # store the vector embeddings into chroma db
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=emebedding,
        # saving the vector directory in working dir
        persist_directory=f"{working_dir}/doc_vectorstore",
    )

    return 0


# function for question answering, essentially retrieving the answer for
# users question and match from the vectordb
def answer_question(user_question):
    # load the persistent memory from chromadb
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=emebedding,
    )

    # create a retriever for doc search
    retriever = vectordb.as_retriever()

    # create a retrievalQA chain to send the questions to the llm swer user question using da llm
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    # collecting the answer
    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer
