# Loading documents from a directory with LangChain
from langchain.document_loaders import DirectoryLoader
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.vectorstores import Pinecone
# import docx2txt
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore


#load openai key
load_dotenv()

def read_pdf(file):
  pdf_reader = PdfReader(file)
  text = ''
  for page in pdf_reader.pages:
    text += page.extract_text()
  return text


def split_docs(documents,chunk_size=1000,chunk_overlap=300):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_text(documents)
  return docs


def text_split(raw_text):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=300,
    length_function=len
  )
  docs = text_splitter.split_text(raw_text)

  return docs

pineconeIndex = st.text_input("Pinecone Index Name")
print('pineconeIndex ::::::::::', pineconeIndex)
st.header("Upload Your File 🗃️")
docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
submit = st.button("Upload")

if docx_file is not None and submit :
    if pineconeIndex is not None and pineconeIndex != '':
        file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
        print('file_details :::::::', file_details)

        # Check File Type
        if docx_file.type == "text/plain":
            raw_text = str(docx_file.read(),"utf-8")

        elif docx_file.type == "application/pdf":
            raw_text = read_pdf(docx_file)
                
        splitedText = text_split(raw_text)

        # embeddings
        embeddings = OpenAIEmbeddings()

        #upload data on pinecone
        PineconeVectorStore.from_texts(splitedText, embeddings, index_name=pineconeIndex)

        st.toast('File uploaded !', icon='🎉')
        st.success('File uploaded Done!')

    else:
        st.error('Error', icon="🚨")