import urllib.request
from langchain_openai import OpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import pandas as pd


import os
from dotenv import load_dotenv


load_dotenv()
api_key= os.getenv("OPENAI_API_KEY")


loader = CSVLoader(
    file_path="crawled_results_cleaned.csv",
    content_columns=["result"],  # 검색할 내용이 있는 컬럼
    metadata_columns=["coverletter"],  # 메타데이터로 사용할 컬럼
    encoding="utf-8"
)


documents = loader.load()
df= pd.set_index(keys='coverletter')
embedding = OpenAIEmbeddings()

vector_store = InMemoryVectorStore.from_documents(documents, embedding)

print(f'총 DB 안의 다큐 갯수: {len(documents)}')