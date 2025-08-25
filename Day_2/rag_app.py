#setup gemini API key
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

#load data using document loader
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('')
docs = loader.load()

#Split the document into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_documents(docs)

#Embedding Model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddingsModel = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Mongo DB as Vector DB
import getpass
MONGODB_CLUSTER_URI = getpass.getpass('MONGODB_CLUSTER_URI')
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
client = MongoClient(MONGODB_CLUSTER_URI)
DB_NAME = "EmbeddingsDB"
COLLECTION_NAME = "EmbeddingVector-3"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "Demo-Embedding-Index-3"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddingsModel,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn= "cosine"
)
vector_store.create_vector_search_index(dimensions=768)
from uuid import uuid4
uuids = [str(uuid4()) for _ in range(len(texts))]
vector_store.add_documents(documents=texts, ids=uuids)





