#api key setup
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

#embedding model setup
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

#perform embedding of tokens
vectors = embedding_model.embed_query("water")
vectors1 = embedding_model.embed_query("tv")

print("water",vectors[:5])
print("TV",vectors1[:5])


print('dimensions:',len(vectors))