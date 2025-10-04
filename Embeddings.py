import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Credentials & Configuration ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
INDEX_NAME = os.environ.get('INDEX_NAME', 'hr-vector-search-index')
EMBEDDING_DIMENSION = 768

# --- Initialize Models ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# --- Initialize Pinecone ---
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Check if the index exists
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' does not exist. Creating it now...")
    spec = ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=spec
    )
    print(f"Waiting for index '{INDEX_NAME}' to be ready...")
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(5)
    print(f"Index '{INDEX_NAME}' is ready.")
else:
    print(f"Index '{INDEX_NAME}' already exists.")

# --- Initialize Pinecone Vector Store ---
index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

print("Pinecone Index Stats:", index.describe_index_stats())
print("Langchain PineconeVectorStore initialized:", vector_store)

# --- Folder containing the .txt files ---
folder_path = r"C:\Users\alokm\OneDrive\Documents\Machine Learning\Assignment\HR Chatbot\Data\Pages_Text"

# --- Text Splitter ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

# --- Process each txt file and upload one by one ---
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        print(f"\nLoading {file_path} ...")
        loader = TextLoader(file_path, encoding="utf-8")  # UTF-8 encoding for Korean text
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        print(f"Split into {len(splits)} chunks.")

        print(f"Adding chunks from {filename} to Pinecone Vector Store...")
        try:
            document_ids = vector_store.add_documents(documents=splits)
            print(f"Successfully added {len(splits)} chunks from {filename}.")
        except Exception as e:
            print(f"Error adding {filename} to Pinecone Vector Store: {e}")