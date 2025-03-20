import os
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables from .env file
load_dotenv()

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMADB_PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIR", "./chroma_db")  # Default storage path
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # Default to 500 if missing

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load PDF and split into chunks
loader = PyPDFLoader("the_seven_habits.pdf")  # Correct loader for PDFs
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Initialize ChromaDB
db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMADB_PERSIST_DIR)
retriever = db.as_retriever(search_kwargs={"k": 10})  # Retrieve top 10 relevant chunks

# Initialize Gemini model
chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)

# Create RAG pipeline
qa_chain = RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever)

def delete_chroma_db():
    """Deletes the ChromaDB database folder."""
    if os.path.exists(CHROMADB_PERSIST_DIR):
        confirm = input("Are you sure you want to delete ChromaDB storage? (yes/no): ")
        if confirm.lower() == "yes":
            shutil.rmtree(CHROMADB_PERSIST_DIR)
            print("ChromaDB storage deleted successfully!")
        else:
            print("Deletion canceled.")
    else:
        print(" ChromaDB storage not found.")

# Interactive chat
print("RAG Chat with Gemini & ChromaDB! (type 'exit' to stop)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    elif user_input.lower() == "delete chroma":
        delete_chroma_db()
    else:
        response = qa_chain.run(user_input)
        print(f"Gemini: {response}")