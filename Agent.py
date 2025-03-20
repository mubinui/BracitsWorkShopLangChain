import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMADB_DIR", "./chromadb")  # Default path for ChromaDB
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))

# Initialize Gemini API client
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
chat_model_selector = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
chat_model_rag = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load and process documents
loader = PyPDFLoader("the_seven_habits.pdf")  # Correct loader for PDFs
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Initialize ChromaDB
db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DB_PATH)
retriever = db.as_retriever(search_kwargs={"k": 10})
  # Fetch top 10 similar docs
qa_chain = RetrievalQA.from_chain_type(llm=chat_model_rag, retriever=retriever)

def search_and_generate(user_input):
    """Handles search queries with real-time Google search."""
    client = genai.Client(api_key=GOOGLE_API_KEY)
    tools = [types.Tool(google_search=types.GoogleSearch())]
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_input)])]

    generate_content_config = types.GenerateContentConfig(
        temperature=1, top_p=0.95, top_k=40, max_output_tokens=8192,
        tools=tools, response_mime_type="text/plain",
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash", contents=contents, config=generate_content_config
    ):
        response_text += chunk.text

    return response_text

# Start the chat loop
print("RAG-Enabled Agent Chat! (type 'exit' to stop)")
chat_history = []
selector_prompt = SystemMessage(
    content="""
    You are a selection agent. Based on the user query, choose an agent to handle it:
    - Reply '1' if the query is related to seven habits of highly effective people.
    - Reply '2' if the query requires real-time web search.
    """
)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Decide which agent to use
    messages = [selector_prompt] + chat_history + [HumanMessage(content=user_input)]
    selection_response = chat_model_selector(messages)
    agent_choice = selection_response.content.strip()

    if agent_choice == "1":
        reply = qa_chain.run(user_input)
    elif agent_choice == "2":
        reply = search_and_generate(user_input)
    else:
        reply = "I'm not sure how to process this request."

    print(f"Gemini: {reply}")
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(HumanMessage(content=reply))