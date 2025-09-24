import os
import stat
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import json
import shutil

# Load environment variables
load_dotenv()
CHROMA_DB_PATH = os.getenv("CHROMADB_DIR", "./chromadb")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# Ensure ChromaDB path is writable
def get_writable_chromadb_path():
    """Get a writable path for ChromaDB."""
    # Try the default path first
    if os.access(os.path.dirname(os.path.abspath(CHROMA_DB_PATH)), os.W_OK):
        return CHROMA_DB_PATH

    # If not writable, use temp directory
    temp_dir = tempfile.gettempdir()
    temp_chroma_path = os.path.join(temp_dir, "chromadb_rag")
    print(f"Using temporary directory for ChromaDB: {temp_chroma_path}")
    return temp_chroma_path


CHROMA_DB_PATH = get_writable_chromadb_path()

# Initialize Ollama models with Gemma 3
chat_model_selector = OllamaLLM(
    model="gemma3:4b",
    base_url=OLLAMA_BASE_URL,
    temperature=0.3
)

chat_model_rag = OllamaLLM(
    model="gemma3:4b",
    base_url=OLLAMA_BASE_URL,
    temperature=0.7
)

# Initialize Ollama embeddings with fixed configuration
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)


def reset_chromadb():
    """Reset ChromaDB if corrupted."""
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Resetting ChromaDB at: {CHROMA_DB_PATH}")
        try:
            # Try to change permissions recursively before deletion
            if os.path.isdir(CHROMA_DB_PATH):
                for root, dirs, files in os.walk(CHROMA_DB_PATH):
                    for d in dirs:
                        try:
                            os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                        except:
                            pass
                    for f in files:
                        try:
                            os.chmod(os.path.join(root, f), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                        except:
                            pass

            shutil.rmtree(CHROMA_DB_PATH)
            print("ChromaDB reset successfully")
            return True
        except Exception as e:
            print(f"Error resetting ChromaDB: {e}")
            return False
    return False


def setup_rag_system(force_reset=False):
    """Initialize the RAG system with PDF documents."""
    global CHROMA_DB_PATH
    print("Loading and processing documents...")
    print(f"ChromaDB path: {CHROMA_DB_PATH}")

    if force_reset:
        reset_chromadb()

    # Check if directory is writable
    parent_dir = os.path.dirname(os.path.abspath(CHROMA_DB_PATH))
    if not os.access(parent_dir, os.W_OK):
        print(f"Directory {parent_dir} is not writable, using temp directory")
        CHROMA_DB_PATH = get_writable_chromadb_path()

    # Always create a fresh ChromaDB to avoid compatibility issues
    db = None
    if os.path.exists(CHROMA_DB_PATH) and not force_reset:
        try:
            print("Loading existing ChromaDB...")
            db = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings
            )
            # Test the database
            test_query = db.similarity_search("test", k=1)
            print("ChromaDB loaded successfully")
        except Exception as e:
            print(f"ChromaDB error: {e}")
            print("Recreating ChromaDB...")
            reset_chromadb()
            db = None

    if db is None:
        print("Creating new ChromaDB...")
        try:
            # Ensure the parent directory exists and is writable
            os.makedirs(os.path.dirname(CHROMA_DB_PATH), exist_ok=True)

            # Load and process documents
            loader = PyPDFLoader("the_seven_habits.pdf")
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from PDF")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            docs = text_splitter.split_documents(documents)
            print(f"Split into {len(docs)} chunks")

            # Create ChromaDB with explicit collection name
            db = Chroma.from_documents(
                docs,
                embeddings,
                persist_directory=CHROMA_DB_PATH,
                collection_name="seven_habits"
            )
            print("ChromaDB created successfully")
        except Exception as e:
            print(f"Error creating ChromaDB: {e}")
            # Try with in-memory database as fallback
            print("Attempting in-memory database as fallback...")
            try:
                loader = PyPDFLoader("the_seven_habits.pdf")
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=100,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                docs = text_splitter.split_documents(documents)

                # Create in-memory ChromaDB (no persistence)
                db = Chroma.from_documents(
                    docs,
                    embeddings,
                    collection_name="seven_habits"
                )
                print("In-memory ChromaDB created successfully (data will not persist)")
            except Exception as e2:
                print(f"Error creating in-memory ChromaDB: {e2}")
                raise e2

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Enhanced prompt template for Seven Habits queries
    custom_prompt = PromptTemplate(
        template="""You are a knowledgeable assistant specialized in "The Seven Habits of Highly Effective People" by Stephen Covey.

Context from the book:
{context}

Question: {question}

Instructions:
- Use the provided context to answer the question accurately
- If the context doesn't contain enough information, say so clearly
- Provide specific examples or quotes from the book when relevant
- Keep your answer focused and practical

Answer: """,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model_rag,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )

    print("RAG system initialized successfully!")
    return qa_chain


def enhanced_web_search(query, num_results=5):
    """Enhanced web search using multiple sources."""
    try:
        search_results = []

        # DuckDuckGo Instant Answer API
        ddg_url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }

        response = requests.get(ddg_url, params=params, timeout=10)
        data = response.json()

        # Extract relevant information from DuckDuckGo
        if data.get("Abstract"):
            search_results.append(f"Summary: {data['Abstract']}")

        if data.get("Answer"):
            search_results.append(f"Direct Answer: {data['Answer']}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                search_results.append(f"Related: {topic['Text']}")

        # Add source information if available
        if data.get("AbstractURL"):
            search_results.append(f"Source: {data['AbstractURL']}")

        return "\n\n".join(search_results) if search_results else "No specific information found."

    except Exception as e:
        return f"Search error: {str(e)}"


def generate_web_response(user_input, search_results):
    """Generate response based on web search results using Gemma 3."""
    prompt = f"""Based on the following search results, provide a comprehensive and accurate answer to the user's question.

Search Results:
{search_results}

User Question: {user_input}

Instructions:
- Synthesize information from the search results
- Provide a clear, well-structured answer
- If the search results are insufficient, acknowledge this
- Be factual and avoid speculation
- Cite sources when possible

Response:"""

    try:
        response = chat_model_rag.invoke(prompt)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"


def select_agent(user_input, chat_history):
    """Intelligent agent selection using Gemma 3."""
    recent_context = ""
    if chat_history:
        recent_context = f"\nRecent conversation context: {chat_history[-2:]}"

    selector_prompt = f"""You are an intelligent routing system. Analyze the user query and respond with ONLY a number:

Reply '1' if the query is about:
- The Seven Habits of Highly Effective People
- Stephen Covey's teachings
- Personal effectiveness principles
- Productivity and self-improvement concepts from the book

Reply '2' if the query requires:
- Current information, news, or recent events
- General knowledge not related to the Seven Habits book
- Technical information or how-to questions
- Any topic not covered in the book

User query: {user_input}{recent_context}

Response (only number 1 or 2):"""

    try:
        response = chat_model_selector.invoke(selector_prompt)
        choice = response.strip().lower()

        # Extract number from response
        if '1' in choice and '2' not in choice:
            return "1"
        elif '2' in choice and '1' not in choice:
            return "2"
        else:
            # Default logic for ambiguous cases
            seven_habits_keywords = [
                'seven habits', 'stephen covey', 'proactive', 'synergy',
                'effectiveness', 'paradigm', 'habit', 'principle',
                'interdependence', 'win-win', 'first things first'
            ]
            query_lower = user_input.lower()
            if any(keyword in query_lower for keyword in seven_habits_keywords):
                return "1"
            else:
                return "2"

    except Exception as e:
        print(f"Selection error: {e}")
        return "1"  # Default to RAG


def main():
    """Main chat loop with enhanced features."""
    print("RAG-Enabled Agent with Gemma 3 Models")
    print("=" * 50)
    print("Required Ollama models:")
    print("- ollama pull gemma3:4b")
    print("- ollama pull gemma3:12b")
    print("- ollama pull nomic-embed-text")
    print("=" * 50)

    # Check if PDF exists
    if not os.path.exists("the_seven_habits.pdf"):
        print("Warning: 'the_seven_habits.pdf' not found.")
        print("Please place the PDF file in the same directory as this script.")
        print("RAG functionality will not work without the PDF.")
        return

    # Initialize RAG system
    retry_count = 0
    max_retries = 2
    qa_chain = None

    while retry_count < max_retries and qa_chain is None:
        try:
            force_reset = retry_count > 0
            qa_chain = setup_rag_system(force_reset=force_reset)
        except Exception as e:
            retry_count += 1
            print(f"Error setting up RAG system (attempt {retry_count}): {e}")
            if retry_count < max_retries:
                print("Retrying with fresh ChromaDB...")
            else:
                print("\nTroubleshooting tips:")
                print("1. Make sure Ollama is running: 'ollama serve'")
                print("2. Check if models are installed: 'ollama list'")
                print("3. Verify PDF file exists in current directory")
                print("4. Try deleting the './chromadb' folder and restart")
                return

    print("\nRAG-Enabled Agent Chat Ready!")
    print("Tip: Ask about The Seven Habits or any general knowledge questions")
    print("Type 'exit' to stop, 'help' for commands, 'reset' to reset ChromaDB")
    print(f"ChromaDB location: {CHROMA_DB_PATH}\n")

    chat_history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            elif user_input.lower() == "help":
                print("\nAvailable commands:")
                print("- Ask about The Seven Habits of Highly Effective People")
                print("- Ask general knowledge questions")
                print("- Type 'exit' to quit")
                print("- Type 'clear' to clear chat history")
                print("- Type 'reset' to reset ChromaDB")
                continue
            elif user_input.lower() == "clear":
                chat_history = []
                print("Chat history cleared!")
                continue
            elif user_input.lower() == "reset":
                try:
                    reset_chromadb()
                    qa_chain = setup_rag_system(force_reset=True)
                    print("ChromaDB reset successfully!")
                except Exception as e:
                    print(f"Error resetting ChromaDB: {e}")
                continue
            elif not user_input:
                continue

            # Decide which agent to use
            agent_choice = select_agent(user_input, chat_history)
            agent_name = "Seven Habits RAG" if agent_choice == "1" else "Web Search"
            print(f"[Using Agent {agent_choice}: {agent_name}]")

            if agent_choice == "1":
                # Use RAG for Seven Habits queries
                result = qa_chain.invoke({"query": user_input})
                reply = result.get("result", "No answer found.")

                # Show source documents if available
                if result.get("source_documents"):
                    print(f"\nSources: {len(result['source_documents'])} relevant passages found")

            elif agent_choice == "2":
                # Use web search for general queries
                print("Searching the web...")
                search_results = enhanced_web_search(user_input)
                reply = generate_web_response(user_input, search_results)

            else:
                reply = "I'm not sure how to process this request."

            print(f"\nGemma3: {reply}\n")

            # Update chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": reply})

            # Keep history manageable (last 10 exchanges)
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing request: {e}")


if __name__ == "__main__":
    main()