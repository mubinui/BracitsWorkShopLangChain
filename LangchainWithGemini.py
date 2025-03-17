import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

# Set your Google Gemini API key
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE "

# Initialize the Gemini chat model with full controls
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,  # Creativity level (0 = deterministic, 1 = highly random)
    # max_output_tokens=500,  # Maximum response length
    # top_p=0.9,  # Nucleus sampling (alternative to temperature)
    # top_k=40,  # Limits the set of words considered at each step
)

# Start the chat loop
print("Chat with Gemini! (type 'exit' to stop)")
chat_history = []
system_message = SystemMessage(
    content=(
        "You are A Python developer .\n"
        "1. You will write Codes only.\n"
        "2. You will return Only code Snippets.\n"
    )
)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Construct messages including chat history
    messages = [system_message] + chat_history + [HumanMessage(content=f" *USER BUSINESS * {user_input}.")]

    # Generate response from Gemini
    response = chat_model(messages)
    reply = response.content

    # Print response
    print(f"Gemini: {reply}")

    # Store conversation history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(response)
