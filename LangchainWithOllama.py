import os
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize Ollama with all available parameters
chat_model = Ollama(
    model="gemma3:4b",  # Change model as needed
    temperature=0.7,  # Controls randomness (0 = deterministic, 1 = creative)
    num_ctx=2048,  # Context window size
    num_gpu=1,  # Use GPU if available
    top_p=0.9,  # Controls nucleus sampling
    top_k=50,  # Controls sampling diversity
    repeat_penalty=1.2,  # Reduces repetition
)

# Start chat loop
print("I am Shakespeare. Give me a topic! (type 'exit' to stop)")
chat_history = []

# System message
system_message = SystemMessage(
    content=(
        "You are William Shakespeare.\n"
        "1. You will write poems in the style of Shakespeare.\n"
        "2. You will use poetic language and old English.\n"
    )
)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Construct messages with history
    messages = [system_message] + chat_history + [HumanMessage(content=f"Write a poem about {user_input}.")]

    # Generate response from Ollama
    response = chat_model.invoke(messages)
    reply = response.strip()

    # Print response
    print(f"Ollama: {reply}")

    # Store conversation history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(SystemMessage(content=reply))
