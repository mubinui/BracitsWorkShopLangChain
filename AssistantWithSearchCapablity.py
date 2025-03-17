import os
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

# Set your Google Gemini API key
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"

# Initialize the Gemini chat model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
)


def search_and_generate(user_input):
    """Handles search queries with real-time Google search."""
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    tools = [types.Tool(google_search=types.GoogleSearch())]
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        tools=tools,
        response_mime_type="text/plain",
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash", contents=contents, config=generate_content_config
    ):
        response_text += chunk.text

    return response_text


# Start the chat loop
print("Chat with Gemini! (type 'exit' to stop)")
chat_history = []
system_message = SystemMessage(
    content="""
    You are a Python developer.
    1. You will write codes only when required.
    2. You will return only code snippets when asked for code.
    """
)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    if "search" in user_input.lower():
        reply = search_and_generate(user_input.replace("search", ""))
    else:
        messages = [system_message] + chat_history + [HumanMessage(content=user_input)]
        response = chat_model(messages)
        reply = response.content

    print(f"Gemini: {reply}")
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(HumanMessage(content=reply))
