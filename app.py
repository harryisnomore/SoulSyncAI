import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, exceptions
from pathlib import Path
from dotenv import load_dotenv
import uuid

# Load environment variables (unchanged)
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

app = Flask(__name__)
CORS(app)

# Azure OpenAI Configuration (unchanged)
client = AzureOpenAI(
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2023-05-15"
)

# Azure Cosmos DB Configuration (unchanged)
COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
COSMOS_DB_CONTAINER = os.getenv("COSMOS_DB_CONTAINER")

if not all([COSMOS_DB_ENDPOINT, COSMOS_DB_KEY, COSMOS_DB_NAME, COSMOS_DB_CONTAINER]):
    raise ValueError("Missing required Cosmos DB environment variables")

cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
database = cosmos_client.get_database_client(COSMOS_DB_NAME)
container = database.get_container_client(COSMOS_DB_CONTAINER)

# Signup endpoint (unchanged)
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    first_name = data.get("firstName")
    last_name = data.get("lastName")
    dob = data.get("dob")
    email = data.get("email")
    password = data.get("password")  # In production, hash this!

    if not all([first_name, last_name, dob, email, password]):
        return jsonify({"error": "Missing required fields"}), 400

    user_id = str(uuid.uuid4())
    user_data = {
        "id": user_id,
        "user_id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "dob": dob,
        "email": email,
        "password": password,  # Store hashed password in production
        "conversation_history": []
    }

    try:
        container.upsert_item(user_data)
        return jsonify({"user_id": user_id, "message": "Signup successful"}), 201
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500

# Login endpoint (unchanged)
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400

    try:
        query = f"SELECT * FROM c WHERE c.email = '{email}'"
        user_docs = list(container.query_items(query, enable_cross_partition_query=True))

        if not user_docs:
            return jsonify({"error": "Email not found"}), 404

        user_doc = user_docs[0]
        if user_doc["password"] != password:  # In production, compare hashed passwords
            return jsonify({"error": "Incorrect password"}), 401

        return jsonify({"user_id": user_doc["user_id"], "message": "Login successful"}), 200

    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

# RESTORED: Generate summary function
def generate_summary(conversation_history):
    """Generate a brief summary of the conversation."""
    if not conversation_history:
        return "No previous conversation found."
    
    summary_prompt = [
        {"role": "system", "content": "Provide a concise 3-5 line summary of the following conversation."}
    ]
    summary_prompt.extend(conversation_history[-10:])  # Limit to last 10 messages

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=summary_prompt
    )
    return response.choices[0].message.content.strip()

# RESTORED: Detect intent function
def detect_intent(user_message, has_previous):
    """Use LLM to detect user intent conversationally with improved handling of negations."""
    intent_prompt = [
        {"role": "system", "content": """Classify the user's intent based on their message. Respond with one of these categories:
        - 'recall_past': User wants to know about or reference the previous conversation.
        - 'continue': User wants to continue the previous conversation.
        - 'start_fresh': User wants to start a new conversation.
        - 'general': User is making a general statement or question with no clear intent.
        Consider negations (e.g., 'no' followed by a request) as part of the intent. Provide only the category name."""},
        {"role": "user", "content": user_message}
    ]
    
    if has_previous:
        intent_prompt.append({"role": "system", "content": "Note: There is a previous conversation available."})
    else:
        intent_prompt.append({"role": "system", "content": "Note: There is no previous conversation available."})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=intent_prompt,
        temperature=0.3
    )
    intent = response.choices[0].message.content.strip()

    if "no" in user_message.lower() and intent not in ["recall_past", "start_fresh"]:
        fallback_prompt = intent_prompt + [
            {"role": "system", "content": "Re-evaluate the intent, considering 'no' as a potential indicator to reverse the previous action or recall past context. Respond with the category name."}
        ]
        fallback_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=fallback_prompt,
            temperature=0.3
        )
        intent = fallback_response.choices[0].message.content.strip()

    return intent

# RESTORED: Get chat response function
def get_chat_response(user_message, conversation_history, user_id, intent):
    """Generate a response with a system prompt based on detected intent."""
    if intent == "recall_past":
        system_prompt = [
            {"role": "system", "content": f"You are SoulSync, a chill AI assistant from the future. Summarize the user's past conversation if available, then assist. The user's ID is {user_id}. Use their chat history to recall past vibes. If no relevant history, focus on the new request."}
        ]
    elif intent == "continue":
        system_prompt = [
            {"role": "system", "content": f"You are SoulSync, a chill AI assistant from the future. Continue the user's previous conversation naturally. The user's ID is {user_id}. Use their chat history to pick up where we left off."}
        ]
    elif intent == "start_fresh":
        system_prompt = [
            {"role": "system", "content": f"You are SoulSync, a chill AI assistant from the future. Start a new conversation. The user's ID is {user_id}. Ignore past chat history."}
        ]
    else:  # general
        system_prompt = [
            {"role": "system", "content": f"You are SoulSync, a chill AI assistant from the future. Respond naturally to the user's new request. The user's ID is {user_id}. Only use chat history if they ask about it explicitly (e.g., 'what did we talk about?', 'remember last chat'). Avoid summarizing past conversations unless asked."}
        ]

    messages = system_prompt
    if intent in ["recall_past", "continue"] and conversation_history:
        messages.extend(conversation_history[-10:])
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    return response.choices[0].message.content.strip()

# CHANGED: Updated chat endpoint to use intelligent responses
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id")
    user_message = data.get("message", "").strip()

    if not user_id or not user_message:
        return jsonify({"error": "Missing user_id or message"}), 400

    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
        user_docs = list(container.query_items(query, enable_cross_partition_query=True))

        if not user_docs:
            return jsonify({"error": "User not found"}), 404

        user_doc = user_docs[0]
        conversation_history = user_doc.get("conversation_history", [])
        has_previous = bool(conversation_history)

        # Use intent detection
        intent = detect_intent(user_message, has_previous)

        # Generate intelligent response
        ai_response = get_chat_response(user_message, conversation_history, user_id, intent)

        # Append new messages to conversation history
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": ai_response})

        # Update user document with summary if recalling past
        user_doc["conversation_history"] = conversation_history
        user_doc["summary"] = generate_summary(conversation_history) if intent == "recall_past" else user_doc.get("summary", "No summary needed")
        container.upsert_item(user_doc)

        return jsonify({"response": ai_response})

    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route("/")
def home():
    return "SoulSync API is running!"

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)