import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, exceptions
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

app = Flask(__name__)
CORS(app)

# Azure OpenAI configuration
client = AzureOpenAI(
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2023-05-15"
)

# Azure Cosmos DB configuration
COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
COSMOS_DB_CONTAINER = os.getenv("COSMOS_DB_CONTAINER")

# Validate environment variables
if not all([COSMOS_DB_ENDPOINT, COSMOS_DB_KEY, COSMOS_DB_NAME, COSMOS_DB_CONTAINER]):
    raise ValueError("Missing required Cosmos DB environment variables")

# Initialize Cosmos DB client
cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
database = cosmos_client.get_database_client(COSMOS_DB_NAME)
container = database.get_container_client(COSMOS_DB_CONTAINER)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_message}]
        )
        ai_response = response.choices[0].message.content
        chat_data = {
            "id": str(hash(user_message)),
            "user_message": user_message,
            "ai_response": ai_response
        }
        container.create_item(body=chat_data)
        return jsonify({"response": ai_response})
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"response": f"CosmosDB Error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

@app.route("/history", methods=["GET"])
def get_chat_history():
    try:
        chat_history = list(container.query_items(
            query="SELECT c.id, c.user_message, c.ai_response FROM c",
            enable_cross_partition_query=True
        ))
        return jsonify(chat_history)
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500

@app.route("/")
def home():
    return "Azure OpenAI + Cosmos DB Chat API is running!"

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)