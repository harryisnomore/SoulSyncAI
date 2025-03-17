import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Azure Open AI configuration
dotenv_path = Path(__file__).resolve().parent
load_dotenv(dotenv_path=dotenv_path)
client = AzureOpenAI(
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2023-05-15"
)

@app.route("/chat", methods=["POST"])
def chat():
    # Get user message from request
    user_message = request.json.get("message", "")

    try:
        # Call Azure Open AI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Replace with your deployed model name
            messages=[{"role": "user", "content": user_message}]
        )

        # Extract AI response
        ai_response = response.choices[0].message.content
        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

@app.route("/")
def home():
    return "Azure Open AI Chat API is running!"

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)