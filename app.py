import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, exceptions
from pathlib import Path
from dotenv import load_dotenv
import uuid
import random
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END

# Load environment variables
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

app = Flask(__name__)
CORS(app)

# Azure OpenAI Configuration
azure_client = AzureOpenAI(
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2023-05-15"
)

# Azure Cosmos DB Configuration
COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
COSMOS_DB_CONTAINER = os.getenv("COSMOS_DB_CONTAINER")

if not all([COSMOS_DB_ENDPOINT, COSMOS_DB_KEY, COSMOS_DB_NAME, COSMOS_DB_CONTAINER]):
    raise ValueError("Missing required Cosmos DB environment variables")

cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
database = cosmos_client.get_database_client(COSMOS_DB_NAME)
container = database.get_container_client(COSMOS_DB_CONTAINER)

# Define State Schema
class AgentState(TypedDict):
    messages: List[dict]
    user_id: str
    next_agent: Optional[str]

# Define Tools for Agents
def store_user_response(user_id: str, response: str, question_index: int) -> str:
    """Store user response in Cosmos DB."""
    query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
    user_docs = list(container.query_items(query, enable_cross_partition_query=True))
    if not user_docs:
        user_doc = {"id": str(uuid.uuid4()), "user_id": user_id, "question_index": 0, "answers": []}
        container.create_item(user_doc)
    else:
        user_doc = user_docs[0]
    
    user_doc["answers"] = user_doc.get("answers", []) + [response]
    user_doc["question_index"] = question_index + 1
    container.upsert_item(user_doc)
    return "Response stored."

def get_user_data(user_id: str) -> dict:
    """Retrieve user data from Cosmos DB."""
    query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
    user_docs = list(container.query_items(query, enable_cross_partition_query=True))
    return user_docs[0] if user_docs else {}

# Helper Function for Azure OpenAI
def invoke_azure_openai(messages, system_prompt):
    """Invoke Azure OpenAI API."""
    response = azure_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Define Agents
def chat_agent(state: AgentState) -> AgentState:
    user_message = state["messages"][-1]["content"]
    user_id = state["user_id"]
    system_prompt = (
        "You are the Chat Agent, the primary interaction point and supervisor for users. "
        "Provide an empathetic and engaging conversation. Begin with a warm, personalized welcome. "
        "Check user history with get_user_data to identify new or returning users. "
        "For new users, introduce SoulSync AI’s capabilities. For returning users, acknowledge past interactions. "
        "Classify user intent into: 'wellness' (mood/mental state), 'therapy' (therapy plans), 'rehab' (post-rehab support), or 'general' (other queries). "
        "If intent is 'wellness', set next_agent to 'wellness_check_agent'. "
        "If intent is 'therapy', set next_agent to 'personalized_therapy_agent'. "
        "If intent is 'rehab', set next_agent to 'post_rehab_follow_up_agent'. "
        "If intent is 'general', respond directly with an empathetic message and set next_agent to None."
    )
    user_data = get_user_data(user_id)
    if not user_data:
        welcome = "Hello! I'm SoulSync AI, here to help you. What’s on your mind?"
    else:
        welcome = "Welcome back! How can I assist you today?"
    
    messages = [{"role": "user", "content": f"{welcome}\n{user_message}"}]
    response = invoke_azure_openai(messages, system_prompt)
    
    # Determine intent and next agent
    intent_map = {
        "wellness": "wellness_check_agent",
        "therapy": "personalized_therapy_agent",
        "rehab": "post_rehab_follow_up_agent",
        "general": None
    }
    intent = "general"
    if "wellness" in response.lower() or "mood" in user_message.lower() or "mental" in user_message.lower():
        intent = "wellness"
    elif "therapy" in response.lower() or "plan" in user_message.lower():
        intent = "therapy"
    elif "rehab" in response.lower() or "relapse" in user_message.lower():
        intent = "rehab"
    
    next_agent = intent_map[intent]
    if next_agent:
        response = f"{response}\nRouting to {next_agent.replace('_', ' ').title()}"

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response, "agent_name": "chat_agent"}],
        "user_id": state["user_id"],
        "next_agent": next_agent
    }

def wellness_agent(state: AgentState) -> AgentState:
    user_message = state["messages"][-1]["content"]
    user_id = state["user_id"]
    system_prompt = (
        "You are the Wellness Check Agent. Check user sentiment and mental state. "
        "Ask 5 psychometric questions one by one (use get_user_data to track progress). "
        "Select randomly from this pool: 'On a scale of 1-10, how would you rate your mood today?', "
        "'How often do you feel anxious or stressed in a week?', 'Do you find it difficult to stay motivated?', "
        "'How well are you sleeping at night? (1-10)', 'Are you feeling socially connected or isolated lately?' "
        "Store answers with store_user_response. After 5 answers, analyze them: if average < 4, suggest contacting a human; else, end with a motivational message."
    )
    user_data = get_user_data(user_id)
    questions = [
        "On a scale of 1-10, how would you rate your mood today?",
        "How often do you feel anxious or stressed in a week?",
        "Do you find it difficult to stay motivated throughout the day?",
        "How well are you sleeping at night? (1-10)",
        "Are you feeling socially connected or isolated lately?"
    ]
    
    question_index = user_data.get("question_index", 0)
    answers = user_data.get("answers", [])
    
    if question_index > 0 and question_index <= 5:
        store_user_response(user_id, user_message, question_index - 1)
        answers.append(user_message)
    
    if len(answers) >= 5:
        avg_score = sum(int(a) if a.isdigit() else 5 for a in answers) / len(answers)
        response = (
            "Your responses suggest a high risk. Please contact a professional immediately." if avg_score < 4
            else "Thank you for your responses. Stay positive and keep taking care of yourself!"
        )
        container.upsert_item({"id": user_data["id"], "user_id": user_id, "question_index": 0, "answers": []})
    elif question_index < 5:
        response = random.choice(questions)
    else:
        response = questions[0]
    
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response, "agent_name": "wellness_check_agent"}],
        "user_id": state["user_id"],
        "next_agent": None
    }

def therapy_agent(state: AgentState) -> AgentState:
    user_message = state["messages"][-1]["content"]
    user_id = state["user_id"]
    system_prompt = (
        "You are the Personalized Therapy Agent. Create therapy plans based on patient history (use get_user_data). "
        "Suggest a therapy schedule or activity plan and provide motivational support."
    )
    user_data = get_user_data(user_id)
    messages = [{"role": "user", "content": f"User history: {user_data}\n{user_message}"}]
    response = invoke_azure_openai(messages, system_prompt)
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response, "agent_name": "personalized_therapy_agent"}],
        "user_id": state["user_id"],
        "next_agent": None
    }

def rehab_agent(state: AgentState) -> AgentState:
    user_message = state["messages"][-1]["content"]
    user_id = state["user_id"]
    system_prompt = (
        "You are the Post-Rehab Follow-Up Agent. Engage users after rehab to prevent relapse. "
        "Track progress (use get_user_data), detect early signs of relapse, and send personalized motivational messages."
    )
    user_data = get_user_data(user_id)
    messages = [{"role": "user", "content": f"User history: {user_data}\n{user_message}"}]
    response = invoke_azure_openai(messages, system_prompt)
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response, "agent_name": "post_rehab_follow_up_agent"}],
        "user_id": state["user_id"],
        "next_agent": None
    }

# Create Workflow with Chat Agent as Supervisor
workflow = StateGraph(AgentState)
workflow.add_node("chat_agent", chat_agent)
workflow.add_node("wellness_check_agent", wellness_agent)
workflow.add_node("personalized_therapy_agent", therapy_agent)
workflow.add_node("post_rehab_follow_up_agent", rehab_agent)

workflow.set_entry_point("chat_agent")
workflow.add_conditional_edges("chat_agent", lambda state: state["next_agent"], {
    "wellness_check_agent": "wellness_check_agent",
    "personalized_therapy_agent": "personalized_therapy_agent",
    "post_rehab_follow_up_agent": "post_rehab_follow_up_agent",
    None: END
})
workflow.add_edge("wellness_check_agent", END)
workflow.add_edge("personalized_therapy_agent", END)
workflow.add_edge("post_rehab_follow_up_agent", END)

app_workflow = workflow.compile()

# Flask Endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id")
    user_message = data.get("message", "").strip()

    if not user_id or not user_message:
        return jsonify({"error": "Missing user_id or message"}), 400

    try:
        # Invoke the workflow with chat_agent as supervisor
        result = app_workflow.invoke({
            "messages": [{"role": "user", "content": user_message}],
            "user_id": user_id,
            "next_agent": None
        })

        # Extract the final response and agent used
        final_message = result["messages"][-1]["content"]
        agent_used = result["messages"][-1].get("agent_name", "chat_agent")

        return jsonify({"response": final_message, "agent_used": agent_used})

    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)