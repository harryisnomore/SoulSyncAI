# Description: This file is used to test the functionality of the agents and the supervisor agent.
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, exceptions
from datetime import datetime, timezone
import uuid
import logging
import smtplib
from email.mime.text import MIMEText
import json

# Setup logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

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

# Initialize Azure OpenAI Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2023-05-15",
    deployment_name="gpt-4o-mini"
)

# Core Functions with explicit docstrings

def get_summary(user_input: str) -> str:
    """Generate a brief summary of the user's input.
    
    Args:
        user_input (str): The user's description of their experience.
    
    Returns:
        str: A summary string prefixed with 'User is experiencing:'.
    """
    logger.debug(f"[WELLNESS_CHECK_AGENT] Summarizing user's concern: {user_input}")
    return f"User is experiencing: {user_input}"

def analyze_sentiment(summary: str) -> str:
    """Perform basic sentiment analysis and return an empathetic response.
    
    Args:
        summary (str): The summary of the user's input.
    
    Returns:
        str: An empathetic response based on detected sentiment.
    """
    logger.debug("[WELLNESS_CHECK_AGENT] Analyzing sentiment...")
    if "hopeless" in summary.lower() or "overwhelmed" in summary.lower():
        return "I'm really sorry you're feeling this way. You're not alone, and support is available."
    elif "stressed" in summary.lower() or "anxious" in summary.lower():
        return "I understand that stress can be overwhelming. Taking small steps can really help."
    return "It's okay to have tough days. Remember, you're stronger than you think!"

def generate_questions(summary: str) -> list:
    """Generate context-based questions based on the user's summary.
    
    Args:
        summary (str): The summary of the user's input.
    
    Returns:
        list: A list of five questions tailored to the summary.
    """
    logger.debug("[WELLNESS_CHECK_AGENT] Generating dynamic questions...")
    return [
        f"Can you describe more about how {summary.split(':')[-1].strip()} affects your daily life?",
        "Do you experience mood swings frequently? (Yes/No)",
        "Have you had trouble sleeping lately? (Yes/No)",
        "Do you feel disconnected from loved ones? (Yes/No)",
        "On a scale of 1-10, how would you rate your emotional well-being?"
    ]

def wellness_check(patient_mood: str, additional_responses: list = None) -> str:
    """Assess the patient's mental health based on their mood description.
    
    Args:
        patient_mood (str): The user's description of their mood.
        additional_responses (list, optional): Responses to follow-up questions. Defaults to None.
    
    Returns:
        str: A risk assessment or questions if more info is needed.
    """
    logger.info(f"[WELLNESS_CHECK_AGENT] Assessing mood: {patient_mood}")
    summary = get_summary(patient_mood)
    sentiment_response = analyze_sentiment(summary)
    
    if additional_responses is None or len(additional_responses) < 5:
        questions = generate_questions(summary)
        return f"{sentiment_response} Would you be comfortable answering these questions? {questions}"
    
    responses = [r.lower() for r in additional_responses]
    if "yes" in responses or any(int(x) >= 7 for x in responses if x.isdigit()):
        return "High risk detected. Our therapist will contact you soon."
    elif any(int(x) >= 4 for x in responses if x.isdigit()):
        return "Medium risk detected. Remember, tough times don't last. You are stronger than you think!"
    else:
        return "Low risk detected. Keep going! Every storm runs out of rain."

def provide_therapy(condition: str, user_id: str = None, chat_history: list = None) -> dict:
    """
    You are the Therapy Planning specialist of SoulSync AI.
    Focus on:
    1. Creating personalized therapy plans tailored to the patient's unique condition and history.
    2. Providing actionable recommendations that address specific triggers and needs.
    3. Setting realistic goals for short-term relief and long-term growth.
    4. Maintaining supportive communication to foster trust and progress.
    """
    logger.info(f"[THERAPY_AGENT] Recommending therapy for: {condition}")

    # Prepare chat summary from history
    chat_summary = "No chat history provided."
    if chat_history:
        user_msgs = [msg["content"] for msg in chat_history if msg["role"] == "user"]
        chat_summary = "Conversation Summary:\n" + "\n".join([f"Patient: {msg}" for msg in user_msgs[-4:]])

    # System prompt for dynamic therapy plan generation
    system_prompt = """
    You are the Therapy Planning specialist of SoulSync AI. Your task is to create a highly personalized therapy plan based solely on the user's current condition and conversation history, without relying on predefined therapy types or hardcoded recommendations. Analyze the input for:
    - Emotional states (e.g., anxiety, sadness, anger, stress, frustration).
    - Triggers or contexts (e.g., work, family, relationships, health).
    - Severity (e.g., mild, moderate, severe based on language like "overwhelming," "sometimes," "constant").
    
    If the input is vague (fewer than 3 words or no clear emotion/trigger), respond with a prompt asking for more details instead of generating a plan.

    Generate a therapy plan with the following structure:"""

    # Check if input is too vague
condition_lower = condition.lower()
if len(condition_lower.split()) < 3 and not any(word in condition_lower for word in ["anxiety", "sad", "anger", "stress", "work", "family"]):
    user_response = (
        "I’d love to craft a therapy plan that’s perfect for you. Could you share more about what’s been on your mind lately "
        "or how you’re feeling? That’ll help me make it truly personal!"
    )
    therapy_plan_text = f"Pending more patient input for personalization.\nCondition: {condition}\n{chat_summary}"
    return {
        "user_response": user_response,
        "therapy_plan": therapy_plan_text,
        "agent_name": "therapy_expert"
    }

# Prepare messages for Azure OpenAI
messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": (
            f"Current Condition: {condition}\n"
            f"{chat_summary}\n\n"
            "Generate a personalized therapy plan following the structure provided in the system prompt."
        )
    }
]

# Invoke Azure OpenAI to generate the therapy plan
try:
    therapy_plan_text = model.invoke(messages).content
    logger.debug(f"Generated therapy plan for user_id={user_id}: {therapy_plan_text}")
except Exception as e:
    logger.error(f"Failed to generate therapy plan for user_id={user_id}: {str(e)}")
    therapy_plan_text = (
        f"Subject: Therapy Recommendation for Patient {user_id or 'Unknown'}\n\n"
        f"Therapy Recommendation:\n"
        f"An error occurred while generating your plan. Please try again or provide more details.\n\n"
        f"{chat_summary}"
    )
    user_response = "I ran into a snag while creating your plan. Could you tell me a bit more so I can try again?"
    return {
        "user_response": user_response,
        "therapy_plan": therapy_plan_text,
        "agent_name": "therapy_expert"
    }

# Extract user-facing response from the plan
user_response = (
    "I’ve created a personalized therapy plan just for you based on what you’ve shared. "
    "Your therapist has the details and will reach out soon to guide you!"
)

return {
    "user_response": user_response,
    "therapy_plan": therapy_plan_text,
    "agent_name": "therapy_expert"
}

def post_rehab_followup(patient_status: str) -> str:
"""Provide guidance based on the patient's post-rehab status.

Args:
    patient_status (str): The user's reported status after rehab.

Returns:
    str: Guidance based on the status.
"""
logger.info(f"[POST_REHAB_AGENT] Following up: {patient_status}")
if "improved" in patient_status.lower():
    return "Continue with self-help practices and regular check-ins."
return "Schedule a follow-up therapy session."

def send_therapy_plan_to_therapist(therapy_plan: str, user_id: str) -> bool:
"""Send the therapy plan to the therapist via email.

Args:
    therapy_plan (str): The therapy plan text to send.
    user_id (str): The user's ID for identification.

Returns:
    bool: True if email sent successfully, False otherwise.
"""
therapist_email = os.getenv("THERAPIST_EMAIL")
smtp_server = os.getenv("SMTP_SERVER")
smtp_port = int(os.getenv("SMTP_PORT"))
smtp_user = os.getenv("SMTP_USER")
smtp_password = os.getenv("SMTP_PASSWORD")

if not all([therapist_email, smtp_server, smtp_port, smtp_user, smtp_password]):
    logger.error("Missing SMTP configuration in environment variables")
    return False

subject = f"Therapy Plan for User {user_id}"
msg = MIMEText(therapy_plan)
msg["Subject"] = subject
msg["From"] = smtp_user
msg["To"] = therapist_email

try:
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
    logger.info(f"Therapy plan emailed to {therapist_email} for user_id={user_id}")
    return True
except Exception as e:
    logger.error(f"Failed to send email for user_id={user_id}: {str(e)}")
    return False

# Log agent creation
logger.info("Creating agents...")
try:
wellness_check_agent = create_react_agent(
    model=model,
    tools=[wellness_check],
    name="wellness_check_expert",
    prompt="You are an expert in mental health assessments. For any user message describing their mood or emotions, use the 'wellness_check' tool to assess their mental state. Pass the user's message as the 'patient_mood' parameter and, if provided, any additional responses as 'additional_responses'. Return the tool's output as your response."
)
logger.info("Wellness check agent created successfully")
except Exception as e:
logger.error(f"Failed to create wellness_check_agent: {str(e)}")
raise

try:
 therapy_agent = create_react_agent(
    model=model,
    tools=[provide_therapy, send_therapy_plan_to_therapist],
    name="therapy_expert",
    prompt="You are the Personalized Therapy Agent, part of SoulSync AI. When a user requests therapy or indicates a need for professional support, use the 'provide_therapy' tool with the user's message as the 'condition' parameter, their user_id, and chat history. Return the 'user_response' from the tool’s output as your response. After generating the plan, use the 'send_therapy_plan_to_therapist' tool to send the 'therapy_plan' to the therapist, including a brief chat summary."
)
logger.info("Therapy agent created successfully")
except Exception as e:
logger.error(f"Failed to create therapy_agent: {str(e)}")
raise
try:
  post_rehab_agent = create_react_agent(
    model=model,
    tools=[post_rehab_followup],
    name="post_rehab_expert",
    prompt="You help patients post-therapy. Use the 'post_rehab_followup' tool with the user's message as the 'patient_status' parameter and return the tool's output as your response."
)
logger.info("Post-rehab agent created successfully")
except Exception as e:
logger.error(f"Failed to create post_rehab_agent: {str(e)}")
raise

# Create Supervisor Agent
logger.info("Creating supervisor agent...")
try:
 chat_agent = create_supervisor(
    [wellness_check_agent, therapy_agent, post_rehab_agent],
    model=model,
    prompt=(
        "Your role is to:\n"
        "1. Understand the user's needs and emotions based on their message and conversation history.\n"
        "2. Provide an empathetic, natural response without mentioning the routing process.\n"
        "3. Classify the user's intent and route to the appropriate agent with necessary data:\n"
        "   - Intent: wellness (emotional distress phrases like 'I feel sad', 'I’m stressed') -> Route to wellness_check_agent with patient_mood=user_message and additional_responses if provided\n"
        "   - Intent: therapy (explicit requests like 'I need therapy', 'I want a therapist') -> Route to therapy_agent with condition=user_message, user_id, and chat_history\n"
        "   - Intent: rehab (post-rehab phrases like 'I relapsed', 'I’m struggling after rehab') -> Route to post_rehab_agent with patient_status=user_message\n"
        "   - Intent: general (other messages like 'hi', 'what is this') -> Respond empathetically without routing, e.g., 'Hello! I’m here to help—how are you feeling today?'\n"
        "**Instructions**:\n"
        "- Prioritize explicit keywords: 'therapy' or 'therapist' -> therapy intent; emotional distress -> wellness intent.\n"
        "- Use conversation history to understand emotional state.\n"
        "- Be strict about intent classification."
        "Do not change the response the other agents are generating, present it to the user as it is and take the conversation ahead until completed. Give user the response from wellness_check_expert as it is, do not interpret it."
    )
)
logger.info("Supervisor agent created successfully")
except Exception as e:
logger.error(f"Failed to create chat_agent: {str(e)}")
raise

# Compile the workflow
logger.info("Compiling workflow...")
app_ai = chat_agent.compile()
logger.info("Workflow compiled successfully")

# Cosmos DB Helper Functions
def get_user_data(user_id: str) -> dict:
 """Retrieve user data from Cosmos DB."""
try:
    query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
    user_docs = list(container.query_items(query, enable_cross_partition_query=True))
    return user_docs[0] if user_docs else {}
except Exception as e:
    logger.error(f"get_user_data: Error for user_id={user_id}: {str(e)}")
    raise

def store_user_data(user_id: str, messages: list, context: dict) -> None:
 """Store or update user data in Cosmos DB."""
try:
    user_data = get_user_data(user_id)
    if not user_data:
        user_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "messages": [],
            "context": {},
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    user_data["messages"] = messages
    user_data["context"] = context
    user_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    container.upsert_item(user_data)
    logger.debug(f"Stored user data for user_id={user_id}")
except Exception as e:
    logger.error(f"store_user_data: Error for user_id={user_id}: {str(e)}")
    raise

# API Routes
@app.route("/signup", methods=["POST"])
def signup():
data = request.get_json()
required_fields = ["firstName", "lastName", "dob", "email", "password"]

if not all(field in data for field in required_fields):
    logger.warning("Signup failed: Missing required fields")
    return jsonify({"error": "Missing required fields"}), 400

query = f"SELECT * FROM c WHERE c.email = '{data['email']}'"
existing_users = list(container.query_items(query, enable_cross_partition_query=True))
if existing_users:
    logger.warning(f"Signup failed: Email {data['email']} already exists")
    return jsonify({"error": "Email already exists"}), 400

user_id = str(uuid.uuid4())
user_doc = {
    "id": str(uuid.uuid4()),
    "user_id": user_id,
    **data,
    "messages": [],
    "context": {},
    "created_at": datetime.now(timezone.utc).isoformat()
}
container.create_item(user_doc)
logger.info(f"User signed up successfully: {user_id}")
return jsonify({"message": "Signup successful", "user_id": user_id}), 201

@app.route("/login", methods=["POST"])
def login():
data = request.get_json()
if not data.get("email") or not data.get("password"):
    logger.warning("Login failed: Missing email or password")
    return jsonify({"error": "Missing email or password"}), 400

query = f"SELECT * FROM c WHERE c.email = '{data['email']}'"
users = list(container.query_items(query, enable_cross_partition_query=True))

if not users or users[0]["password"] != data["password"]:
    logger.warning(f"Login failed for email {data['email']}: Invalid credentials")
    return jsonify({"error": "Invalid email or password"}), 401

logger.info(f"User logged in: {users[0]['user_id']}")
return jsonify({"message": "Login successful", "user_id": users[0]["user_id"]}), 200

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history():
user_id = request.args.get("user_id")
if not user_id:
    logger.warning("Missing user_id in get_chat_history request")
    return jsonify({"error": "Missing user_id"}), 400

try:
    logger.debug(f"Fetching chat history for user_id={user_id}")
    user_data = get_user_data(user_id)
    if not user_data:
        logger.warning(f"No user data found for user_id={user_id}")
        return jsonify({"messages": []}), 200
    messages = user_data.get("messages", [])
    logger.info(f"Chat history retrieved for user_id={user_id}: {len(messages)} messages")
    return jsonify({"messages": messages})
except Exception as e:
    logger.exception(f"Error fetching chat history for user_id={user_id}: {str(e)}")
    return jsonify({"error": "Failed to fetch chat history"}), 500

@app.route("/chat", methods=["POST"])
def chat():
data = request.get_json()
user_message = data.get("message", "").strip()
user_id = data.get("user_id", "")
additional_responses = data.get("additional_responses", [])

if not user_message or not user_id:
    logger.warning("Chat request failed: Missing user_id or message")
    return jsonify({"error": "Missing user_id or message"}), 400

try:
    user_data = get_user_data(user_id)
    if not user_data:
        logger.warning(f"No user data found for user_id={user_id}")
        return jsonify({"error": "User not found"}), 404

    messages = user_data.get("messages", [])
    context = user_data.get("context", {})
    
    messages.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    logger.debug(f"User message added for user_id={user_id}: {user_message}")

    last_user_messages = [msg["content"] for msg in messages[-3:] if msg["role"] == "user"]
    if len(last_user_messages) >= 2 and all(msg == last_user_messages[-1] for msg in last_user_messages[-2:]):
        response = "I notice you've said the same thing a few times. Can you tell me more about how you’re feeling?"
        agent_used = "chat_agent"
        logger.info(f"Repetitive message detected for user_id={user_id}, handled by chat_agent")
    else:
        langchain_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages[-5:]]
        response_data = {
            "messages": langchain_messages,
            "additional_responses": additional_responses,
            "user_id": user_id,
            "chat_history": messages
        }
        logger.debug(f"Input to supervisor for user_id={user_id}: {response_data}")
        
        max_iterations = 10
        iteration = 0
        final_response = None
        agent_used = "chat_agent"  # Default

        while iteration < max_iterations:
            response_data = app_ai.invoke(response_data)
            iteration += 1
            next_agent = response_data.get("next_agent")
            logger.info(f"Supervisor iteration {iteration} for user_id={user_id}: Routing to {next_agent if next_agent else 'no further agent'}")

            # Check the last message for tool output or agent response
            last_message = response_data["messages"][-1]
            logger.debug(f"Last message in iteration {iteration}: {last_message}")

            if hasattr(last_message, "content") and last_message.content:  # AIMessage with content
                final_response = last_message.content
                agent_used = getattr(last_message, "name", "chat_agent")
            elif last_message.get("role") == "tool" and last_message.get("content"):  # Tool output
                final_response = last_message["content"]
                agent_used = response_data["messages"][-2].get("name", "Unknown Agent")  # Agent that called the tool
            elif isinstance(last_message, dict) and "user_response" in last_message:  # Therapy agent dict
                final_response = last_message["user_response"]
                agent_used = last_message.get("agent_name", "Unknown Agent")

            if not next_agent or iteration == max_iterations:
                break

        if iteration >= max_iterations:
            final_response = "I seem to be having trouble processing your request. Can you tell me more?"
            agent_used = "chat_agent"
            logger.warning(f"Max iterations reached for user_id={user_id}, defaulting to chat_agent")

        response = final_response
        logger.info(f"Final response for user_id={user_id} from {agent_used}: {response}")

        # Handle therapy agent email sending
        if agent_used == "therapy_expert" and isinstance(last_message, dict) and "therapy_plan" in last_message:
            success = send_therapy_plan_to_therapist(last_message["therapy_plan"], user_id)
            if not success:
                response += " (Note: There was an issue sending the plan to your therapist; we’ll retry later.)"
            logger.info(f"Therapy plan email attempt for user_id={user_id}: {'Success' if success else 'Failed'}")

    messages.append({
        "role": "assistant",
        "content": response,
        "agent_name": agent_used,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    store_user_data(user_id, messages, context)
    return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})
except Exception as e:
    logger.exception(f"Chat Error for user_id={user_id}: {str(e)}")
    return jsonify({"error": str(e)}), 500

@app.route("/welcome", methods=["POST"])
def welcome():
logger.info("Welcome endpoint called")
return jsonify({"response": "Welcome to SoulSync! How can I assist you today?"})

if __name__ == "__main__":
logger.info("Starting SoulSync application...")
app.run(debug=True, host="127.0.0.1", port=5000)