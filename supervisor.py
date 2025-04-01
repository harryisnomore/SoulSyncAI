# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor.supervisor import create_supervisor
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from datetime import datetime, timezone
import uuid
import json
import base64
from datetime import datetime, timezone
from cryptography.fernet import Fernet
import logging
import smtplib
from email.mime.text import MIMEText
import time

# Setup logging
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
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

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

cipher_suite= Fernet(ENCRYPTION_KEY.encode())

def encrypt_data(data: dict) -> str:
    """Encrypt dictionary data using Fernet."""
    json_data = json.dumps(data).encode()
    encrypted_data = cipher_suite.encrypt(json_data)
    return encrypted_data.decode()

def decrypt_data(encrypted_data: str) -> dict:
    """Decrypt Fernet-encrypted data back to dictionary."""
    decrypted_data = cipher_suite.decrypt(encrypted_data.encode())
    return json.loads(decrypted_data.decode())

def invoke_azure_openai(messages, system_prompt, max_retries=3):
    """Invoke Azure OpenAI API with error handling and rate limiting."""
    retry_count = 0
    base_delay = 1
    while retry_count < max_retries:
        try:
            response = model.invoke(
                [{"role": "system", "content": system_prompt}] + messages,
                temperature=0.7
            )
            return response.content.strip()
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                logger.error(f"invoke_azure_openai: Max retries reached: {str(e)}")
                return "Error processing request."
            delay = base_delay * (2 ** retry_count)
            logger.warning(f"Retrying in {delay}s due to {str(e)}")
            time.sleep(delay)

# Cosmos DB Helper Functions
def get_user_data(user_id: str) -> dict:
    """Retrieve and decrypt user data from Cosmos DB."""
    try:
        query = "SELECT * FROM c WHERE c.user_id = @user_id"
        parameters = [{"name": "@user_id", "value": user_id}]
        user_docs = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

        if not user_docs:
            user_doc = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "messages": encrypt_data([]),  # Encrypt empty list
                "context": encrypt_data({}),   # Encrypt empty dict
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            container.create_item(user_doc)
            logger.info(f"get_user_data: Created new user_doc for user_id={user_id}")
            return {"user_id": user_id, "messages": [], "context": {}}

        user_doc = user_docs[0]
        decrypted_messages = decrypt_data(user_doc["messages"])
        decrypted_context = decrypt_data(user_doc["context"])

        logger.debug(f"get_user_data: Retrieved user_doc for user_id={user_id}")
        return {"user_id": user_id, "messages": decrypted_messages, "context": decrypted_context}

    except Exception as e:
        logger.error(f"get_user_data: Error for user_id={user_id}: {str(e)}")
        raise


def store_user_data(user_id: str, messages: list, context: dict) -> None:
    """Store or update user data in Cosmos DB, ensuring valid input format."""
    try:
        if not user_id:
            raise ValueError("Invalid user_id: Cannot be None or empty")

        user_data = get_user_data(user_id)
        user_data["id"] = str(user_id)  # Ensure `id` is a string
        user_data["messages"] = messages
        user_data["context"] = context

        # Remove system-generated fields (if present)
        unwanted_keys = ["_rid", "_self", "_etag", "_attachments", "_ts"]
        for key in unwanted_keys:
            user_data.pop(key, None)

        # Log sanitized data
        logger.info(f"Upserting sanitized user_data: {json.dumps(user_data, indent=2)}")

        # Upsert into Cosmos DB
        container.upsert_item(user_data)

    except Exception as e:
        logger.error(f"store_user_data error for user_id={user_id}: {str(e)}")
        raise



# Core Functions
def get_summary(user_input: str) -> str:
    """Generate a concise summary of the user's input by extracting key emotions or concerns."""
    summary_prompt = """
    Summarize the user's input by extracting the key emotions or mental health concerns (e.g., stress, anxiety, overwhelmed, self-doubt) in a concise phrase. Avoid repeating the entire input. If no specific emotions are mentioned, return 'general emotional distress'.
    Example:
    Input: "Still feeling a bit overwhelmed, but I’m ready to try whatever might help."
    Output: feeling overwhelmed
    Input: "the difficulty in managing daily stress, and some feelings of self-doubt"
    Output: stress and self-doubt
    Input: "I’m not sure what’s wrong"
    Output: general emotional distress
    """
    messages = [{"role": "user", "content": f"Input: {user_input}"}]
    summary = invoke_azure_openai(messages, summary_prompt)
    return summary

def wellness_check(patient_mood: str, additional_responses: list = None) -> dict:
    """Assess the patient's mental health using Azure OpenAI for dynamic responses ."""
    summary = get_summary(patient_mood)
    
    if additional_responses is None or len(additional_responses) < 5:
        questions_prompt = f"""
        Based on the user's mood summary '{summary}', generate 5 empathetic, conversational questions to assess their mental health. Return them as a numbered list:
        1. A question about how this affects their daily life
        2. A yes/no question about mood swings
        3. A yes/no question about sleep
        4. A yes/no question about social connection
        5. A 1-10 scale question about emotional well-being
        Keep the tone warm and supportive.
        """
        questions_response = invoke_azure_openai([{"role": "user", "content": questions_prompt}], "Generate questions based on the summary.")
        questions = [line.split(". ", 1)[1] for line in questions_response.split("\n") if line.strip() and ". " in line]

        initial_prompt = f"""
        The user said: '{patient_mood}'. Their mood summary is '{summary}'. Craft a warm, empathetic response acknowledging their feelings and introducing the first question: '{questions[0]}'.
        """
        response = invoke_azure_openai([{"role": "user", "content": initial_prompt}], "Respond empathetically and introduce the first question.")
        
        return {
            "response": response,
            "state": "questions",
            "questions": questions,
            "agent_name": "wellness_check_expert",
            "continue_flow": True
        }
    
    responses = [r.lower() for r in additional_responses]
    risk_level = "low"
    if "yes" in responses or any(int(x) <= 7 for x in responses if x.isdigit()):  
        risk_level = "high"
    elif any(int(x) >= 4 for x in responses if x.isdigit()):
        risk_level = "medium"

    verdict_prompt = f"""
    The user’s mood was '{patient_mood}' (summary: '{summary}'). Their responses to 5 wellness questions were: {responses}. Risk level is '{risk_level}'.
    Craft a supportive, empathetic response summarizing their state, acknowledging their answers, and providing a verdict. For high risk, mention therapist contact; for medium, suggest self-care or therapy options; for low, offer encouragement. Keep it natural and warm.
    """
    response = invoke_azure_openai([{"role": "user", "content": verdict_prompt}], "Summarize and provide a verdict based on the user's responses.")
    
    return {
        "response": response,
        "state": "verdict",
        "agent_name": "wellness_check_expert",
        "continue_flow": False,
        "risk_level": risk_level
    }

def provide_therapy(condition: str, user_id: str = None, chat_history: list = None) -> dict:
    """Create a personalized therapy plan and email with full chat history and insights."""
    user_data = get_user_data(user_id)
    user_name = f"{user_data.get('firstName', '')} {user_data.get('lastName', '')}".strip() or "Unknown User"

    # Full chat history including user and agent interactions
    chat_summary = "No prior conversation provided."
    if chat_history:
        chat_summary = "Full Chat History:\n" + "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-6:]])

    # Refine condition and generate insights
    refine_prompt = """
    Based on the user's condition and chat history, refine the condition description to capture their emotional state and needs in a warm, natural way. Return a concise description.
    """
    messages = [{"role": "user", "content": f"Condition: {condition}\nChat History: {chat_summary}"}]
    refined_condition = invoke_azure_openai(messages, refine_prompt)

    insights_prompt = f"""
    Based on the chat history:
    {chat_summary}
    Provide a concise summary of insights into the user’s emotional state, including key emotions, triggers, and impacts on daily life. Keep it professional and clear for a therapist (e.g., 'The user expresses persistent stress from work, leading to sleep issues and fatigue').
    """
    insights = invoke_azure_openai(messages + [{"role": "user", "content": insights_prompt}], "Generate insights.")

    severity_prompt = """
    Assess the severity of the user's condition as 'low', 'moderate', or 'high'. Return only the severity level.
    """
    severity = invoke_azure_openai(messages, severity_prompt).lower()
    if severity not in ["low", "moderate", "high"]:
        severity = "moderate"

    triggers_prompt = """
    Identify specific triggers (e.g., work, family, health). Return a comma-separated list or 'None' if unclear.
    """
    triggers = invoke_azure_openai(messages, triggers_prompt).split(", ")
    if triggers == ["None"]:
        triggers = []

    therapy_prompt = f"""
    For a user named '{user_name}' with condition '{refined_condition}', severity '{severity}', and triggers '{', '.join(triggers)}', create a personalized therapy plan. Use a professional yet supportive tone and format it as follows:
    - **Therapy Type**: [type]
    - **Focus Areas**: [area1], [area2], [area3 if applicable]
    - **Actionable Recommendations**: [rec1]; [rec2]; [rec3 if applicable]
    - **Realistic Goals**:
      - Short-term: [short-term goal]
      - Long-term: [long-term goal]
    - **Supportive Communication**: [comm1]; [comm2 if applicable]
    """
    therapy_response = invoke_azure_openai(messages + [{"role": "user", "content": therapy_prompt}], "Create a therapy plan.")

    # Enhanced email with chat history and insights
    therapy_plan_text = f"""
Subject: Therapy Recommendation for {user_name},

Dear Therapist,

Please find below a personalized therapy plan for {user_name}, based on their recent interactions with SoulSync:

{therapy_response}

Insights from Conversation:
{insights}

{chat_summary}

Best regards,
SoulSync Team
    """

    user_response_prompt = f"""
    Based on the therapy plan: '{therapy_response}', craft a short, warm message to the user. Mention the therapy type, one focus area, one recommendation, and that a therapist will reach out. Keep it encouraging and casual, explaining why therapy is needed (e.g., 'to help with your stress and sleep struggles').
    """
    user_response = invoke_azure_openai([{"role": "user", "content": user_response_prompt}], "Generate a user-friendly response.")

    return {
        "user_response": user_response,
        "therapy_plan": therapy_plan_text,
        "agent_name": "therapy_expert"
    }

def post_rehab_followup(patient_status: str, user_id: str = None, chat_history: list = None) -> dict:
    """Assess post-rehab status, analyze therapy feedback, provide solutions, and escalate high-risk cases using Azure OpenAI."""
    # Summarize the patient's initial status
    summary = get_summary(patient_status)

    # Prepare chat history summary for context
    chat_summary = "No prior conversation provided."
    if chat_history:
        chat_summary = "Full Chat History:\n" + "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-6:]])

    # Analyze the user's post-rehab status using Azure OpenAI
    analysis_prompt = f"""
    Analyze the user's post-rehab status: '{patient_status}' (summary: '{summary}'). Consider the following aspects based on their input and chat history:
    {chat_summary}
    1. **Emotional State**: Identify key emotions (e.g., anxious, hopeful, overwhelmed).
    2. **Recovery Progress**: Assess their progress since rehab (e.g., improving, struggling, relapsed).
    3. **Support System**: Determine if they feel supported (e.g., by friends, family, or groups).
    4. **Therapy Feedback**: Extract any feedback or sentiments about their therapy experience (e.g., helpful, unhelpful, neutral).
    5. **Risk Factors**: Identify specific risk factors (e.g., recent setbacks, lack of support, negative therapy feedback).
    Return the analysis in the following format:
    - Emotional State: [emotions]
    - Recovery Progress: [progress]
    - Support System: [supported/unsupported/unclear]
    - Therapy Feedback: [positive/negative/neutral/unclear]
    - Risk Factors: [factor1], [factor2], [factor3 if applicable]
    """
    analysis_response = invoke_azure_openai([{"role": "user", "content": analysis_prompt}], "Analyze the user's post-rehab status.")
    
    # Parse the analysis response
    analysis = {}
    for line in analysis_response.split("\n"):
        if ": " in line:
            key, value = line.split(": ", 1)
            key = key.strip("- ").lower().replace(" ", "_")
            if key == "risk_factors":
                value = [factor.strip() for factor in value.strip("[]").split(", ") if factor.strip()]
            analysis[key] = value

    # Assess risk level using Azure OpenAI
    risk_prompt = f"""
    Based on the following analysis of the user's post-rehab status:
    - Emotional State: {analysis.get('emotional_state', 'unknown')}
    - Recovery Progress: {analysis.get('recovery_progress', 'unknown')}
    - Support System: {analysis.get('support_system', 'unclear')}
    - Therapy Feedback: {analysis.get('therapy_feedback', 'unclear')}
    - Risk Factors: {', '.join(analysis.get('risk_factors', []))}
    Assess the risk level as 'low', 'medium', or 'high'. Consider:
    - High risk: Significant setbacks (e.g., relapse), lack of support, negative therapy feedback, or severe emotional distress.
    - Medium risk: Struggling with recovery, moderate emotional distress, or some risk factors.
    - Low risk: Improving, supported, and minimal risk factors.
    Return only the risk level.
    """
    risk_level = invoke_azure_openai([{"role": "user", "content": risk_prompt}], "Assess risk level.").lower()

    # Generate tailored solutions and response
    solutions_prompt = f"""
    The user’s post-rehab status was '{patient_status}' (summary: '{summary}'). Analysis:
    - Emotional State: {analysis.get('emotional_state', 'unknown')}
    - Recovery Progress: {analysis.get('recovery_progress', 'unknown')}
    - Support System: {analysis.get('support_system', 'unclear')}
    - Therapy Feedback: {analysis.get('therapy_feedback', 'unclear')}
    - Risk Factors: {', '.join(analysis.get('risk_factors', []))}
    Risk level is '{risk_level}'.
    Craft a supportive, empathetic response summarizing their state and providing tailored solutions:
    - Acknowledge their emotional state and recovery progress.
    - Comment on their therapy feedback (e.g., "It’s great to hear therapy has been helpful" or "I’m sorry therapy hasn’t been as helpful as hoped").
    - Provide solutions based on their risk level and analysis:
      - High risk: Recommend immediate therapy, coping strategies (e.g., mindfulness), and support resources.
      - Medium risk: Suggest support groups, self-care practices, and a follow-up with their therapist.
      - Low risk: Offer praise, encourage continued progress, and suggest maintenance strategies (e.g., journaling).
    Keep the tone natural, warm, and non-judgmental.
    """
    response = invoke_azure_openai([{"role": "user", "content": solutions_prompt}], "Generate a response with tailored solutions.")

    # Prepare return data
    return_data = {
        "response": response,
        "state": "verdict",
        "agent_name": "post_rehab_expert",
        "continue_flow": False,
        "risk_level": risk_level
    }

    # Escalate high-risk cases to the therapy_expert
    if risk_level == "high":
        therapy_data = provide_therapy(summary, user_id, chat_history)
        response += "\n\nGiven the challenges you’re facing, I’ve escalated your case to our therapy expert. " + therapy_data["user_response"]
        return_data["response"] = response
        return_data["agent_name"] = therapy_data["agent_name"]
        send_therapy_plan_to_therapist(therapy_data["therapy_plan"], user_id)

    return return_data

def send_therapy_plan_to_therapist(therapy_plan: str, user_id: str) -> bool:
    """Send the therapy plan to the therapist via email."""
    therapist_email = os.getenv("THERAPIST_EMAIL")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([therapist_email, smtp_server, smtp_port, smtp_user, smtp_password]):
        logger.error("Missing SMTP configuration")
        return False

    subject = therapy_plan.split("\n")[0].strip() if therapy_plan.startswith("Subject:") else f"Therapy Plan for User {user_id}"
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

# Agents
logger.info("Creating agents...")
wellness_check_agent = create_react_agent(
    model=model,
    tools=[wellness_check],
    name="wellness_check_expert",
    prompt="""
    You’re a compassionate mental health companion. When a user shares their mood or emotions:
    1. Pass their message to 'wellness_check' as 'patient_mood'.
    2. If 'additional_responses' are provided, include them to get a risk verdict.
    3. Return the 'response' field as-is—trust the tool to craft something warm and tailored.
    The tool might start a question flow; let the chat endpoint manage it.
    """
)

therapy_agent = create_react_agent(
    model=model,
    tools=[provide_therapy, send_therapy_plan_to_therapist],
    name="therapy_expert",
    prompt="""
    You’re a friendly therapy guide. When a user needs support or requests therapy:
    1. Use 'provide_therapy' with their message as 'condition', plus 'user_id' and 'chat_history'.
    2. Pass the 'therapy_plan' and 'user_id' to 'send_therapy_plan_to_therapist'.
    3. Return the 'user_response'—it’s already crafted to be encouraging and personal.
    """
)

post_rehab_agent = create_react_agent(
    model=model,
    tools=[post_rehab_followup],
    name="post_rehab_expert",
    prompt="""
    You’re a supportive post-rehab ally for users navigating life after rehabilitation. When a user mentions their post-rehab experience:
    1. Pass their message to 'post_rehab_followup' as 'patient_status', along with 'user_id' and 'chat_history'.
    2. Return the 'response' field—it’ll be empathetic, tailored, and may include escalation to a therapist if high risk is detected.
    The tool will analyze the user's input and provide a comprehensive response.
    """
)

# Supervisor Agent
logger.info("Creating supervisor agent...")
try:
    chat_agent = create_supervisor(
        agents=[wellness_check_agent, therapy_agent, post_rehab_agent],
        model=model,
        prompt=
        """You’re the warm, guiding heart of SoulSync, a mental health support platform. Your job is to listen closely and either answer mental health-related informational queries directly or route user queries to the right agent—wellness_check_expert, therapy_expert, or post_rehab_expert—based on what they share, but only for mental health, therapy, or rehab topics. Anything else is out of scope.

        Here’s how to help:

1. **Greeting Handling**:
   - For greetings (e.g., "Hello," "Hi," "Hey"), use Azure OpenAI to respond with a warm, inviting message nudging them to share feelings. Example input: "Hey" → "Hey there! I’m so glad you’re here—what’s on your mind today?"

2. **Scope Check**:
   - Stick to mental health, emotional well-being, therapy, or rehab. For unrelated queries (e.g., "What’s the weather?" or "What is Python?"), say: "I’m sorry, I’m here just for mental health and support stuff. What’s on your mind today—any feelings or challenges you’d like to share?"

3. **Informational Mental Health Queries**:
   - If the query is a general or informational question about mental health, emotional well-being, therapy, or rehab (e.g., "What is mental health?", "What are the best practices to improve mental health?", "What factors affect mental health?"), use Azure OpenAI to provide a concise, supportive answer. Example:
     - Input: "What is mental health?"
     - Output: "Mental health refers to your emotional, psychological, and social well-being—it’s how you think, feel, and handle life’s ups and downs. Does that spark any thoughts or feelings you’d like to share?"
     - Input: "What are the best practices to improve mental health?"
     - Output: "Some great ways to improve mental health include practicing mindfulness, staying connected with loved ones, getting enough sleep, and seeking professional support when needed. What’s been on your mind lately—any challenges you’d like to explore?"

4. **Routing for Support-Oriented Queries**:
   - Therapy requests (e.g., "I need therapy for stress"): Route to therapy_expert with a note like, “I hear you, and I’m getting our therapy expert to help right away.” and give the next response of the therapy_expert right away.
   - Emotional wellness: If they mention feelings like stress, sadness, or anxiety (e.g., "I’m so stressed lately"), check context:
     - If recent wellness check (context["wellness_check_completed"] == true) and "high" or "medium" risk, route to therapy_expert: “You’ve been through a lot—let’s get you therapy support for {context["last_summary"]}.”
     - Otherwise, route to wellness_check_expert: “I’d love to check in on how you’re doing—let’s start there.”
   - Post-rehab hints (e.g., "I’m struggling since rehab"): Route to post_rehab_expert: “Thanks for sharing—it’s a big step. Let’s talk about life after rehab.”
   - General mental health (e.g., "How do I feel less anxious?"): Lean toward wellness_check_expert unless therapy feels urgent: “Let’s figure out what’s going on first.”

5. **Multi-Step Flows**:
   - If 'continue_flow': true, step back—let the chat endpoint handle it. Jump in only when 'continue_flow': false or they switch topics.

6. **Clarification**:
   - If vague (e.g., "I’m not okay"), ask: “I’m here for you—can you tell me more about what’s feeling off emotionally?”
   - If specific (e.g., "I’m anxious"), route directly.

Be warm and caring—make them feel safe. Use Azure OpenAI for greetings, scope errors, and informational mental health queries; otherwise, route and let agents handle responses. You’re the heart of SoulSync—let’s make a difference together!""",
        supervisor_name="supervisor",
        include_agent_name="inline",
        output_mode="full_history"
    )
    logger.info("Supervisor agent created successfully")
except Exception as e:
    logger.error(f"Failed to create chat_agent: {str(e)}")
    raise

app_ai = chat_agent.compile()
logger.info("Workflow compiled successfully")

# API Routes
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    required_fields = ["firstName", "lastName", "dob", "email", "password"]
    
    if not all(field in data for field in required_fields):
        logger.warning("Signup failed: Missing required fields")
        return jsonify({"error": "Missing required fields"}), 400

    query = "SELECT * FROM c WHERE c.email = @email"
    parameters = [{"name": "@email", "value": data['email']}]
    existing_users = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
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

    query = "SELECT * FROM c WHERE c.email = @email"
    parameters = [{"name": "@email", "value": data['email']}]
    users = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    
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
        user_data = get_user_data(user_id)
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

    if not user_message or not user_id:
        logger.warning("Chat request failed: Missing user_id or message")
        return jsonify({"error": "Missing user_id or message"}), 400

    try:
        user_data = get_user_data(user_id)
        messages = user_data.get("messages", [])
        context = user_data.get("context", {})
        logger.debug(f"Initial context for user_id={user_id}: {context}")

        messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching user data for user_id={user_id}: {str(e)}")
        return jsonify({"error": "Failed to fetch user data"}), 500

    # Handle Wellness Check Flow
    if context.get("wellness_questions") and context.get("current_question_index") is not None:
        current_question_index = context["current_question_index"]
        wellness_answers = context.get("wellness_answers", [])
        wellness_questions = context["wellness_questions"]

        if current_question_index in [1, 2, 3]:
            if user_message.lower() not in ["yes", "no"]:
                response = "I’d love a simple 'Yes' or 'No' here—whatever feels true for you!"
                agent_used = "wellness_check_expert"
                messages.append({"role": "assistant", "content": response, "agent_name": agent_used, "timestamp": datetime.now(timezone.utc).isoformat()})
                store_user_data(user_id, messages, context)
                return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})
        elif current_question_index == 4:
            try:
                response_value = int(user_message)
                if not (1 <= response_value <= 10):
                    response = "Could you give me a number between 1 and 10? That’ll help me understand how you’re feeling!"
                    agent_used = "wellness_check_expert"
                    messages.append({"role": "assistant", "content": response, "agent_name": agent_used, "timestamp": datetime.now(timezone.utc).isoformat()})
                    store_user_data(user_id, messages, context)
                    return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})
            except ValueError:
                response = "Just a number between 1 and 10 would be perfect—whenever you’re ready!"
                agent_used = "wellness_check_expert"
                messages.append({"role": "assistant", "content": response, "agent_name": agent_used, "timestamp": datetime.now(timezone.utc).isoformat()})
                store_user_data(user_id, messages, context)
                return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})

        wellness_answers.append(user_message)
        context["wellness_answers"] = wellness_answers
        logger.debug(f"Stored user response for question {current_question_index}: {user_message}")

        if current_question_index + 1 < len(wellness_questions):
            context["current_question_index"] = current_question_index + 1
            next_question = wellness_questions[current_question_index + 1]
            transition_prompt = f"""
            The user answered '{user_message}' to '{wellness_questions[current_question_index]}'. Craft a warm, empathetic transition to the next question: '{next_question}'. Keep it natural and supportive, reflecting on their last answer.
            """
            response = invoke_azure_openai([{"role": "user", "content": transition_prompt}], "Generate a transition.")
            agent_used = "wellness_check_expert"
        else:
            response_data = wellness_check(context.get("last_patient_mood", ""), wellness_answers)
            response = response_data["response"]
            agent_used = response_data["agent_name"]
            context["wellness_check_completed"] = True
            context["last_summary"] = get_summary(context.get("last_patient_mood", ""))
            context["risk_level"] = response_data.get("risk_level", "low")
            context.pop("wellness_questions", None)
            context.pop("wellness_answers", None)
            context.pop("current_question_index", None)
            context.pop("last_patient_mood", None)

            if context["risk_level"] in ["high", "medium"]:
                therapy_data = provide_therapy(context["last_summary"], user_id, messages)
                response += "\n\n" + therapy_data["user_response"]
                agent_used = therapy_data["agent_name"]
                send_therapy_plan_to_therapist(therapy_data["therapy_plan"], user_id)
    else:
        # Check for mental health or post-rehab keywords to initiate flows
        last_user_message = user_message.lower()
        mental_health_keywords = ["stress", "anxious", "overthinking", "doubt", "sad", "overwhelmed"]
        post_rehab_keywords = ["rehab", "recovery", "relapse", "struggling since rehab", "after rehab"]

        # Initiate Wellness Check Flow
        if any(keyword in last_user_message for keyword in mental_health_keywords):
            if context.get("wellness_check_completed") and context.get("risk_level") in ["high", "medium"]:
                logger.debug(f"User has completed a wellness check with risk level {context['risk_level']}, routing to therapy_expert")
                therapy_data = provide_therapy(context.get("last_summary", user_message), user_id, messages)
                response = therapy_data["user_response"]
                agent_used = therapy_data["agent_name"]
                send_therapy_plan_to_therapist(therapy_data["therapy_plan"], user_id)
                messages.append({
                    "role": "assistant",
                    "content": response,
                    "agent_name": agent_used,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                store_user_data(user_id, messages, context)
                return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})

            logger.debug(f"Starting wellness check flow for user_id={user_id} due to mental health keywords")
            response_data = wellness_check(user_message)
            response = response_data["response"]
            agent_used = response_data["agent_name"]
            if response_data.get("state") == "questions" and "questions" in response_data:
                context["wellness_questions"] = response_data["questions"]
                context["wellness_answers"] = []
                context["current_question_index"] = 0
                context["last_patient_mood"] = user_message
                logger.debug(f"Started wellness check flow with context: {context}")
                messages.append({
                    "role": "assistant",
                    "content": response,
                    "agent_name": agent_used,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                store_user_data(user_id, messages, context)
                return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})

        # Initiate Post-Rehab Follow-Up (Single-Step Analysis)
        elif any(keyword in last_user_message for keyword in post_rehab_keywords):
            if context.get("post_rehab_check_completed") and context.get("post_rehab_risk_level") in ["high", "medium"]:
                logger.debug(f"User has completed a post-rehab check with risk level {context['post_rehab_risk_level']}, routing to therapy_expert")
                therapy_data = provide_therapy(context.get("last_post_rehab_summary", user_message), user_id, messages)
                response = therapy_data["user_response"]
                agent_used = therapy_data["agent_name"]
                send_therapy_plan_to_therapist(therapy_data["therapy_plan"], user_id)
                messages.append({
                    "role": "assistant",
                    "content": response,
                    "agent_name": agent_used,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                store_user_data(user_id, messages, context)
                return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})

            logger.debug(f"Starting post-rehab follow-up analysis for user_id={user_id} due to post-rehab keywords")
            response_data = post_rehab_followup(user_message, user_id, messages)
            response = response_data["response"]
            agent_used = response_data["agent_name"]
            context["post_rehab_check_completed"] = True
            context["last_post_rehab_summary"] = get_summary(user_message)
            context["post_rehab_risk_level"] = response_data.get("risk_level", "low")
            logger.debug(f"Completed post-rehab follow-up with context: {context}")
            messages.append({
                "role": "assistant",
                "content": response,
                "agent_name": agent_used,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            store_user_data(user_id, messages, context)
            return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})

        # Default: Let the supervisor handle the message
        langchain_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages[-5:]]
        input_data = {
            "messages": langchain_messages,
            "additional_responses": [],
            "user_id": user_id,
            "chat_history": messages,
            "context": context
        }

        response_data = app_ai.invoke(input_data)
        last_message = response_data["messages"][-1]

        if isinstance(last_message, dict) and "response" in last_message:
            response = last_message["response"]
            agent_used = last_message.get("agent_name", "supervisor")
            if last_message.get("state") == "questions" and "questions" in last_message:
                context["wellness_questions"] = last_message["questions"]
                context["wellness_answers"] = []
                context["current_question_index"] = 0
                context["last_patient_mood"] = user_message
                response = last_message["response"]
                agent_used = "wellness_check_expert"
        elif hasattr(last_message, "content") and last_message.content:
            if user_message.lower() in ["hello", "hi", "hey", "hii"]:
                greeting_prompt = f"""
                The user said '{user_message}'. Generate a warm, inviting response encouraging them to share their feelings or concerns about mental health.
                """
                response = invoke_azure_openai([{"role": "user", "content": greeting_prompt}], "Generate a greeting.")
                agent_used = "supervisor"
            else:
                response = last_message.content
                agent_used = getattr(last_message, "name", "supervisor")
        else:
            response = "I’m having a little trouble here—could you tell me more about what’s on your mind?"
            agent_used = "supervisor"

    messages.append({
        "role": "assistant",
        "content": response,
        "agent_name": agent_used,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    store_user_data(user_id, messages, context)
    logger.info(f"Final response for user_id={user_id} from {agent_used}: {response}")
    return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})

@app.route("/welcome", methods=["POST"])
def welcome():
    logger.info("Welcome endpoint called")
    return jsonify({"response": "Welcome to SoulSync! Your mental health companion"})

if __name__ == "__main__":
    logger.info("Starting SoulSync application...")
    app.run(debug=True, host="127.0.0.1", port=5000)


# app.py
# app.py
# app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# from langchain_openai import AzureChatOpenAI
# from langgraph.prebuilt import create_react_agent
# from langgraph_supervisor.supervisor import create_supervisor
# from dotenv import load_dotenv
# from azure.cosmos import CosmosClient
# from datetime import datetime, timezone
# import uuid
# import logging
# import smtplib
# from email.mime.text import MIMEText
# import time

# # Setup logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)
# CORS(app)

# # Azure Cosmos DB Configuration
# COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
# COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
# COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
# COSMOS_DB_CONTAINER = os.getenv("COSMOS_DB_CONTAINER")

# if not all([COSMOS_DB_ENDPOINT, COSMOS_DB_KEY, COSMOS_DB_NAME, COSMOS_DB_CONTAINER]):
#     raise ValueError("Missing required Cosmos DB environment variables")

# cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
# database = cosmos_client.get_database_client(COSMOS_DB_NAME)
# container = database.get_container_client(COSMOS_DB_CONTAINER)

# # Initialize Azure OpenAI Model
# model = AzureChatOpenAI(
#     azure_endpoint=os.getenv("ENDPOINT"),
#     api_key=os.getenv("API_KEY"),
#     api_version="2023-05-15",
#     deployment_name="gpt-4o-mini"
# )

# def invoke_azure_openai(messages, system_prompt, max_retries=3):
#     """Invoke Azure OpenAI API with error handling and rate limiting."""
#     retry_count = 0
#     base_delay = 1
#     while retry_count < max_retries:
#         try:
#             response = model.invoke(
#                 [{"role": "system", "content": system_prompt}] + messages,
#                 temperature=0.7
#             )
#             return response.content.strip()
#         except Exception as e:
#             retry_count += 1
#             if retry_count == max_retries:
#                 logger.error(f"invoke_azure_openai: Max retries reached: {str(e)}")
#                 return "Error processing request."
#             delay = base_delay * (2 ** retry_count)
#             logger.warning(f"Retrying in {delay}s due to {str(e)}")
#             time.sleep(delay)

# # Cosmos DB Helper Functions
# def get_user_data(user_id: str) -> dict:
#     """Retrieve user data from Cosmos DB."""
#     try:
#         query = "SELECT * FROM c WHERE c.user_id = @user_id"
#         parameters = [{"name": "@user_id", "value": user_id}]
#         user_docs = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
#         if not user_docs:
#             user_doc = {
#                 "id": str(uuid.uuid4()),
#                 "user_id": user_id,
#                 "messages": [],
#                 "context": {},
#                 "created_at": datetime.now(timezone.utc).isoformat()
#             }
#             container.create_item(user_doc)
#             logger.info(f"get_user_data: Created new user_doc for user_id={user_id}")
#             return user_doc
#         logger.debug(f"get_user_data: Retrieved user_doc for user_id={user_id}: {user_docs[0]}")
#         return user_docs[0]
#     except Exception as e:
#         logger.error(f"get_user_data: Error for user_id={user_id}: {str(e)}")
#         raise

# def store_user_data(user_id: str, messages: list, context: dict) -> None:
#     """Store or update user data in Cosmos DB."""
#     try:
#         user_data = get_user_data(user_id)
#         user_data["messages"] = messages
#         user_data["context"] = context
#         user_data["updated_at"] = datetime.now(timezone.utc).isoformat()
#         container.upsert_item(user_data)
#         logger.debug(f"store_user_data: Stored user data for user_id={user_id}, context={context}")
#     except Exception as e:
#         logger.error(f"store_user_data: Error for user_id={user_id}: {str(e)}")
#         raise

# # Core Functions
# def get_summary(user_input: str) -> str:
#     """Generate a concise summary of the user's input by extracting key emotions or concerns."""
#     summary_prompt = """
#     Summarize the user's input by extracting the key emotions or mental health concerns (e.g., stress, anxiety, overwhelmed, self-doubt) in a concise phrase. Avoid repeating the entire input. If no specific emotions are mentioned, return 'general emotional distress'.
#     """
#     messages = [{"role": "user", "content": f"Input: {user_input}"}]
#     summary = invoke_azure_openai(messages, summary_prompt)
#     return summary

# def wellness_check(patient_mood: str, additional_responses: list = None, max_convo_turns: int = 3) -> dict:
#     """
#     Engage the user in a multi-turn conversation to assess their mental health, dynamically adapting the flow.
    
#     Args:
#         patient_mood (str): The user's initial mood or statement.
#         additional_responses (list, optional): Prior user responses in the conversation.
#         max_convo_turns (int): Max turns before forcing a verdict or questions (default: 3).
    
#     Returns:
#         dict: Response details including state, response text, and flow control flags.
#     """
#     summary = get_summary(patient_mood)
#     logger.debug(f"Wellness check started - patient_mood='{patient_mood}', summary='{summary}'")

#     # Initial conversation start
#     if additional_responses is None:
#         initial_prompt = f"""
#         The user said: '{patient_mood}'. Their mood summary is '{summary}'. 
#         Craft a warm, empathetic response acknowledging their feelings and gently probing for more insight. 
#         Keep it open-ended, conversational, and avoid direct yes/no or numbered questions.
#         """
#         response = invoke_azure_openai([{"role": "user", "content": initial_prompt}], "Start a conversation.")
#         logger.info(f"Wellness check - initial response: '{response}'")
#         return {
#             "response": response,
#             "state": "conversation",
#             "agent_name": "wellness_check_expert",
#             "continue_flow": True
#         }

#     # Build conversation history
#     convo_turns = len(additional_responses)
#     convo_history = f"Initial: '{patient_mood}'" + "\n" + "\n".join(
#         [f"Turn {i+1}: '{resp}'" for i, resp in enumerate(additional_responses)]
#     )
#     logger.debug(f"Wellness check - convo_turns={convo_turns}, history: {convo_history}")

#     # Check clarity after each turn
#     clarity_prompt = f"""
#     Based on the conversation so far:
#     {convo_history}
#     Assess if the user's emotional state is clear enough to summarize (e.g., 'stress and isolation'). 
#     Return 'clear' if sufficient, 'unclear' if more detail is needed.
#     """
#     clarity = invoke_azure_openai([{"role": "user", "content": clarity_prompt}], "Check clarity.").lower()
#     logger.debug(f"Wellness check - clarity: '{clarity}'")

#     if clarity == "clear" or convo_turns >= max_convo_turns:
#         # Summarize and assess risk
#         convo_summary_prompt = f"""
#         The full conversation is:
#         {convo_history}
#         Summarize the user's emotional state (e.g., 'stress and isolation'). If still unclear, return 'unclear'.
#         """
#         convo_summary = invoke_azure_openai([{"role": "user", "content": convo_summary_prompt}], "Summarize conversation.")
#         logger.debug(f"Wellness check - summary: '{convo_summary}'")

#         if convo_summary.lower() != "unclear":
#             risk_prompt = f"""
#             Based on the summary '{convo_summary}', assess the risk level as 'low', 'medium', or 'high'. 
#             Consider emotional intensity, frequency, and daily life impacts. Return only the risk level.
#             """
#             risk_level = invoke_azure_openai([{"role": "user", "content": risk_prompt}], "Assess risk.").lower()
#             if risk_level not in ["low", "medium", "high"]:
#                 risk_level = "medium"
#                 logger.warning(f"Wellness check - invalid risk level, defaulting to 'medium'")

#             verdict_prompt = f"""
#             Summary: '{convo_summary}', risk level: '{risk_level}'. Craft a supportive, empathetic response summarizing 
#             their state, acknowledging their openness, and explaining:
#             - High/medium risk: Why therapy is recommended (e.g., 'ongoing stress affecting sleep').
#             - Low risk: Why therapy isn’t needed now (e.g., 'manageable with small steps') and suggest self-care.
#             Keep it natural, warm, and transparent.
#             """
#             response = invoke_azure_openai([{"role": "user", "content": verdict_prompt}], "Provide verdict.")
#             logger.info(f"Wellness check - verdict: risk_level='{risk_level}', response='{response}'")
#             return {
#                 "response": response,
#                 "state": "verdict",
#                 "agent_name": "wellness_check_expert",
#                 "continue_flow": False,
#                 "risk_level": risk_level,
#                 "convo_summary": convo_summary
#             }
#         else:
#             return _transition_to_questions(convo_history)
#     else:
#         # Continue conversation
#         continue_prompt = f"""
#         The conversation so far is:
#         {convo_history}
#         Craft a warm, empathetic response building on what the user shared, encouraging elaboration. 
#         Keep it open-ended and conversational.
#         """
#         response = invoke_azure_openai([{"role": "user", "content": continue_prompt}], "Continue conversation.")
#         logger.debug(f"Wellness check - continuing convo, turn {convo_turns + 1}: '{response}'")
#         return {
#             "response": response,
#             "state": "conversation",
#             "agent_name": "wellness_check_expert",
#             "continue_flow": True
#         }

# def _transition_to_questions(convo_history: str) -> dict:
#     """Transition to structured questions when clarity is insufficient."""
#     questions_prompt = f"""
#     Based on the conversation:
#     {convo_history}
#     Generate 5 empathetic questions to assess mental health, as a numbered list:
#     1. How this affects their daily life
#     2. Yes/no about mood swings
#     3. Yes/no about sleep
#     4. Yes/no about social connection
#     5. 1-10 scale about emotional well-being
#     Keep the tone warm and context-aware.
#     """
#     questions_response = invoke_azure_openai([{"role": "user", "content": questions_prompt}], "Generate questions.")
#     questions = [line.split(". ", 1)[1] for line in questions_response.split("\n") if line.strip() and ". " in line]
#     logger.debug(f"Wellness check - generated questions: {questions}")

#     transition_prompt = f"""
#     The conversation so far is:
#     {convo_history}
#     The mood is unclear. Craft a warm response explaining the shift to specific questions, then introduce: '{questions[0]}'.
#     Example: "I really appreciate you sharing—I’d love to understand more. Can I ask a few gentle questions? 
#     Here’s the first: {questions[0]}"
#     """
#     response = invoke_azure_openai([{"role": "user", "content": transition_prompt}], "Transition to questions.")
#     logger.info(f"Wellness check - transitioning to questions: '{response}'")
#     return {
#         "response": response,
#         "state": "questions",
#         "questions": questions,
#         "agent_name": "wellness_check_expert",
#         "continue_flow": True
#     }

# def provide_therapy(condition: str, user_id: str = None, chat_history: list = None) -> dict:
#     """Create a personalized therapy plan and email with full chat history and insights."""
#     user_data = get_user_data(user_id)
#     user_name = f"{user_data.get('firstName', '')} {user_data.get('lastName', '')}".strip() or "Unknown User"

#     # Full chat history including user and agent interactions
#     chat_summary = "No prior conversation provided."
#     if chat_history:
#         chat_summary = "Full Chat History:\n" + "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-6:]])

#     # Refine condition and generate insights
#     refine_prompt = """
#     Based on the user's condition and chat history, refine the condition description to capture their emotional state and needs in a warm, natural way. Return a concise description.
#     """
#     messages = [{"role": "user", "content": f"Condition: {condition}\nChat History: {chat_summary}"}]
#     refined_condition = invoke_azure_openai(messages, refine_prompt)

#     insights_prompt = f"""
#     Based on the chat history:
#     {chat_summary}
#     Provide a concise summary of insights into the user’s emotional state, including key emotions, triggers, and impacts on daily life. Keep it professional and clear for a therapist (e.g., 'The user expresses persistent stress from work, leading to sleep issues and fatigue').
#     """
#     insights = invoke_azure_openai(messages + [{"role": "user", "content": insights_prompt}], "Generate insights.")

#     severity_prompt = """
#     Assess the severity of the user's condition as 'low', 'moderate', or 'high'. Return only the severity level.
#     """
#     severity = invoke_azure_openai(messages, severity_prompt).lower()
#     if severity not in ["low", "moderate", "high"]:
#         severity = "moderate"

#     triggers_prompt = """
#     Identify specific triggers (e.g., work, family, health). Return a comma-separated list or 'None' if unclear.
#     """
#     triggers = invoke_azure_openai(messages, triggers_prompt).split(", ")
#     if triggers == ["None"]:
#         triggers = []

#     therapy_prompt = f"""
#     For a user named '{user_name}' with condition '{refined_condition}', severity '{severity}', and triggers '{', '.join(triggers)}', create a personalized therapy plan. Use a professional yet supportive tone and format it as follows:
#     - **Therapy Type**: [type]
#     - **Focus Areas**: [area1], [area2], [area3 if applicable]
#     - **Actionable Recommendations**: [rec1]; [rec2]; [rec3 if applicable]
#     - **Realistic Goals**:
#       - Short-term: [short-term goal]
#       - Long-term: [long-term goal]
#     - **Supportive Communication**: [comm1]; [comm2 if applicable]
#     """
#     therapy_response = invoke_azure_openai(messages + [{"role": "user", "content": therapy_prompt}], "Create a therapy plan.")

#     # Enhanced email with chat history and insights
#     therapy_plan_text = f"""
# Subject: Therapy Recommendation for {user_name}

# Dear Therapist,

# Please find below a personalized therapy plan for {user_name}, based on their recent interactions with SoulSync:

# {therapy_response}

# Insights from Conversation:
# {insights}

# {chat_summary}

# Best regards,
# SoulSync Team
#     """

#     user_response_prompt = f"""
#     Based on the therapy plan: '{therapy_response}', craft a short, warm message to the user. Mention the therapy type, one focus area, one recommendation, and that a therapist will reach out. Keep it encouraging and casual, explaining why therapy is needed (e.g., 'to help with your stress and sleep struggles').
#     """
#     user_response = invoke_azure_openai([{"role": "user", "content": user_response_prompt}], "Generate a user-friendly response.")

#     return {
#         "user_response": user_response,
#         "therapy_plan": therapy_plan_text,
#         "agent_name": "therapy_expert"
#     }

# def post_rehab_followup(patient_status: str) -> str:
#     """Assess post-rehab status using Azure OpenAI for dynamic responses."""
#     patient_status = patient_status.lower()
#     high_risk_keywords = ["relapse", "not better", "worse", "overwhelmed"]
#     medium_risk_keywords = ["struggling", "some issues", "not fully okay"]
#     low_risk_keywords = ["improved", "better", "recovering"]
    
#     risk_level = "unknown"
#     if any(word in patient_status for word in high_risk_keywords):
#         risk_level = "high"
#     elif any(word in patient_status for word in medium_risk_keywords):
#         risk_level = "medium"
#     elif any(word in patient_status for word in low_risk_keywords):
#         risk_level = "low"

#     prompt = f"""
#     The user said: '{patient_status}'. Risk level is '{risk_level}' based on keywords. Craft an empathetic, encouraging response. For high risk, suggest immediate therapy; for medium, recommend support options; for low, offer praise; for unknown, seek more info. Keep it natural and supportive.
#     """
#     return invoke_azure_openai([{"role": "user", "content": prompt}], "Generate a post-rehab response.")

# def send_therapy_plan_to_therapist(therapy_plan: str, user_id: str) -> bool:
#     """Send the therapy plan to the therapist via email."""
#     therapist_email = os.getenv("THERAPIST_EMAIL")
#     smtp_server = os.getenv("SMTP_SERVER")
#     smtp_port = int(os.getenv("SMTP_PORT", 587))
#     smtp_user = os.getenv("SMTP_USER")
#     smtp_password = os.getenv("SMTP_PASSWORD")

#     if not all([therapist_email, smtp_server, smtp_port, smtp_user, smtp_password]):
#         logger.error("Missing SMTP configuration")
#         return False

#     subject = therapy_plan.split("\n")[0].strip() if therapy_plan.startswith("Subject:") else f"Therapy Plan for User {user_id}"
#     msg = MIMEText(therapy_plan)
#     msg["Subject"] = subject
#     msg["From"] = smtp_user
#     msg["To"] = therapist_email

#     try:
#         with smtplib.SMTP(smtp_server, smtp_port) as server:
#             server.starttls()
#             server.login(smtp_user, smtp_password)
#             server.send_message(msg)
#         logger.info(f"Therapy plan emailed to {therapist_email} for user_id={user_id}")
#         return True
#     except Exception as e:
#         logger.error(f"Failed to send email for user_id={user_id}: {str(e)}")
#         return False

# # Agents
# logger.info("Creating agents...")
# wellness_check_agent = create_react_agent(
#     model=model,
#     tools=[wellness_check],
#     name="wellness_check_expert",
#     prompt="""
#     You’re a compassionate mental health companion. When a user shares their mood or emotions:
#     1. Pass their message to 'wellness_check' as 'patient_mood'.
#     2. If 'additional_responses' are provided, include them to continue the conversation or get a risk verdict.
#     3. Return the 'response' field as-is—trust the tool to craft something warm and tailored.
#     The tool will engage in a multi-turn conversation (up to 3 turns) before assessing mood or transitioning to questions; let the chat endpoint manage the flow.
#     """
# )

# therapy_agent = create_react_agent(
#     model=model,
#     tools=[provide_therapy, send_therapy_plan_to_therapist],
#     name="therapy_expert",
#     prompt="""
#     You’re a friendly therapy guide. When a user needs support or requests therapy:
#     1. Use 'provide_therapy' with their message as 'condition', plus 'user_id' and 'chat_history'.
#     2. Pass the 'therapy_plan' and 'user_id' to 'send_therapy_plan_to_therapist'.
#     3. Return the 'user_response'—it’s already crafted to be encouraging and personal.
#     """
# )

# post_rehab_agent = create_react_agent(
#     model=model,
#     tools=[post_rehab_followup],
#     name="post_rehab_expert",
#     prompt="""
#     You’re a supportive post-rehab ally. When a user mentions life after rehab:
#     1. Send their message to 'post_rehab_followup' as 'patient_status'.
#     2. Return the response—it’ll be empathetic and tailored to their needs.
#     """
# )

# # Supervisor Agent
# logger.info("Creating supervisor agent...")
# try:
#     chat_agent = create_supervisor(
#         agents=[wellness_check_agent, therapy_agent, post_rehab_agent],
#         model=model,
#         prompt="""
#     You’re the warm, guiding heart of SoulSync, a mental health support platform. Your job is to listen closely 
#     and route user queries to the right agent—wellness_check_expert, therapy_expert, or post_rehab_expert—based 
#     on what they share, but only for mental health, therapy, or rehab topics.

#     1. **Initial Message**:
#        - If 'messages' is empty (app just opened): Respond with a warm greeting, e.g., "Hey there! I’m here to help—what’s on your mind today? Any feelings or challenges you’d like to share?"

#     2. **Greeting Handling**:
#        - For greetings (e.g., "Hello," "Hi"), respond: "Hey! I’m here for you—how are you feeling today?"

#     3. **Scope Check**:
#        - For unrelated queries, say: "I’m sorry, I’m here just for mental health and support stuff. What’s on your mind today—any feelings or challenges you’d like to share?"

#     4. **Routing**:
#        - Mental health keywords (e.g., "stress", "anxious"): Route to wellness_check_expert with, “I hear you—let’s check in on how you’re feeling.”
#        - Therapy requests: Route to therapy_expert with, “I hear you—let’s get our therapy expert to help.”
#        - Post-rehab: Route to post_rehab_expert with, “Let’s talk about life after rehab.”

#     5. **Multi-Step Flows**:
#        - If 'continue_flow': true, step back. Jump in when 'continue_flow': false or topic switches.

#     6. **Clarification**:
#        - If vague (e.g., "I’m not okay"), ask: “I’m here—can you tell me more about what’s feeling off?”

#     Be warm and caring—make them feel safe.
#     """,
#         supervisor_name="supervisor",
#         include_agent_name="inline",
#         output_mode="full_history"
#     )
#     logger.info("Supervisor agent created successfully")
# except Exception as e:
#     logger.error(f"Failed to create chat_agent: {str(e)}")
#     raise

# app_ai = chat_agent.compile()
# logger.info("Workflow compiled successfully")

# # API Routes
# @app.route("/signup", methods=["POST"])
# def signup():
#     data = request.get_json()
#     required_fields = ["firstName", "lastName", "dob", "email", "password"]
    
#     if not all(field in data for field in required_fields):
#         logger.warning("Signup failed: Missing required fields")
#         return jsonify({"error": "Missing required fields"}), 400

#     query = "SELECT * FROM c WHERE c.email = @email"
#     parameters = [{"name": "@email", "value": data['email']}]
#     existing_users = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
#     if existing_users:
#         logger.warning(f"Signup failed: Email {data['email']} already exists")
#         return jsonify({"error": "Email already exists"}), 400

#     user_id = str(uuid.uuid4())
#     user_doc = {
#         "id": str(uuid.uuid4()),
#         "user_id": user_id,
#         **data,
#         "messages": [],
#         "context": {},
#         "created_at": datetime.now(timezone.utc).isoformat()
#     }
#     container.create_item(user_doc)
#     logger.info(f"User signed up successfully: {user_id}")
#     return jsonify({"message": "Signup successful", "user_id": user_id}), 201

# @app.route("/login", methods=["POST"])
# def login():
#     data = request.get_json()
#     if not data.get("email") or not data.get("password"):
#         logger.warning("Login failed: Missing email or password")
#         return jsonify({"error": "Missing email or password"}), 400

#     query = "SELECT * FROM c WHERE c.email = @email"
#     parameters = [{"name": "@email", "value": data['email']}]
#     users = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    
#     if not users or users[0]["password"] != data["password"]:
#         logger.warning(f"Login failed for email {data['email']}: Invalid credentials")
#         return jsonify({"error": "Invalid email or password"}), 401

#     logger.info(f"User logged in: {users[0]['user_id']}")
#     return jsonify({"message": "Login successful", "user_id": users[0]["user_id"]}), 200

# @app.route("/get_chat_history", methods=["GET"])
# def get_chat_history():
#     user_id = request.args.get("user_id")
#     if not user_id:
#         logger.warning("Missing user_id in get_chat_history request")
#         return jsonify({"error": "Missing user_id"}), 400

#     try:
#         user_data = get_user_data(user_id)
#         messages = user_data.get("messages", [])
#         logger.info(f"Chat history retrieved for user_id={user_id}: {len(messages)} messages")
#         return jsonify({"messages": messages})
#     except Exception as e:
#         logger.exception(f"Error fetching chat history for user_id={user_id}: {str(e)}")
#         return jsonify({"error": "Failed to fetch chat history"}), 500

# @app.route("/chat", methods=["POST"])
# def chat():
#     """Handle user messages, routing to wellness check or supervisor as needed."""
#     data = request.get_json()
#     user_message = data.get("message", "").strip()
#     user_id = data.get("user_id", "")

#     if not user_message or not user_id:
#         logger.warning(f"Chat request failed: Missing user_id='{user_id}' or message='{user_message}'")
#         return jsonify({"error": "Missing user_id or message"}), 400

#     # Fetch user data
#     try:
#         user_data = get_user_data(user_id)
#         messages = user_data.get("messages", [])
#         context = user_data.get("context", {})
#         logger.debug(f"Chat - user_id={user_id}, initial context: {context}")
#         messages.append({
#             "role": "user",
#             "content": user_message,
#             "timestamp": datetime.now(timezone.utc).isoformat()
#         })
#     except Exception as e:
#         logger.error(f"Error fetching user data for user_id={user_id}: {str(e)}")
#         return jsonify({"error": "Failed to fetch user data"}), 500

#     # Wellness check: Structured questions
#     if context.get("wellness_questions") and context.get("current_question_index") is not None:
#         current_question_index = context["current_question_index"]
#         wellness_questions = context["wellness_questions"]
#         wellness_answers = context.get("wellness_answers", [])
#         current_question = wellness_questions[current_question_index]
        
#         logger.debug(f"Chat - processing wellness question {current_question_index + 1}: '{current_question}'")

#         # Validate responses based on question type
#         if current_question_index in [1, 2, 3]:  # Yes/No questions
#             if user_message.lower() not in ["yes", "no"]:
#                 response = "I’d love a simple 'Yes' or 'No' here—whatever feels true for you!"
#                 agent_used = "wellness_check_expert"
#                 logger.debug(f"Chat - invalid yes/no response: '{user_message}'")
#             else:
#                 wellness_answers.append(user_message.lower())
#                 response, agent_used = _process_next_question_or_verdict(
#                     user_id, messages, context, wellness_questions, wellness_answers, current_question_index
#                 )
#         elif current_question_index == 4:  # 1-10 scale question
#             try:
#                 response_value = int(user_message)
#                 if not (1 <= response_value <= 10):
#                     response = "Could you give me a number between 1 and 10? That’ll help me understand how you’re feeling!"
#                     agent_used = "wellness_check_expert"
#                     logger.debug(f"Chat - out-of-range response: '{user_message}'")
#                 else:
#                     wellness_answers.append(str(response_value))
#                     response, agent_used = _process_next_question_or_verdict(
#                         user_id, messages, context, wellness_questions, wellness_answers, current_question_index
#                     )
#             except ValueError:
#                 response = "Just a number between 1 and 10 would be perfect—whenever you’re ready!"
#                 agent_used = "wellness_check_expert"
#                 logger.debug(f"Chat - non-numeric response: '{user_message}'")
#         else:  # Open-ended question (index 0)
#             wellness_answers.append(user_message)
#             response, agent_used = _process_next_question_or_verdict(
#                 user_id, messages, context, wellness_questions, wellness_answers, current_question_index
#             )

#     # Wellness check: Conversation phase
#     elif context.get("wellness_conversation") and context.get("convo_turns", 0) < 3:
#         context["convo_turns"] = context.get("convo_turns", 0) + 1
#         if "wellness_convo_responses" not in context:
#             context["wellness_convo_responses"] = []
#         context["wellness_convo_responses"].append(user_message)
        
#         response_data = wellness_check(context.get("last_patient_mood", ""), context["wellness_convo_responses"])
#         response = response_data["response"]
#         agent_used = response_data["agent_name"]
#         logger.debug(f"Chat - wellness convo turn {context['convo_turns']}: state={response_data['state']}")

#         if response_data["state"] == "questions":
#             context["wellness_questions"] = response_data["questions"]
#             context["wellness_answers"] = context["wellness_convo_responses"]
#             context["current_question_index"] = 0
#             context.pop("wellness_conversation", None)
#             context.pop("convo_turns", None)
#         elif response_data["state"] == "verdict":
#             context["wellness_check_completed"] = True
#             context["last_summary"] = response_data.get("convo_summary", "general emotional distress")
#             context["risk_level"] = response_data.get("risk_level", "low")
#             context.pop("wellness_conversation", None)
#             context.pop("convo_turns", None)
#             context.pop("wellness_convo_responses", None)
#             if context["risk_level"] in ["high", "medium"]:
#                 therapy_data = provide_therapy(context["last_summary"], user_id, messages)
#                 response += "\n\n" + therapy_data["user_response"]
#                 agent_used = therapy_data["agent_name"]
#                 send_therapy_plan_to_therapist(therapy_data["therapy_plan"], user_id)

#     # New message: Check for mental health triggers or route to supervisor
#     else:
#         mental_health_keywords = ["stress", "anxious", "overthinking", "doubt", "sad", "overwhelmed"]
#         last_user_message = user_message.lower()
        
#         if any(keyword in last_user_message for keyword in mental_health_keywords):
#             if context.get("wellness_check_completed") and context.get("risk_level") in ["high", "medium"]:
#                 logger.debug(f"Chat - completed wellness check, risk={context['risk_level']}, routing to therapy")
#                 therapy_data = provide_therapy(context.get("last_summary", user_message), user_id, messages)
#                 response = therapy_data["user_response"]
#                 agent_used = therapy_data["agent_name"]
#                 send_therapy_plan_to_therapist(therapy_data["therapy_plan"], user_id)
#             else:
#                 logger.debug(f"Chat - starting wellness check for user_id={user_id}")
#                 response_data = wellness_check(user_message)
#                 response = response_data["response"]
#                 agent_used = response_data["agent_name"]
#                 context["wellness_conversation"] = True
#                 context["convo_turns"] = 1
#                 context["wellness_convo_responses"] = [user_message]
#                 context["last_patient_mood"] = user_message
#         else:
#             langchain_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages[-5:]]
#             input_data = {
#                 "messages": langchain_messages,
#                 "additional_responses": [],
#                 "user_id": user_id,
#                 "chat_history": messages,
#                 "context": context
#             }
#             response_data = app_ai.invoke(input_data)
#             last_message = response_data["messages"][-1]

#             if isinstance(last_message, dict) and "response" in last_message:
#                 response = last_message["response"]
#                 agent_used = last_message.get("agent_name", "supervisor")
#                 if last_message.get("state") == "conversation":
#                     context["wellness_conversation"] = True
#                     context["convo_turns"] = 1
#                     context["wellness_convo_responses"] = [user_message]
#                     context["last_patient_mood"] = user_message
#                 elif last_message.get("state") == "questions":
#                     context["wellness_questions"] = last_message["questions"]
#                     context["wellness_answers"] = [user_message]
#                     context["current_question_index"] = 0
#             else:
#                 response = last_message.content if hasattr(last_message, "content") else "I’m here to help—what’s on your mind?"
#                 agent_used = getattr(last_message, "name", "supervisor")

#     # Store and return response
#     messages.append({
#         "role": "assistant",
#         "content": response,
#         "agent_name": agent_used,
#         "timestamp": datetime.now(timezone.utc).isoformat()
#     })
#     store_user_data(user_id, messages, context)
#     logger.info(f"Chat - response for user_id={user_id} from {agent_used}: '{response}'")
#     return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})

# def _process_next_question_or_verdict(user_id, messages, context, wellness_questions, wellness_answers, current_question_index):
#     """Helper to process next question or finalize wellness check."""
#     if current_question_index + 1 < len(wellness_questions):
#         context["current_question_index"] = current_question_index + 1
#         next_question = wellness_questions[current_question_index + 1]
#         transition_prompt = f"""
#         The user answered '{wellness_answers[-1]}' to '{wellness_questions[current_question_index]}'. 
#         Craft a warm, empathetic transition to: '{next_question}'. Reflect on their last answer naturally.
#         """
#         response = invoke_azure_openai([{"role": "user", "content": transition_prompt}], "Transition to next question.")
#         agent_used = "wellness_check_expert"
#         logger.debug(f"Chat - moving to question {current_question_index + 2}: '{next_question}'")
#     else:
#         # All questions answered, get verdict
#         full_responses = context.get("wellness_convo_responses", []) + wellness_answers
#         response_data = wellness_check(context.get("last_patient_mood", ""), full_responses)
#         response = response_data["response"]
#         agent_used = response_data["agent_name"]
#         context["wellness_check_completed"] = True
#         context["last_summary"] = response_data.get("convo_summary", "general emotional distress")
#         context["risk_level"] = response_data.get("risk_level", "low")
#         context.pop("wellness_questions", None)
#         context.pop("wellness_answers", None)
#         context.pop("current_question_index", None)
#         context.pop("wellness_conversation", None)
#         context.pop("wellness_convo_responses", None)
#         context.pop("last_patient_mood", None)
        
#         if context["risk_level"] in ["high", "medium"]:
#             therapy_data = provide_therapy(context["last_summary"], user_id, messages)
#             response += "\n\n" + therapy_data["user_response"]
#             agent_used = therapy_data["agent_name"]
#             send_therapy_plan_to_therapist(therapy_data["therapy_plan"], user_id)
#         logger.info(f"Chat - wellness check completed: risk_level={context['risk_level']}")

#     context["wellness_answers"] = wellness_answers
#     return response, agent_used

# @app.route("/welcome", methods=["POST"])
# def welcome():
#     logger.info("Welcome endpoint called")
#     return jsonify({"response": "Welcome to SoulSync! How can I assist you today?"})

# if __name__ == "__main__":
#     logger.info("Starting SoulSync application...")
#     app.run(debug=True, host="127.0.0.1", port=5000)