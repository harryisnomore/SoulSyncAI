from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor.supervisor import create_supervisor
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from datetime import datetime, timezone,timedelta
import uuid
import logging
import smtplib
from email.mime.text import MIMEText
import time
from werkzeug.utils import secure_filename
import random
import json
from cryptography.fernet import Fernet
import base64

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

# Encryption setup
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key())
cipher = Fernet(ENCRYPTION_KEY)

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

# LangMem Memory Store
memory_store = InMemoryStore(
    index={
        "dims": 1536,  # Match the dimension of your Azure embedding model
        "embed": AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("ENDPOINT2"),
            api_key=os.getenv("API_KEY2"),
            api_version="2023-05-15",
            deployment="text-embedding-ada-002"  # Replace with your Azure embedding deployment name
        )
    }
)

def test_cosmos_db_connection():
    """Test the connection to Azure Cosmos DB.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    try:
        databases = list(cosmos_client.list_databases())
        logger.info(f"Cosmos DB connection successful. Found databases: {len(databases)}")
        return True
    except Exception as e:
        logger.error(f"Cosmos DB connection failed: {str(e)}")
        return False

# Initialize Azure OpenAI Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2023-05-15",
    deployment_name="gpt-4o-mini"
)

def invoke_azure_openai(messages, system_prompt, max_retries=5):
    """Invoke the Azure OpenAI model with retry logic.

    Args:
        messages (list): List of message dictionaries with 'role' and 'content'.
        system_prompt (str): System prompt to guide the model's behavior.
        max_retries (int, optional): Maximum retry attempts on failure. Defaults to 5.

    Returns:
        str: The model's response or an error message if retries are exhausted.
    """
    retry_count = 0
    base_delay = 2
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
                return "Sorry, I’m having trouble connecting right now—let’s try again soon."
            delay = base_delay * (2 ** retry_count) + random.uniform(0, 1)
            logger.warning(f"Retrying in {delay:.2f}s due to {str(e)}")
            time.sleep(delay)

def encrypt_message(message: str) -> str:
    """Encrypt a message using Fernet symmetric encryption.

    Args:
        message (str): The plaintext message to encrypt.

    Returns:
        str: Base64-encoded encrypted message.
    """
    return base64.urlsafe_b64encode(cipher.encrypt(message.encode())).decode()

def decrypt_message(encrypted_message: str) -> str:
    """Decrypt a Fernet-encrypted message.

    Args:
        encrypted_message (str): Base64-encoded encrypted message.

    Returns:
        str: Decrypted plaintext message, or empty string if decryption fails.
    """
    try:
        return cipher.decrypt(base64.urlsafe_b64decode(encrypted_message.encode())).decode()
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        return ""

def get_user_data(user_id: str) -> dict:
    """Retrieve or create user data from Cosmos DB.

    Args:
        user_id (str): Unique identifier for the user.

    Returns:
        dict: User data document with decrypted messages and context.

    Raises:
        Exception: If querying or creating the user document fails.
    """
    try:
        query = "SELECT * FROM c WHERE c.user_id = @user_id"
        parameters = [{"name": "@user_id", "value": user_id}]
        user_docs = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        if not user_docs:
            user_doc = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "messages": [],
                "context": {},
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            container.create_item(user_doc)
            logger.info(f"get_user_data: Created new user_doc for user_id={user_id}")
            return user_doc
        user_doc = user_docs[0]
        logger.debug(f"get_user_data: Retrieved user_doc for user_id={user_id}: {user_doc}")

        # Handle case where messages/context might be encrypted strings from a previous bug
        if isinstance(user_doc.get("messages"), str):
            try:
                decrypted_messages = decrypt_message(user_doc["messages"])
                user_doc["messages"] = json.loads(decrypted_messages) if decrypted_messages else []
            except Exception as e:
                logger.warning(f"Failed to decrypt messages for user_id={user_id}, resetting to empty list: {str(e)}")
                user_doc["messages"] = []
        if isinstance(user_doc.get("context"), str):
            try:
                decrypted_context = decrypt_message(user_doc["context"])
                user_doc["context"] = json.loads(decrypted_context) if decrypted_context else {}
            except Exception as e:
                logger.warning(f"Failed to decrypt context for user_id={user_id}, resetting to empty dict: {str(e)}")
                user_doc["context"] = {}

        return user_doc
    except Exception as e:
        logger.error(f"get_user_data: Error for user_id={user_id}: {str(e)}")
        raise

def get_user_data(user_id: str) -> dict:
    """Retrieve or create user data from Cosmos DB.

    Args:
        user_id (str): Unique identifier for the user.

    Returns:
        dict: User data document with decrypted messages and context.

    Raises:
        Exception: If querying or creating the user document fails.
    """
    try:
        query = "SELECT * FROM c WHERE c.user_id = @user_id"
        parameters = [{"name": "@user_id", "value": user_id}]
        user_docs = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        if not user_docs:
            user_doc = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "messages": [],
                "context": {},
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            container.create_item(user_doc)
            logger.info(f"get_user_data: Created new user_doc for user_id={user_id}")
            return user_doc
        user_doc = user_docs[0]
        logger.debug(f"get_user_data: Retrieved user_doc for user_id={user_id}: {user_doc}")

        # Handle case where messages/context might be encrypted strings from a previous bug
        if isinstance(user_doc.get("messages"), str):
            try:
                decrypted_messages = decrypt_message(user_doc["messages"])
                user_doc["messages"] = json.loads(decrypted_messages) if decrypted_messages else []
            except Exception as e:
                logger.warning(f"Failed to decrypt messages for user_id={user_id}, resetting to empty list: {str(e)}")
                user_doc["messages"] = []
        if isinstance(user_doc.get("context"), str):
            try:
                decrypted_context = decrypt_message(user_doc["context"])
                user_doc["context"] = json.loads(decrypted_context) if decrypted_context else {}
            except Exception as e:
                logger.warning(f"Failed to decrypt context for user_id={user_id}, resetting to empty dict: {str(e)}")
                user_doc["context"] = {}

        return user_doc
    except Exception as e:
        logger.error(f"get_user_data: Error for user_id={user_id}: {str(e)}")
        raise

def store_user_data(user_id: str, messages: list, context: dict) -> None:
    """Store or update user data in Cosmos DB.

    Args:
        user_id (str): Unique identifier for the user.
        messages (list): List of message dictionaries with encrypted content.
        context (dict): Contextual data for the user’s session.

    Raises:
        Exception: If upserting the user document fails.
    """
    try:
        user_data = get_user_data(user_id)
        user_data["messages"] = messages  # Store as list of dicts, content already encrypted
        user_data["context"] = context    # Store as dict, no encryption needed here
        user_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        container.upsert_item(user_data)
        logger.debug(f"store_user_data: Stored user data for user_id={user_id}, context={context}")
    except Exception as e:
        logger.error(f"store_user_data: Error for user_id={user_id}: {str(e)}")
        raise

def generate_otp():
    """Generate a 6-digit one-time password (OTP).

    Returns:
        str: A random 6-digit OTP.
    """
    return str(random.randint(100000, 999999))

def send_otp_email(email: str, otp: str):
    """Send an OTP to the user’s email via SMTP.

    Args:
        email (str): Recipient email address.
        otp (str): One-time password to send.

    Returns:
        bool: True if email sent successfully, False otherwise.
    """
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([smtp_server, smtp_port, smtp_user, smtp_password]):
        logger.error("Missing SMTP configuration for OTP")
        return False

    subject = "Your SoulSync OTP Verification Code"
    body = f"Hello,\n\nYour OTP for SoulSync signup is: {otp}\n\nThis code is valid for 10 minutes.\n\nBest,\nSoulSync Team"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = email

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        logger.info(f"OTP {otp} sent to {email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send OTP to {email}: {str(e)}")
        return False

def get_summary(user_input: str) -> str:
    """Generate a concise summary of the user’s emotional state.

    Args:
        user_input (str): User’s input text.

    Returns:
        str: Summary of key emotions or 'general emotional distress' if none specified.
    """
    summary_prompt = """
    Summarize the user's input by extracting the key emotions or mental health concerns (e.g., stress, anxiety, overwhelmed, self-doubt) in a concise phrase. Avoid repeating the entire input. If no specific emotions are mentioned, return 'general emotional distress'.
    """
    messages = [{"role": "user", "content": f"Input: {user_input}"}]
    return invoke_azure_openai(messages, summary_prompt)

def wellness_check(patient_mood: str, additional_responses: list = None) -> dict:
    """Perform a mental health wellness check based on user mood.

    Args:
        patient_mood (str): User’s reported mood or emotional state.
        additional_responses (list, optional): List of user responses to wellness questions.

    Returns:
        dict: Response data including encrypted message, state, and agent details.
    """
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
            "response": encrypt_message(response),
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
        "response": encrypt_message(response),
        "state": "verdict",
        "agent_name": "wellness_check_expert",
        "continue_flow": False,
        "risk_level": risk_level
    }

def provide_therapy(condition: str, user_id: str = None, chat_history: list = None) -> dict:
    """Generate a personalized therapy plan for the user.

    Args:
        condition (str): User’s reported emotional or mental health condition.
        user_id (str, optional): Unique identifier for the user.
        chat_history (list, optional): List of prior encrypted chat messages.

    Returns:
        dict: Therapy plan and user response, with encrypted user response.
    """
    user_data = get_user_data(user_id)
    user_name = f"{user_data.get('firstName', '')} {user_data.get('lastName', '')}".strip() or "Unknown User"

    chat_summary = "No prior conversation provided."
    if chat_history:
        decrypted_history = [f"{msg['role'].capitalize()}: {decrypt_message(msg['content'])}" for msg in chat_history[-6:]]
        chat_summary = "Full Chat History:\n" + "\n".join(decrypted_history)

    refine_prompt = """
    Based on the user's condition and chat history, refine the condition description to capture their emotional state and needs in a warm, natural way. Return a concise description.
    """
    messages = [{"role": "user", "content": f"Condition: {condition}\nChat History: {chat_summary}"}]
    refined_condition = invoke_azure_openai(messages, refine_prompt)

    insights_prompt = f"""
    Based on the chat history:
    {chat_summary}
    Provide a concise summary of insights into the user’s emotional state, including key emotions, triggers, and impacts on daily life. Keep it professional and clear for a therapist.
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
    Based on the therapy plan: '{therapy_response}', craft a short, warm message to the user. Mention the therapy type, one focus area, one recommendation, and that a therapist will reach out. Keep it encouraging and casual, explaining why therapy is needed.
    """
    user_response = invoke_azure_openai([{"role": "user", "content": user_response_prompt}], "Generate a user-friendly response.")

    return {
        "user_response": encrypt_message(user_response),
        "therapy_plan": therapy_plan_text,
        "agent_name": "therapy_expert"
    }

def post_rehab_followup(patient_status: str, user_id: str = None, chat_history: list = None) -> dict:
    """Assess and respond to a user’s post-rehabilitation status.

    Args:
        patient_status (str): User’s reported status post-rehab.
        user_id (str, optional): Unique identifier for the user.
        chat_history (list, optional): List of prior encrypted chat messages.

    Returns:
        dict: Response data including encrypted message, risk level, and agent details.
    """
    summary = get_summary(patient_status)

    chat_summary = "No prior conversation provided."
    if chat_history:
        decrypted_history = [f"{msg['role'].capitalize()}: {decrypt_message(msg['content'])}" for msg in chat_history[-6:]]
        chat_summary = "Full Chat History:\n" + "\n".join(decrypted_history)

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
    
    analysis = {}
    for line in analysis_response.split("\n"):
        if ": " in line:
            key, value = line.split(": ", 1)
            key = key.strip("- ").lower().replace(" ", "_")
            if key == "risk_factors":
                value = [factor.strip() for factor in value.strip("[]").split(", ") if factor.strip()]
            analysis[key] = value

    risk_prompt = f"""
    Based on the following analysis of the user's post-rehab status:
    - Emotional State: {analysis.get('emotional_state', 'unknown')}
    - Recovery Progress: {analysis.get('recovery_progress', 'unknown')}
    - Support System: {analysis.get('support_system', 'unclear')}
    - Therapy Feedback: {analysis.get('therapy_feedback', 'unclear')}
    - Risk Factors: {', '.join(analysis.get('risk_factors', []))}
    Assess the risk level as 'low', 'medium', or 'high'.
    Return only the risk level.
    """
    risk_level = invoke_azure_openai([{"role": "user", "content": risk_prompt}], "Assess risk level.").lower()

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
    - Comment on their therapy feedback.
    - Provide solutions based on their risk level and analysis.
    Keep the tone natural, warm, and non-judgmental.
    """
    response = invoke_azure_openai([{"role": "user", "content": solutions_prompt}], "Generate a response with tailored solutions.")

    return_data = {
        "response": encrypt_message(response),
        "state": "verdict",
        "agent_name": "post_rehab_expert",
        "continue_flow": False,
        "risk_level": risk_level
    }

    if risk_level == "high":
        therapy_data = provide_therapy(summary, user_id, chat_history)
        decrypted_response = decrypt_message(return_data["response"])
        decrypted_therapy_response = decrypt_message(therapy_data["user_response"])
        response = encrypt_message(f"{decrypted_response}\n\nGiven the challenges you’re facing, I’ve escalated your case to our therapy expert. {decrypted_therapy_response}")
        return_data["response"] = response
        return_data["agent_name"] = therapy_data["agent_name"]
        send_therapy_plan_to_therapist(therapy_data["therapy_plan"], user_id)

    return return_data

def send_therapy_plan_to_therapist(therapy_plan: str, user_id: str) -> bool:
    """Email a therapy plan to a designated therapist.

    Args:
        therapy_plan (str): Formatted therapy plan text.
        user_id (str): Unique identifier for the user.

    Returns:
        bool: True if email sent successfully, False otherwise.
    """
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
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        logger.info(f"Therapy plan emailed to {therapist_email} for user_id={user_id}")
        return True
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error for user_id={user_id}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending email for user_id={user_id}: {str(e)}")
        return False

# Agents with LangMem Integration
logger.info("Creating memory tools...")
memory_manage_tool = create_manage_memory_tool(namespace=("memories",))
memory_search_tool = create_search_memory_tool(namespace=("memories",))

logger.info("Creating agents...")
wellness_check_agent = create_react_agent(
    model=model,
    tools=[wellness_check, memory_manage_tool, memory_search_tool],
    name="wellness_check_expert",
    prompt="""
    You’re a compassionate mental health companion with long-term memory. Use memory tools to store and retrieve user preferences or past emotional states:
    1. Pass the user’s message to 'wellness_check' as 'patient_mood'.
    2. If 'additional_responses' are provided, include them for a risk verdict.
    3. Use 'create_manage_memory_tool' to store key insights (e.g., "User feels stressed often").
    4. Use 'create_search_memory_tool' to check past interactions if relevant.
    Return the 'response' field as-is from wellness_check.
    """,
    store=memory_store
)

therapy_agent = create_react_agent(
    model=model,
    tools=[provide_therapy, send_therapy_plan_to_therapist, memory_manage_tool, memory_search_tool],
    name="therapy_expert",
    prompt="""
    You’re a friendly therapy guide with memory. Use memory tools to personalize therapy:
    1. Use 'provide_therapy' with the message as 'condition', plus 'user_id' and 'chat_history'.
    2. Pass 'therapy_plan' and 'user_id' to 'send_therapy_plan_to_therapist'.
    3. Use 'create_manage_memory_tool' to store therapy preferences or triggers.
    4. Use 'create_search_memory_tool' to recall past therapy needs.
    Return the 'user_response' from provide_therapy.
    """,
    store=memory_store
)

post_rehab_agent = create_react_agent(
    model=model,
    tools=[post_rehab_followup, memory_manage_tool, memory_search_tool],
    name="post_rehab_expert",
    prompt="""
    You’re a supportive post-rehab ally with memory. Use memory tools to track recovery:
    1. Pass the message to 'post_rehab_followup' as 'patient_status', with 'user_id' and 'chat_history'.
    2. Use 'create_manage_memory_tool' to store recovery milestones or struggles.
    3. Use 'create_search_memory_tool' to reference past rehab progress.
    Return the 'response' field from post_rehab_followup.
    """,
    store=memory_store
)

logger.info("Creating supervisor agent...")
try:
    chat_agent = create_supervisor(
        agents=[wellness_check_agent, therapy_agent, post_rehab_agent],
        model=model,
        prompt=
        """You are the Supervisor Agent of SoulSync AI, responsible for guiding users in mental health, therapy, and post-rehab support. 

**Your Job:**
1. **Identify the Patient:** The patient's name is provided in the context as `patient_name` (e.g., "Harshit   "). Use this name in all greetings and responses to personalize the interaction.
2. **Retrieve PDF Context:** The patient's health and emotional context is provided in `context["pdf_context"]`. Use this context to tailor your responses, mentioning it naturally when relevant.
3. **Personalized Greeting:** Always greet the patient by their name in the first message of an interaction (e.g., "Hello Harshit   !"). Incorporate the PDF context if relevant (e.g., "I see from your recent report that your health state is Stable.").
4. **Recall Memory:** Use LangChain's memory tools to recall previous interactions and emotional states.
5. **Decide Which Expert:** Determine which expert should handle the request (options: `wellness_check_expert`, `therapy_expert`, `post_rehab_expert`).
6. **Generate or Retrieve Response:** Provide a warm, empathetic response based on the patient's history, current query, and PDF context.

**Rules:**
- Always start the interaction by greeting the patient with their name (e.g., "Hello Harshit   ! I'm here to assist you.").
- Use the PDF context (`context["pdf_context"]`) to inform your responses. Mention the `health_state` or `emotional_state` if relevant to the conversation, but only if they are not "Unknown".
- If the query is therapy-related (e.g., contains "therapy", "therapist", "counseling"), forward it to `therapy_expert`.
- If it's about emotional wellness (e.g., contains "stress", "anxious", "sad"), use past memory and PDF context to assess risk levels and forward to `wellness_check_expert`.
- If it's post-rehab related (e.g., contains "rehab", "recovery", "relapse"), act as `post_rehab_expert` and guide them, using the PDF context if relevant.
- Use the patient's name naturally in responses to build rapport (e.g., "Harshit   , I see you've been feeling stressed lately, and your recent report mentions an Anxious emotional state. Let’s work on that.").

**How to Respond:**
1. **Greeting:** Welcome the patient by their name and incorporate PDF context if relevant (e.g., "Hello Harshit   ! I see from your recent report that your health state is Stable. How are you feeling today?").
2. **Off-Topic:** Gently redirect to mental health topics using their name and PDF context if applicable (e.g., "Harshit   , I’d love to help with your mental health needs. Your recent report mentions an Anxious emotional state—would you like to talk about that?").
3. **Expert Transfer:** If transferring to an expert, inform the patient using their name (e.g., "Harshit   , I’m transferring you to our therapy expert. Let’s get started.").
4. **Concise & Clear:** Provide direct, supportive replies without long explanations.
5. **One Thought Per Message:** Avoid overloading users with too much info at once.
6. **Soft & Encouraging:** Maintain a warm, empathetic tone (e.g., "Harshit   , I’m here for you. Let’s take this one step at a time.").
7. **Acknowledge Context:** Reference past user context and PDF context when responding (e.g., "Harshit   , I remember you mentioned feeling anxious last week, and your report also notes an Anxious emotional state. How are you feeling now?").
8. **Clarify if Unsure:** Ask for more details if the query is unclear, using their name and PDF context if relevant (e.g., "Harshit   , can you tell me more about how you’re feeling? Your recent report mentions a Stable health state.").
9. **Memory Tools:** Use `create_manage_memory_tool` to store key insights (e.g., "User feels stressed often") and `create_search_memory_tool` to check past interactions if relevant.

**Example Responses:**
- Greeting: "Hello Harshit ! I see from your recent report that your health state is Stable and your emotional state is Anxious. How are you feeling today?"
- Redirecting: "Harshit , I’d love to help with your mental health needs. Your recent report mentions an Anxious emotional state—would you like to talk about how you’re feeling?"
- Using Context: "Harshit , I see you’ve been feeling stressed lately, and your recent report also notes an Anxious emotional state. Let’s work on that together. How can I assist you today?"
- Clarification: "Harshit , I’m not sure I fully understand—can you tell me more about what’s been going on? Your report mentions a Stable health state, which is a good sign."
""",
        supervisor_name="supervisor",
        include_agent_name="inline",
        output_mode="full_history",
    )
    logger.info("Supervisor agent created successfully")
except Exception as e:
    logger.error(f"Failed to create chat_agent: {str(e)}")
    raise

app_ai = chat_agent.compile()
logger.info("Workflow compiled successfully")

@app.route("/signup", methods=["POST"])
def signup():
    """Handle user signup by generating and sending an OTP.

    Returns:
        Response: JSON response with user_id and status message, or error.
    """
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

    otp = generate_otp()
    if not send_otp_email(data['email'], otp):
        return jsonify({"error": "Failed to send OTP"}), 500

    user_id = str(uuid.uuid4())
    temp_user_doc = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        **data,
        "otp": otp,
        "otp_expiry": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        "verified": False,
        "messages": [],
        "context": {},
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    container.create_item(temp_user_doc)
    logger.info(f"Temporary user created with OTP for user_id={user_id}")
    return jsonify({"message": "OTP sent to your email", "user_id": user_id}), 200

@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    """Verify user OTP and complete signup.

    Returns:
        Response: JSON response with status message or error.
    """
    data = request.get_json()
    user_id = data.get("user_id")
    otp = data.get("otp")

    if not user_id or not otp:
        logger.warning("OTP verification failed: Missing user_id or OTP")
        return jsonify({"error": "Missing user_id or OTP"}), 400

    try:
        user_data = get_user_data(user_id)
        if user_data.get("verified"):
            return jsonify({"error": "User already verified"}), 400

        if user_data.get("otp") != otp:
            logger.warning(f"OTP verification failed for user_id={user_id}: Invalid OTP")
            return jsonify({"error": "Invalid OTP"}), 401

        if datetime.fromisoformat(user_data["otp_expiry"]) < datetime.now(timezone.utc):
            logger.warning(f"OTP verification failed for user_id={user_id}: OTP expired")
            return jsonify({"error": "OTP expired"}), 401

        user_data["verified"] = True
        user_data.pop("otp")
        user_data.pop("otp_expiry")
        container.upsert_item(user_data)
        logger.info(f"User verified successfully: {user_id}")
        return jsonify({"message": "Signup successful", "user_id": user_id}), 201
    except Exception as e:
        logger.error(f"Error verifying OTP for user_id={user_id}: {str(e)}")
        return jsonify({"error": "Failed to verify OTP"}), 500

@app.route("/login", methods=["POST"])
def login():
    """Authenticate user login with email and password.

    Returns:
        Response: JSON response with user_id and status message, or error.
    """
    data = request.get_json()
    if not data.get("email") or not data.get("password"):
        logger.warning("Login failed: Missing email or password")
        return jsonify({"error": "Missing email or password"}), 400

    query = "SELECT * FROM c WHERE c.email = @email"
    parameters = [{"name": "@email", "value": data['email']}]
    users = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    
    if not users or users[0]["password"] != data["password"] or not users[0].get("verified"):
        logger.warning(f"Login failed for email {data['email']}: Invalid credentials or unverified")
        return jsonify({"error": "Invalid email, password, or account not verified"}), 401

    logger.info(f"User logged in: {users[0]['user_id']}")
    return jsonify({"message": "Login successful", "user_id": users[0]["user_id"]}), 200

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history():
    """Retrieve decrypted chat history for a user.

    Returns:
        Response: JSON response with decrypted messages or error.
    """
    user_id = request.args.get("user_id")
    if not user_id:
        logger.warning("Missing user_id in get_chat_history request")
        return jsonify({"error": "Missing user_id"}), 400

    try:
        user_data = get_user_data(user_id)
        messages = [
            {"role": msg["role"], "content": decrypt_message(msg["content"]), "timestamp": msg["timestamp"]}
            for msg in user_data.get("messages", [])
        ]
        logger.info(f"Chat history retrieved for user_id={user_id}: {len(messages)} messages")
        return jsonify({"messages": messages})
    except Exception as e:
        logger.exception(f"Error fetching chat history for user_id={user_id}: {str(e)}")
        return jsonify({"error": "Failed to fetch chat history"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Handle user chat interactions with AI agents, ensuring the patient's name is used in greetings.

    Returns:
        Response: JSON response with decrypted AI response, user_id, and agent used, or error.
    """
    data = request.get_json()
    user_message = data.get("message", "").strip()
    user_id = data.get("user_id", "")

    if not user_message or not user_id:
        logger.warning("Chat request failed: Missing user_id or message")
        return jsonify({"error": "Missing user_id or message"}), 400

    try:
        user_data = get_user_data(user_id)
        patient_name = f"{user_data.get('firstName', '')}".strip() or "User"
        messages = user_data.get("messages", [])
        context = user_data.get("context", {})
        context["patient_name"] = patient_name  # Add patient name to context
        logger.debug(f"Initial context for user_id={user_id}: {context}")

        encrypted_user_message = encrypt_message(user_message)
        messages.append({
            "role": "user",
            "content": encrypted_user_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching user data for user_id={user_id}: {str(e)}")
        return jsonify({"error": f"Failed to fetch user data: {str(e)}"}), 500

    try:
        decrypted_messages = [
            {"role": msg["role"], "content": decrypt_message(msg["content"])}
            for msg in messages
        ]

        if context.get("wellness_questions") and context.get("current_question_index") is not None:
            current_question_index = context["current_question_index"]
            wellness_answers = context.get("wellness_answers", [])
            wellness_questions = context["wellness_questions"]

            if current_question_index in [1, 2, 3]:
                if user_message.lower() not in ["yes", "no"]:
                    response = encrypt_message("I’d love a simple 'Yes' or 'No' here—whatever feels true for you!")
                    agent_used = "wellness_check_expert"
                    messages.append({"role": "assistant", "content": response, "agent_name": agent_used, "timestamp": datetime.now(timezone.utc).isoformat()})
                    store_user_data(user_id, messages, context)
                    return jsonify({"user_id": user_id, "response": decrypt_message(response), "agent_used": agent_used})
            elif current_question_index == 4:
                try:
                    response_value = int(user_message)
                    if not (1 <= response_value <= 10):
                        response = encrypt_message("Could you give me a number between 1 and 10? That’ll help me understand how you’re feeling!")
                        agent_used = "wellness_check_expert"
                        messages.append({"role": "assistant", "content": response, "agent_name": agent_used, "timestamp": datetime.now(timezone.utc).isoformat()})
                        store_user_data(user_id, messages, context)
                        return jsonify({"user_id": user_id, "response": decrypt_message(response), "agent_used": agent_used})
                except ValueError:
                    response = encrypt_message("Just a number between 1 and 10 would be perfect—whenever you’re ready!")
                    agent_used = "wellness_check_expert"
                    messages.append({"role": "assistant", "content": response, "agent_name": agent_used, "timestamp": datetime.now(timezone.utc).isoformat()})
                    store_user_data(user_id, messages, context)
                    return jsonify({"user_id": user_id, "response": decrypt_message(response), "agent_used": agent_used})

            wellness_answers.append(user_message)
            context["wellness_answers"] = wellness_answers
            logger.debug(f"Stored user response for question {current_question_index}: {user_message}")

            if current_question_index + 1 < len(wellness_questions):
                context["current_question_index"] = current_question_index + 1
                next_question = wellness_questions[current_question_index + 1]
                transition_prompt = f"""
                The user answered '{user_message}' to '{wellness_questions[current_question_index]}'. Craft a warm, empathetic transition to the next question: '{next_question}'. Keep it natural and supportive, reflecting on their last answer.
                """
                response = encrypt_message(invoke_azure_openai([{"role": "user", "content": transition_prompt}], "Generate a transition."))
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
                    decrypted_response = decrypt_message(response)
                    response = encrypt_message(f"{decrypted_response}\n\n{decrypt_message(therapy_data['user_response'])}")
                    agent_used = therapy_data["agent_name"]
                    send_therapy_plan_to_therapist(therapy_data["therapy_plan"], user_id)
        else:
            last_user_message = user_message.lower()
            mental_health_keywords = ["stress", "anxious", "overthinking", "doubt", "sad", "overwhelmed"]
            post_rehab_keywords = ["rehab", "recovery", "relapse", "struggling since rehab", "after rehab"]

            if any(keyword in last_user_message for keyword in mental_health_keywords):
                if context.get("wellness_check_completed") and context.get("risk_level") in ["high", "medium"]:
                    therapy_data = provide_therapy(context.get("last_summary", user_message), user_id, messages)
                    response = therapy_data["user_response"]
                    agent_used = therapy_data["agent_name"]
                    send_therapy_plan_to_therapist(therapy_data["therapy_plan"], user_id)
                else:
                    response_data = wellness_check(user_message)
                    response = response_data["response"]
                    agent_used = response_data["agent_name"]
                    if response_data.get("state") == "questions" and "questions" in response_data:
                        context["wellness_questions"] = response_data["questions"]
                        context["wellness_answers"] = []
                        context["current_question_index"] = 0
                        context["last_patient_mood"] = user_message
                        logger.debug(f"Started wellness check flow with context: {context}")
            elif any(keyword in last_user_message for keyword in post_rehab_keywords):
                if context.get("post_rehab_check_completed") and context.get("post_rehab_risk_level") in ["high", "medium"]:
                    therapy_data = provide_therapy(context.get("last_post_rehab_summary", user_message), user_id, messages)
                    response = therapy_data["user_response"]
                    agent_used = therapy_data["agent_name"]
                    send_therapy_plan_to_therapist(therapy_data["therapy_plan"], user_id)
                else:
                    response_data = post_rehab_followup(user_message, user_id, messages)
                    response = response_data["response"]
                    agent_used = response_data["agent_name"]
                    context["post_rehab_check_completed"] = True
                    context["last_post_rehab_summary"] = get_summary(user_message)
                    context["post_rehab_risk_level"] = response_data.get("risk_level", "low")
                    logger.debug(f"Completed post-rehab follow-up with context: {context}")
            else:
                input_data = {
                    "messages": decrypted_messages[-5:],
                    "additional_responses": [],
                    "user_id": user_id,
                    "chat_history": messages,
                    "context": context
                }

                response_data = app_ai.invoke(input_data)
                last_message = response_data["messages"][-1]

                if isinstance(last_message, dict) and "response" in last_message:
                    response = encrypt_message(last_message["response"])
                    agent_used = last_message.get("agent_name", "supervisor")
                    if last_message.get("state") == "questions" and "questions" in last_message:
                        context["wellness_questions"] = last_message["questions"]
                        context["wellness_answers"] = []
                        context["current_question_index"] = 0
                        context["last_patient_mood"] = user_message
                        response = encrypt_message(last_message["response"])
                        agent_used = "wellness_check_expert"
                elif hasattr(last_message, "content") and last_message.content:
                    if user_message.lower() in ["hello", "hi", "hey", "hii"]:
                        greeting_prompt = f"""
                        The user said '{user_message}'. The patient's name is '{patient_name}'. Generate a warm, inviting response encouraging them to share their feelings or concerns about mental health, starting with a greeting using their name (e.g., "Hello {patient_name}!"). Chat history: {decrypted_messages}.
                        """
                        response = encrypt_message(invoke_azure_openai([{"role": "user", "content": greeting_prompt}], "Generate a greeting."))
                        agent_used = "supervisor"
                    else:
                        response = encrypt_message(last_message.content)
                        agent_used = getattr(last_message, "name", "supervisor")
                else:
                    response = encrypt_message(f"{patient_name}, I’m having a little trouble here—could you tell me more about what’s on your mind?")
                    agent_used = "supervisor"

        messages.append({
            "role": "assistant",
            "content": response,
            "agent_name": agent_used,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        store_user_data(user_id, messages, context)
        logger.info(f"Final response for user_id={user_id} from {agent_used}")
        return jsonify({"user_id": user_id, "response": decrypt_message(response), "agent_used": agent_used})

    except Exception as e:
        logger.error(f"Chat processing failed for user_id={user_id}: {str(e)}")
        return jsonify({"error": f"Chat processing failed: {str(e)}"}), 500

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    """Upload a PDF for a user and store it in Cosmos DB.

    Returns:
        Response: JSON response with status message or error.
    """
    user_id = request.form.get("user_id")
    if not user_id:
        logger.warning("Upload PDF failed: Missing user_id")
        return jsonify({"error": "Missing user_id"}), 400

    if "pdf_file" not in request.files:
        logger.warning("Upload PDF failed: No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files["pdf_file"]
    if file.filename == "":
        logger.warning("Upload PDF failed: No selected file")
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".pdf"):
        try:
            # Read the PDF file as binary and encode it as base64
            pdf_binary = file.read()
            pdf_base64 = base64.b64encode(pdf_binary).decode("utf-8")

            # Fetch the user document from Cosmos DB
            user_data = get_user_data(user_id)
            if not user_data:
                logger.warning(f"Upload PDF failed: User not found for user_id={user_id}")
                return jsonify({"error": "User not found"}), 404

            # Store the PDF data in the user document
            user_data["pdf_data"] = pdf_base64
            user_data["pdf_filename"] = secure_filename(file.filename)
            user_data["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Update the user document in Cosmos DB
            container.upsert_item(user_data)
            logger.info(f"PDF uploaded and stored for user_id={user_id}, filename={file.filename}")
            return jsonify({"message": f"PDF {file.filename} uploaded successfully for user {user_id}"}), 200
        except Exception as e:
            logger.error(f"Error uploading PDF for user_id={user_id}: {str(e)}")
            return jsonify({"error": f"Failed to upload PDF: {str(e)}"}), 500
    else:
        logger.warning("Upload PDF failed: Invalid file format")
        return jsonify({"error": "Invalid file format, only PDFs are allowed"}), 400

@app.route("/welcome", methods=["POST"])
def welcome():
    """Provide a welcome message to new users, personalized with their name.

    Returns:
        Response: JSON response with decrypted welcome message.
    """
    logger.info("Welcome endpoint called")
    data = request.get_json()
    user_id = data.get("user_id", "")
    if not user_id:
        logger.warning("Welcome request failed: Missing user_id")
        return jsonify({"error": "Missing user_id"}), 400

    try:
        response = encrypt_message(f"Welcome to SoulSync! Your mental health companion")
        logger.info(f"Welcome message sent for user_id={user_id}")
        return jsonify({"response": decrypt_message(response)}), 200
    except Exception as e:
        logger.error(f"Error in welcome endpoint for user_id={user_id}: {str(e)}")
        return jsonify({"error": "Failed to generate welcome message"}), 500

if __name__ == "__main__":
    logger.info("Starting SoulSync application...")
    if not test_cosmos_db_connection():
        logger.critical("Exiting due to Cosmos DB connection failure")
        exit(1)
    app.run(debug=True, host="127.0.0.1", port=5000)