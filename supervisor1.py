from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor.supervisor import create_supervisor
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from datetime import datetime, timezone, timedelta
import uuid
import logging
import smtplib
from email.mime.text import MIMEText
import time
import random
import string
from cryptography.fernet import Fernet

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

if not all([COSMOS_DB_ENDPOINT, COSMOS_DB_KEY, COSMOS_DB_NAME, COSMOS_DB_CONTAINER,ENCRYPTION_KEY]):
    raise ValueError("Missing required Cosmos DB environment variables")

cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
database = cosmos_client.get_database_client(COSMOS_DB_NAME)
container = database.get_container_client(COSMOS_DB_CONTAINER)

# Initialize Encryption
fernet = Fernet(ENCRYPTION_KEY.encode())

# Initialize Azure OpenAI Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2023-05-15",
    deployment_name="gpt-4o-mini"
)

# Encryption/Decryption Helpers
def encrypt_data(data):
    if isinstance(data, dict) or isinstance(data, list):
        data = str(data)  # Convert complex objects to string
    return fernet.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data):
    try:
        return fernet.decrypt(encrypted_data.encode()).decode()
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        raise

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

# OTP Helper Functions
def generate_otp(length=6):
    """Generate a random 6-digit OTP."""
    return ''.join(random.choices(string.digits, k=length))

def send_otp_email(email: str, otp: str) -> bool:
    """Send an OTP to the user's email."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([smtp_server, smtp_port, smtp_user, smtp_password]):
        logger.error("Missing SMTP configuration for sending OTP")
        return False

    subject = "SoulSync OTP Verification"
    body = f"""
    Dear User,

    Your One-Time Password (OTP) for SoulSync verification is: {otp}

    This OTP is valid for 5 minutes. Please do not share it with anyone.

    Best regards,
    SoulSync Team
    """
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        logger.info(f"OTP sent to {email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send OTP to {email}: {str(e)}")
        return False

def store_otp(email: str, otp: str) -> bool:
    try:
        encrypted_email = encrypt_data(email)
        logger.debug(f"Encrypted email in store_otp for {email}: {encrypted_email}")
        otp_doc = {
            "id": f"otp_{email}",  # Use plain email for the id
            "email": encrypted_email,
            "otp": otp,
            "patientId": encrypted_email,  # Partition key
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        }
        logger.debug(f"Attempting to upsert OTP document: {otp_doc}")
        
        container.upsert_item(otp_doc)
        
        query = "SELECT * FROM c WHERE c.id = @id"
        parameters = [{"name": "@id", "value": f"otp_{email}"}]
        updated_docs = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        
        if not updated_docs:
            logger.error(f"Failed to verify OTP update for {email}: Document not found after upsert")
            return False
        
        updated_doc = updated_docs[0]
        if updated_doc["otp"] != otp:
            logger.error(f"OTP update failed for {email}: Stored OTP ({updated_doc['otp']}) does not match new OTP ({otp})")
            return False
        
        logger.debug(f"Successfully stored OTP {otp} for {email}")
        return True
    except Exception as e:
        logger.error(f"Failed to store OTP for {email}: {str(e)}")
        return False

def validate_otp(email: str, otp: str) -> bool:
    try:
        encrypted_email = encrypt_data(email)
        logger.debug(f"Encrypted email in validate_otp for {email}: {encrypted_email}")
        query = "SELECT * FROM c WHERE c.id = @id"
        parameters = [{"name": "@id", "value": f"otp_{email}"}]  # Use plain email for the id
        otp_docs = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        if not otp_docs:
            logger.warning(f"No OTP found for {email}")
            return False

        otp_doc = otp_docs[0]
        stored_otp = otp_doc.get("otp")
        expires_at = datetime.fromisoformat(otp_doc.get("expires_at").replace("Z", "+00:00"))
        current_time = datetime.now(timezone.utc)

        logger.debug(f"Validating OTP for {email}: Entered={otp}, Stored={stored_otp}, ExpiresAt={expires_at}, CurrentTime={current_time}")

        if current_time > expires_at:
            logger.warning(f"OTP expired for {email}")
            container.delete_item(item=otp_doc["id"], partition_key=otp_doc["patientId"])
            return False

        if stored_otp != otp:
            logger.warning(f"Invalid OTP for {email}: Entered={otp}, Stored={stored_otp}")
            return False

        container.delete_item(item=otp_doc["id"], partition_key=otp_doc["patientId"])
        logger.info(f"OTP validated and deleted for {email}")
        return True
    except Exception as e:
        logger.error(f"Error validating OTP for {email}: {str(e)}")
        return False

# Modified Cosmos DB Helper Functions
def get_user_data(user_id: str) -> dict:
    try:
        query = "SELECT * FROM c WHERE c.user_id = @user_id"
        parameters = [{"name": "@user_id", "value": user_id}]
        user_docs = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        if not user_docs:
            user_doc = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "messages": encrypt_data("[]"),
                "context": encrypt_data("{}"),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            container.create_item(user_doc)
            logger.info(f"get_user_data: Created new user_doc for user_id={user_id}")
            return {
                "id": user_doc["id"],
                "user_id": user_id,
                "messages": [],
                "context": {},
                "created_at": user_doc["created_at"]
            }
        
        user_doc = user_docs[0]
        # Ensure messages and context fields exist, initialize if missing
        if "messages" not in user_doc:
            user_doc["messages"] = encrypt_data("[]")
            logger.warning(f"Messages field missing for user_id={user_id}, initializing to empty list")
        if "context" not in user_doc:
            user_doc["context"] = encrypt_data("{}")
            logger.warning(f"Context field missing for user_id={user_id}, initializing to empty dict")

        decrypted_doc = {
            "id": user_doc["id"],
            "user_id": user_doc["user_id"],
            "messages": eval(decrypt_data(user_doc["messages"])),
            "context": eval(decrypt_data(user_doc["context"])),
            "created_at": user_doc["created_at"]
        }
        for field in ["email", "firstName", "lastName", "dob", "password"]:
            if field in user_doc:
                decrypted_doc[field] = decrypt_data(user_doc[field])
        if "updated_at" in user_doc:
            decrypted_doc["updated_at"] = user_doc["updated_at"]
        logger.debug(f"get_user_data: Retrieved user_doc for user_id={user_id}")
        return decrypted_doc
    except Exception as e:
        logger.error(f"get_user_data: Error for user_id={user_id}: {str(e)}")
        raise

def store_user_data(user_id: str, messages: list, context: dict) -> None:
    try:
        user_data = get_user_data(user_id)
        user_data["messages"] = messages
        user_data["context"] = context
        user_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        encrypted_data = {
            "id": user_data["id"],
            "user_id": user_id,
            "messages": encrypt_data(str(messages)),
            "context": encrypt_data(str(context)),
            "created_at": user_data["created_at"],
            "updated_at": user_data["updated_at"]
        }
        # Encrypt user credentials if present
        for field in ["email", "firstName", "lastName", "dob", "password"]:
            if field in user_data:
                encrypted_data[field] = encrypt_data(user_data[field])
        
        container.upsert_item(encrypted_data)
        logger.debug(f"store_user_data: Stored encrypted user data for user_id={user_id}")
    except Exception as e:
        logger.error(f"store_user_data: Error for user_id={user_id}: {str(e)}")
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
    """Assess the patient's mental health using Azure OpenAI for dynamic responses."""
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
    patient_email = user_data.get("email", "unknown@example.com")  # Retrieve patient's email

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

Patient Contact Email: {patient_email}

Best regards,
SoulSync Team
    """

    user_response_prompt = f"""
    Based on the therapy plan: '{therapy_response}', craft a short, warm message to the user. Mention the therapy type, one focus area, one recommendation, and that a therapist will reach out. Keep it encouraging and casual, explaining why therapy is needed (e.g., 'to help with your stress and sleep struggles').
    """
    user_response = invoke_azure_openai([{"role": "user", "content": user_response_prompt}], "Generate a user-friendly response.")

    # Send the therapy plan to the therapist, including the patient's email
    send_therapy_plan_to_therapist(therapy_plan_text, user_id, patient_email)

    return {
        "user_response": user_response,
        "therapy_plan": therapy_plan_text,
        "agent_name": "therapy_expert"
    }

def post_rehab_followup(patient_status: str, user_id: str = None, chat_history: list = None) -> dict:
    """Assess post-rehab status, analyze therapy feedback, provide solutions, and escalate high-risk cases using Azure OpenAI."""
    # Summarize the patient's initial status
    summary = get_summary(patient_status)

    # Retrieve user data to get the patient's email
    user_data = get_user_data(user_id)
    patient_email = user_data.get("email", "unknown@example.com")  # Fallback if email is missing

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
        # Pass the patient's email to the send_therapy_plan_to_therapist function
        send_therapy_plan_to_therapist(therapy_data["therapy_plan"], user_id, patient_email)

    return return_data

def send_therapy_plan_to_therapist(therapy_plan: str, user_id: str, patient_email: str = "unknown@example.com") -> bool:
    """Send the therapy plan to the therapist via email, including the patient's email."""
    therapist_email = os.getenv("THERAPIST_EMAIL")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([therapist_email, smtp_server, smtp_port, smtp_user, smtp_password]):
        logger.error("Missing SMTP configuration")
        return False

    # Extract the subject from the therapy plan, or use a default
    subject = therapy_plan.split("\n")[0].strip() if therapy_plan.startswith("Subject:") else f"Therapy Plan for User {user_id}"

    # Update the email body to include the patient's email
    email_body = f"{therapy_plan}\n\nPatient Contact Email: {patient_email}"

    msg = MIMEText(email_body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = therapist_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        logger.info(f"Therapy plan emailed to {therapist_email} for user_id={user_id}, patient_email=REDACTED")
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

    logger.debug(f"Received signup data: {data}")

    for key, value in data.items():
        if not isinstance(value, (str, int, float, bool, type(None))):
            logger.error(f"Non-serializable value found in signup data: {key}={value} (type: {type(value)})")
            return jsonify({"error": f"Invalid data type for {key}: must be a string or number"}), 400

    email = data["email"]
    encrypted_email = encrypt_data(email)
    logger.debug(f"Encrypted email for {email}: {encrypted_email}")
    
    query = "SELECT * FROM c WHERE c.email = @email"
    parameters = [{"name": "@email", "value": encrypted_email}]
    existing_users = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    if existing_users:
        logger.warning(f"Signup failed: Email {email} already exists")
        return jsonify({"error": "Email already exists"}), 400

    otp = generate_otp()
    logger.debug(f"Generated OTP for {email}: {otp}")
    
    if not send_otp_email(email, otp):
        logger.error(f"Failed to send OTP email to {email}")
        return jsonify({"error": "Failed to send OTP"}), 500
    
    if not store_otp(email, otp):
        logger.error(f"Failed to store OTP for {email}")
        return jsonify({"error": "Failed to store OTP"}), 500

    encrypted_data = {key: encrypt_data(value) if key in required_fields else value for key, value in data.items()}
    logger.debug(f"Encrypted signup data: {encrypted_data}")

    pending_signup_doc = {
        "id": f"pending_signup_{email}",  # Use plain email for the id
        "type": "pending_signup",
        "email": encrypted_email,
        "patientId": encrypted_email,
        "signup_data": encrypted_data,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    }
    try:
        container.upsert_item(pending_signup_doc)
        logger.info(f"Pending signup stored for {email}")
        return jsonify({"message": "OTP sent to your email."}), 200
    except Exception as e:
        logger.error(f"Failed to store pending signup for {email}: {str(e)}")
        return jsonify({"error": "Failed to process signup"}), 500

@app.route("/verify_signup_otp", methods=["POST"])
def verify_signup_otp():
    data = request.get_json()
    email = data.get("email")
    otp = data.get("otp")
    if not email or not otp:
        logger.warning("Verify signup OTP failed: Missing email or OTP")
        return jsonify({"error": "Missing email or OTP"}), 400

    if not validate_otp(email, otp):
        return jsonify({"error": "Invalid or expired OTP. Please resend a new OTP."}), 401

    encrypted_email = encrypt_data(email)
    logger.debug(f"Encrypted email for {email}: {encrypted_email}")
    
    query = "SELECT * FROM c WHERE c.id = @id AND c.type = 'pending_signup'"
    parameters = [{"name": "@id", "value": f"pending_signup_{email}"}]
    pending_docs = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

    if not pending_docs:
        logger.warning(f"No pending signup found for {email}")
        return jsonify({"error": "No pending signup found"}), 404

    pending_doc = pending_docs[0]
    signup_data = {key: decrypt_data(value) for key, value in pending_doc["signup_data"].items()}
    expires_at = datetime.fromisoformat(pending_doc["expires_at"].replace("Z", "+00:00"))

    if datetime.now(timezone.utc) > expires_at:
        container.delete_item(item=pending_doc["id"], partition_key=pending_doc["patientId"])
        logger.warning(f"Pending signup expired for {email}")
        return jsonify({"error": "Signup request expired. Please start the signup process again."}), 401

    user_id = str(uuid.uuid4())
    user_doc = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        **{key: encrypt_data(value) for key, value in signup_data.items()},
        "messages": encrypt_data("[]"),  # Add messages field
        "context": encrypt_data("{}"),  # Add context field
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    try:
        container.create_item(user_doc)
        container.delete_item(item=pending_doc["id"], partition_key=pending_doc["patientId"])
        logger.info(f"User signed up: {user_id}")
        return jsonify({"message": "Signup successful", "user_id": user_id}), 201
    except Exception as e:
        logger.error(f"Failed to complete signup for {email}: {str(e)}")
        return jsonify({"error": "Failed to complete signup"}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data.get("email") or not data.get("password"):
        logger.warning("Login failed: Missing email or password")
        return jsonify({"error": "Missing email or password"}), 400

    email = data["email"]
    encrypted_email = encrypt_data(email)
    query = "SELECT * FROM c WHERE c.email = @email"
    parameters = [{"name": "@email", "value": encrypted_email}]
    users = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

    if not users or users[0]["password"] != data["password"]:
        logger.warning(f"Login failed for {email}: Invalid credentials")
        return jsonify({"error": "Invalid email or password"}), 401

    otp = generate_otp()
    if not send_otp_email(email, otp) or not store_otp(email, otp):
        return jsonify({"error": "Failed to send or store OTP"}), 500

    pending_login_doc = {
        "id": f"pending_login_{email}",
        "type": "pending_login",
        "email": encrypted_email,
        "patientId": encrypted_email,
        "user_id": users[0]["user_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    }
    try:
        container.upsert_item(pending_login_doc)
        logger.info(f"Pending login stored for {email}")
        return jsonify({"message": "OTP sent to your email. Please verify to complete login."}), 200
    except Exception as e:
        logger.error(f"Failed to store pending login for {email}: {str(e)}")
        return jsonify({"error": "Failed to process login"}), 500

@app.route("/verify_login_otp", methods=["POST"])
def verify_login_otp():
    data = request.get_json()
    email = data.get("email")
    otp = data.get("otp")
    if not email or not otp:
        logger.warning("Verify login OTP failed: Missing email or OTP")
        return jsonify({"error": "Missing email or OTP"}), 400

    if not validate_otp(email, otp):
        return jsonify({"error": "Invalid or expired OTP. Please resend a new OTP."}), 401

    encrypted_email = encrypt_data(email)
    logger.debug(f"Encrypted email for {email}: {encrypted_email}")
    
    query = "SELECT * FROM c WHERE c.id = @id AND c.type = 'pending_login'"
    parameters = [{"name": "@id", "value": f"pending_login_{email}"}]
    pending_docs = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

    if not pending_docs:
        logger.warning(f"No pending login found for {email}")
        return jsonify({"error": "No pending login found"}), 404

    pending_doc = pending_docs[0]
    user_id = pending_doc["user_id"]
    expires_at = datetime.fromisoformat(pending_doc["expires_at"].replace("Z", "+00:00"))

    if datetime.now(timezone.utc) > expires_at:
        container.delete_item(item=pending_doc["id"], partition_key=pending_doc["patientId"])
        logger.warning(f"Pending login expired for {email}")
        return jsonify({"error": "Login request expired. Please start the login process again."}), 401

    try:
        container.delete_item(item=pending_doc["id"], partition_key=pending_doc["patientId"])
        logger.info(f"User logged in: {user_id}")
        return jsonify({"message": "Login successful", "user_id": user_id}), 200
    except Exception as e:
        logger.error(f"Failed to complete login for {email}: {str(e)}")
        return jsonify({"error": "Failed to complete login"}), 500

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



@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    required_fields = ["firstName", "lastName", "dob", "email", "password"]
    
    if not all(field in data for field in required_fields):
        logger.warning("Signup failed: Missing required fields")
        return jsonify({"error": "Missing required fields"}), 400

    logger.debug(f"Received signup data: {data}")

    for key, value in data.items():
        if not isinstance(value, (str, int, float, bool, type(None))):
            logger.error(f"Non-serializable value found in signup data: {key}={value} (type: {type(value)})")
            return jsonify({"error": f"Invalid data type for {key}: must be a string or number"}), 400

    email = data["email"]
    encrypted_email = encrypt_data(email)
    query = "SELECT * FROM c WHERE c.email = @email"
    parameters = [{"name": "@email", "value": encrypted_email}]
    existing_users = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    if existing_users:
        logger.warning(f"Signup failed: Email {email} already exists")
        return jsonify({"error": "Email already exists"}), 400

    otp = generate_otp()
    logger.debug(f"Generated OTP for {email}: {otp}")
    
    if not send_otp_email(email, otp):
        logger.error(f"Failed to send OTP email to {email}")
        return jsonify({"error": "Failed to send OTP"}), 500
    
    if not store_otp(email, otp):
        logger.error(f"Failed to store OTP for {email}")
        return jsonify({"error": "Failed to store OTP"}), 500

    encrypted_data = {key: encrypt_data(value) if key in required_fields else value for key, value in data.items()}
    logger.debug(f"Encrypted signup data: {encrypted_data}")

    pending_signup_doc = {
        "id": f"pending_signup_{email}",
        "type": "pending_signup",
        "email": encrypted_email,
        "patientId": encrypted_email,
        "signup_data": encrypted_data,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    }
    try:
        container.upsert_item(pending_signup_doc)
        logger.info(f"Pending signup stored for {email}")
        return jsonify({"message": "OTP sent to your email."}), 200
    except Exception as e:
        logger.error(f"Failed to store pending signup for {email}: {str(e)}")
        return jsonify({"error": "Failed to process signup"}), 500

@app.route("/verify_signup_otp", methods=["POST"])
def verify_signup_otp():
    data = request.get_json()
    email = data.get("email")
    otp = data.get("otp")
    if not email or not otp:
        logger.warning("Verify signup OTP failed: Missing email or OTP")
        return jsonify({"error": "Missing email or OTP"}), 400

    if not validate_otp(email, otp):
        return jsonify({"error": "Invalid or expired OTP"}), 401

    query = "SELECT * FROM c WHERE c.id = @id AND c.type = 'pending_signup'"
    parameters = [{"name": "@id", "value": f"pending_signup_{email}"}]
    pending_docs = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

    if not pending_docs:
        logger.warning(f"No pending signup found for {email}")
        return jsonify({"error": "No pending signup data found"}), 404

    pending_doc = pending_docs[0]
    signup_data = pending_doc["signup_data"]
    expires_at = datetime.fromisoformat(pending_doc.get("expires_at").replace("Z", "+00:00"))

    if datetime.now(timezone.utc) > expires_at:
        logger.warning(f"Pending signup expired for {email}")
        container.delete_item(item=pending_doc["patientId"], partition_key=pending_doc["patientId"])
        return jsonify({"error": "Signup request expired"}), 401

    user_id = str(uuid.uuid4())
    user_doc = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        **signup_data,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    try:
        container.create_item(user_doc)
        container.delete_item(item=pending_doc["id"], partition_key=pending_doc["patientId"])
        logger.info(f"User signed up: {user_id}")
        return jsonify({"message": "Signup successful", "user_id": user_id}), 201
    except Exception as e:
        logger.error(f"Failed to complete signup for {email}: {str(e)}")
        return jsonify({"error": "Failed to complete signup"}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data.get("email") or not data.get("password"):
        logger.warning("Login failed: Missing email or password")
        return jsonify({"error": "Missing email or password"}), 400

    email = data["email"]
    encrypted_email = encrypt_data(email)
    logger.debug(f"Encrypted email for {email}: {encrypted_email}")
    
    query = "SELECT * FROM c WHERE c.email = @email"
    parameters = [{"name": "@email", "value": encrypted_email}]
    users = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

    if not users or decrypt_data(users[0]["password"]) != data["password"]:
        logger.warning(f"Login failed for {email}: Invalid credentials")
        return jsonify({"error": "Invalid email or password"}), 401

    otp = generate_otp()
    logger.debug(f"Generated OTP for {email}: {otp}")
    
    if not send_otp_email(email, otp):
        logger.error(f"Failed to send OTP email to {email}")
        return jsonify({"error": "Failed to send OTP"}), 500
    
    if not store_otp(email, otp):
        logger.error(f"Failed to store OTP for {email}")
        return jsonify({"error": "Failed to store OTP"}), 500

    pending_login_doc = {
        "id": f"pending_login_{email}",  # Use plain email for the id
        "type": "pending_login",
        "email": encrypted_email,
        "patientId": encrypted_email,
        "user_id": users[0]["user_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    }
    try:
        container.upsert_item(pending_login_doc)
        logger.info(f"Pending login stored for {email}")
        return jsonify({"message": "OTP sent to your email."}), 200
    except Exception as e:
        logger.error(f"Failed to store pending login for {email}: {str(e)}")
        return jsonify({"error": "Failed to process login"}), 500

@app.route("/verify_login_otp", methods=["POST"])
def verify_login_otp():
    data = request.get_json()
    email = data.get("email")
    otp = data.get("otp")
    if not email or not otp:
        logger.warning("Verify login OTP failed: Missing email or OTP")
        return jsonify({"error": "Missing email or OTP"}), 400

    if not validate_otp(email, otp):
        return jsonify({"error": "Invalid or expired OTP. Please resend a new OTP."}), 401

    encrypted_email = encrypt_data(email)
    logger.debug(f"Encrypted email for {email}: {encrypted_email}")
    
    query = "SELECT * FROM c WHERE c.id = @id AND c.type = 'pending_login'"
    parameters = [{"name": "@id", "value": f"pending_login_{email}"}]  # Use plain email for the id
    pending_docs = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

    if not pending_docs:
        logger.warning(f"No pending login found for {email}")
        return jsonify({"error": "No pending login found"}), 404

    pending_doc = pending_docs[0]
    user_id = pending_doc["user_id"]
    expires_at = datetime.fromisoformat(pending_doc["expires_at"].replace("Z", "+00:00"))

    if datetime.now(timezone.utc) > expires_at:
        container.delete_item(item=pending_doc["id"], partition_key=pending_doc["patientId"])
        logger.warning(f"Pending login expired for {email}")
        return jsonify({"error": "Login request expired. Please start the login process again."}), 401

    try:
        container.delete_item(item=pending_doc["id"], partition_key=pending_doc["patientId"])
        logger.info(f"User logged in: {user_id}")
        return jsonify({"message": "Login successful", "user_id": user_id}), 200
    except Exception as e:
        logger.error(f"Failed to complete login for {email}: {str(e)}")
        return jsonify({"error": "Failed to complete login"}), 500
    
@app.route("/resend_otp", methods=["POST"])
def resend_otp():
    data = request.get_json()
    email = data.get("email")
    if not email:
        logger.warning("Resend OTP failed: Missing email")
        return jsonify({"error": "Missing email"}), 400

    encrypted_email = encrypt_data(email)
    logger.debug(f"Encrypted email for {email}: {encrypted_email}")
    
    query = "SELECT * FROM c WHERE c.id = @id AND c.type = 'pending_signup'"
    parameters = [{"name": "@id", "value": f"pending_signup_{email}"}]  # Use plain email for the id
    pending_docs = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

    if not pending_docs:
        logger.warning(f"No pending signup found for {email} to resend OTP")
        return jsonify({"error": "No pending signup found"}), 404

    otp = generate_otp()
    logger.debug(f"Generated new OTP for {email}: {otp}")
    
    if not send_otp_email(email, otp):
        logger.error(f"Failed to send OTP email to {email}")
        return jsonify({"error": "Failed to send OTP"}), 500
    
    if not store_otp(email, otp):
        logger.error(f"Failed to store new OTP for {email}")
        return jsonify({"error": "Failed to store OTP"}), 500

    logger.info(f"Resent OTP {otp} for {email}")
    return jsonify({"message": "New OTP sent to your email."}), 200

if __name__ == "__main__":
    logger.info("Starting SoulSync application...")
    app.run(debug=True, host="127.0.0.1", port=5000)