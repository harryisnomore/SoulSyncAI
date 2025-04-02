import os
import smtplib
from email.mime.text import MIMEText
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
from openai._exceptions import RateLimitError
from azure.cosmos import CosmosClient, exceptions
from pathlib import Path
from dotenv import load_dotenv
import uuid
import random
from typing import TypedDict, List, Optional
from datetime import datetime, timedelta, timezone
import time

from langgraph.graph import StateGraph, END
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Email Configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
THERAPIST_EMAIL = "harshit. @auxiliobits.com"

if not all([SMTP_USER, SMTP_PASSWORD]):
    logger.warning("Email configuration is incomplete. Emails will not be sent.")

# Define State Schema
class AgentState(TypedDict):
    messages: List[dict]
    user_id: str
    next_agent: Optional[str]
    context: dict  # Store conversation context

# Define Tools for Agents
def store_user_response(user_id: str, response: str, question_index: int) -> str:
    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
        user_docs = list(container.query_items(query, enable_cross_partition_query=True))
        
        if not user_docs:
            user_doc = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "question_index": 0,
                "answers": [],
                "last_assessment": None,
                "recent_assessment_notified": False
            }
            logger.info(f"store_user_response: Creating new user_doc for user_id={user_id}: {user_doc}")
            container.create_item(user_doc)
        else:
            user_doc = user_docs[0]
            logger.info(f"store_user_response: Retrieved user_doc for user_id={user_id}: {user_doc}")

        # Validate response for scale questions
        if question_index in [0, 1, 3]:
            try:
                response_value = int(response)
                if not (1 <= response_value <= 10):
                    response = "5"
            except ValueError:
                response = "5"

        answers = user_doc.get("answers", [])
        while len(answers) <= question_index:
            answers.append(None)
        answers[question_index] = response
        user_doc["answers"] = answers
        user_doc["question_index"] = question_index + 1
        user_doc["last_assessment"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"store_user_response: Updating user_doc for user_id={user_id}: {user_doc}")
        container.upsert_item(user_doc)
        return "Response stored."

    except Exception as e:
        logger.error(f"store_user_response: Error for user_id={user_id}, question_index={question_index}: {str(e)}", exc_info=True)
        raise

def get_user_data(user_id: str) -> dict:
    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
        user_docs = list(container.query_items(query, enable_cross_partition_query=True))
        if not user_docs:
            # Create a new user document with default values
            user_doc = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "question_index": 0,
                "answers": [],
                "last_assessment": None,
                "messages": [],
                "context": {},
                "next_agent": None,
                "recent_assessment_notified": False
            }
            container.create_item(user_doc)
            logger.info(f"get_user_data: Created new user_doc for user_id={user_id}: {user_doc}")
            return user_doc
        return user_docs[0]
    except Exception as e:
        logger.error(f"get_user_data: Error for user_id={user_id}: {str(e)}", exc_info=True)
        raise

def invoke_azure_openai(messages, system_prompt, max_retries=3):
    """Invoke Azure OpenAI API with error handling and rate limiting."""
    retry_count = 0  
    base_delay = 1  # Initial delay in seconds

    while retry_count < max_retries:
        try:
            response = azure_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}] + messages,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            retry_count += 1
            if retry_count == max_retries:
                logger.error(f"invoke_azure_openai: Max retries reached for rate limit error: {str(e)}")
                raise
            # Exponential backoff: delay = base_delay * 2^retry_count
            delay = base_delay * (2 ** retry_count)
            logger.warning(f"invoke_azure_openai: Rate limit error, retrying in {delay} seconds (attempt {retry_count}/{max_retries})")
            time.sleep(delay)

        except Exception as e:
            logger.error(f"invoke_azure_openai: Error - {str(e)}", exc_info=True)
            # Check if the error is a content filter error
            if "content_filter" in str(e).lower() and "400" in str(e):
                # Return a fallback response based on the context
                if "Determine the primary intent" in system_prompt:
                    return "wellness"  # Default to wellness for intent analysis
                elif "Determine if the user wants to switch intents" in system_prompt:
                    return "Yes"  # Default to switching intents
                else:
                    return "I’m here to help. Let’s try rephrasing your message to continue our conversation."
            else:
                # For other errors, raise the exception to be caught higher up
                raise

def store_therapy_plan(user_id: str, therapy_plan: str, recommendation: str) -> str:
    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
        user_docs = list(container.query_items(query, enable_cross_partition_query=True))
        if not user_docs:
            user_doc = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "therapy_plans": []
            }
            container.create_item(user_doc)
        else:
            user_doc = user_docs[0]

        therapy_plans = user_doc.get("therapy_plans", [])
        therapy_plans.append({
            "plan": therapy_plan,
            "recommendation": recommendation,
            "created_at": datetime.now(timezone.utc).isoformat()
        })
        user_doc["therapy_plans"] = therapy_plans

        container.upsert_item(user_doc)
        return "Therapy plan stored."
    except Exception as e:
        logger.error(f"store_therapy_plan: Error for user_id={user_id}: {str(e)}", exc_info=True)
        raise

def send_therapy_recommendation_email(user_id: str, conversation: List[dict], recommendation: str):
    if not all([SMTP_USER, SMTP_PASSWORD]):
        logger.error("Email configuration is missing. Cannot send email.")
        return False

    try:
        user_data = get_user_data(user_id)
        if not user_data:
            logger.error(f"No user data found for user_id: {user_id}")
            return False

        patient_details = (
            f"Name: {user_data.get('firstName', 'Unknown')} {user_data.get('lastName', 'Unknown')}\n"
            f"Email: {user_data.get('email', 'Unknown')}\n"
            f"DOB: {user_data.get('dob', 'Unknown')}"
        )

        summary = "Conversation Summary:\n"
        for msg in conversation[-5:]:
            summary += f"{msg['role'].capitalize()}: {msg['content']}\n"

        email_body = (
            f"Subject: Therapy Recommendation for Patient {user_id}\n\n"
            f"{patient_details}\n\n"
            f"{summary}\n\n"
            f"Therapy Recommendation:\n{recommendation}"
        )

        msg = MIMEText(email_body)
        msg["Subject"] = f"Therapy Recommendation for {user_data.get('firstName', 'Unknown')}"
        msg["From"] = SMTP_USER
        msg["To"] = THERAPIST_EMAIL

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, THERAPIST_EMAIL, msg.as_string())
            
        logger.info(f"Email successfully sent to {THERAPIST_EMAIL} for user {user_id}")
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP Authentication failed. Check email credentials.")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP Error: {str(e)}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending email: {str(e)}", exc_info=True)
        return False

# Define Agents
def chat_agent(state: AgentState) -> AgentState:
    # Find the most recent user message
    user_message = None
    for message in reversed(state["messages"]):
        if message["role"] == "user":
            user_message = message["content"]
            break
    
    if user_message is None:
        logger.warning(f"chat_agent: No user message found in messages for user_id={state['user_id']}")
        return state

    # Check if the last message is already an assistant response from chat_agent for this user message
    if state["messages"] and state["messages"][-1]["role"] == "assistant" and state["messages"][-1].get("agent_name") == "chat_agent":
        logger.info(f"chat_agent: Last message is already an assistant response for user_id={state['user_id']}, skipping further processing")
        return state

    user_id = state["user_id"]
    context = state.get("context", {})

    # Check if we've already processed this user message in this workflow invocation
    if "processed_message" in context and context["processed_message"] == user_message:
        last_processed_timestamp = context.get("processed_timestamp", 0)
        last_message_timestamp = state["messages"][-1].get("timestamp", 0)
        if last_message_timestamp <= last_processed_timestamp:
            logger.info(f"chat_agent: Already processed message '{user_message}' for user_id={user_id} at timestamp {last_processed_timestamp}, skipping further processing")
            return state

    system_prompt = """
    Your role is to:
    1. Understand the user's needs and emotions based on their message and conversation history.
    2. Provide an empathetic, natural response without mentioning the routing process.
    3. Classify the user's intent into one of the following categories and determine the appropriate agent to route to:
       - Intent: wellness (for messages indicating mental suppression, emotional distress, or a need for mood assessment. Examples: "I am not well", "I feel sad", "I’m feeling down", "I feel depressed", "I am feeling sad", "I had a fight with my family", "I am frustrated", "I feel overwhelmed", "I am stressed")
         - Route to: wellness_check_agent
       - Intent: therapy (for messages explicitly requesting therapy or indicating a need for professional mental health support. Examples: "I need therapy", "I want to talk to a therapist", "I need professional help", "I want a therapy plan", "I need a therapist", "I want a personalized therapy plan", "I need help from a therapist")
         - Route to: personalized_therapy_agent
       - Intent: rehab (for messages related to post-rehabilitation support, relapse, or recovery. Examples: "I relapsed", "I need rehab support", "I’m struggling after rehab")
         - Route to: post_rehab_follow_up_agent
       - Intent: general (for all other messages that don't fit the above categories. Examples: "How are you?", "Tell me about SoulSync", "hii", "why", "what is this")
         - Route to: None (no further routing needed)
    4. Return the response, intent, and agent in the format:
       Response: [Your empathetic response here]
       Intent: [wellness/therapy/rehab/general]
       Agent: [wellness_check_agent/personalized_therapy_agent/post_rehab_follow_up_agent/None]

    **Important Instructions**:
    - Prioritize explicit keywords: 
      - If the message contains "therapy", "therapist", or "therapy plan", classify the intent as "therapy" and route to "personalized_therapy_agent".
      - If the message contains phrases indicating emotional distress such as "not well", "feel sad", "depressed", "feeling down", "frustrated", "stressed", "overwhelmed", or mentions conflicts like "fight with", classify the intent as "wellness" and route to "wellness_check_agent".
    - Use the conversation history to understand the user's emotional state. If the user has previously expressed distress or requested therapy, consider this context when classifying the intent.
    - Be strict about following the examples provided for each intent category.
    """

    try:
        user_data = get_user_data(user_id)
    except Exception as e:
        logger.error(f"chat_agent: Failed to get user data for user_id={user_id}: {str(e)}", exc_info=True)
        raise

    # Prepare the conversation history (last 5 messages for context)
    conversation_history = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"][-5:]])
    
    messages = [{
        "role": "user",
        "content": f"User History: {user_data}\nRecent Conversation:\n{conversation_history}\nCurrent Message: {user_message}"
    }]
    
    logger.info(f"chat_agent: Invoking Azure OpenAI for user_id={user_id}, message='{user_message}'")
    result = invoke_azure_openai(messages, system_prompt)
    logger.info(f"chat_agent: Azure OpenAI response for user_id={user_id}: {result}")

    # Parse the response, intent, and agent
    try:
        parts = result.split("\n")
        if len(parts) != 3:
            raise ValueError("Expected 3 parts (Response, Intent, Agent), but got: " + str(parts))
        
        response_part = parts[0].replace("Response:", "").strip()
        intent_part = parts[1].replace("Intent:", "").strip().lower()
        agent_part = parts[2].replace("Agent:", "").strip()

        response = response_part
        intent = intent_part
        next_agent = agent_part if agent_part != "None" else None
    except Exception as e:
        logger.warning(f"chat_agent: Failed to parse Azure OpenAI response for user_id={user_id}: {str(e)}, response={result}")
        response = "I’m here to help. Could you please share more about how you’re feeling?"
        intent = "general"
        next_agent = None

    # Log the parsed values for debugging
    logger.info(f"chat_agent: Parsed values for user_id={user_id}, message='{user_message}', response='{response}', intent='{intent}', next_agent='{next_agent}'")

    # Fallback routing for explicit therapy requests
    if "therapy" in user_message.lower() or "therapist" in user_message.lower():
        logger.info(f"chat_agent: Fallback check for therapy request in message='{user_message}'")
        next_agent = "personalized_therapy_agent"
        intent = "therapy"
        logger.info(f"chat_agent: Fallback routing applied for user_id={user_id}, message='{user_message}', set next_agent to 'personalized_therapy_agent'")

    # Fallback routing for wellness-related messages
    if any(keyword in user_message.lower() for keyword in ["not well", "feel sad", "depressed", "feeling down", "frustrated", "stressed", "overwhelmed", "fight with"]):
        logger.info(f"chat_agent: Fallback check for wellness request in message='{user_message}'")
        next_agent = "wellness_check_agent"
        intent = "wellness"
        logger.info(f"chat_agent: Fallback routing applied for user_id={user_id}, message='{user_message}', set next_agent to 'wellness_check_agent'")

    # Validate the intent and agent
    valid_intents = ["wellness", "therapy", "rehab", "general"]
    valid_agents = ["wellness_check_agent", "personalized_therapy_agent", "post_rehab_follow_up_agent", None]
    if intent not in valid_intents:
        logger.warning(f"chat_agent: Invalid intent '{intent}' for user_id={user_id}, message='{user_message}', defaulting to 'general'")
        intent = "general"
        next_agent = None
    if next_agent not in valid_agents:
        logger.warning(f"chat_agent: Invalid agent '{next_agent}' for user_id={user_id}, message='{user_message}', defaulting to None")
        next_agent = None

    # Log the final routing decision
    logger.info(f"chat_agent: Final routing decision for user_id={user_id}, message='{user_message}', intent='{intent}', next_agent='{next_agent}'")

    # Update context with the processed message and timestamp
    context.update({
        "last_intent": intent,
        "conversation_start": context.get("conversation_start", datetime.now(timezone.utc).isoformat()),
        "processed_message": user_message,
        "processed_timestamp": datetime.now(timezone.utc).timestamp()
    })

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response, "agent_name": "chat_agent", "timestamp": datetime.now(timezone.utc).timestamp()}],
        "user_id": user_id,
        "next_agent": next_agent,
        "context": context
    }

def wellness_agent(state: AgentState) -> AgentState:
    # Find the most recent user message
    user_message = None
    for message in reversed(state["messages"]):
        if message["role"] == "user":
            user_message = message["content"]
            break
    
    if user_message is None:
        logger.warning(f"wellness_check_agent: No user message found in messages for user_id={state['user_id']}")
        return state

    user_id = state["user_id"]
    context = state.get("context", {})

    system_prompt = """
    You are the Wellness Assessment specialist of SoulSync AI.
    Focus on:
    1. Natural conversation flow
    2. Emotional support
    3. Careful assessment of mental well-being
    4. Appropriate recommendations based on responses

    Avoid:
    1. Repetitive questions
    2. Clinical or impersonal language
    3. Abrupt topic changes
    """

    # Check if the user wants to switch intents (e.g., "I need therapy")
    intent_check_prompt = f"Analyze this message: {user_message}\nDoes the user want to switch to a different intent (e.g., therapy, rehab, general)? Respond with 'Yes' or 'No'."
    intent_switch = invoke_azure_openai(
        [{"role": "user", "content": intent_check_prompt}],
        "Determine if the user wants to switch intents."
    )
    if "Yes" in intent_switch:
        # Reset the wellness check state and route back to chat_agent
        try:
            user_data = get_user_data(user_id)
            user_data["question_index"] = 0
            user_data["answers"] = []
            user_data["recent_assessment_notified"] = False
            container.upsert_item(user_data)
        except Exception as e:
            logger.error(f"wellness_check_agent: Failed to reset state for user_id={user_id}: {str(e)}", exc_info=True)
            raise
        response = "I understand, let’s switch gears. I’ll pass you back to the main chat agent to assist with your request."
        return {
            "messages": state["messages"] + [{"role": "assistant", "content": response, "agent_name": "wellness_check_agent"}],
            "user_id": user_id,
            "next_agent": None,  # Route back to chat_agent for intent re-evaluation
            "context": context
        }

    try:
        user_data = get_user_data(user_id)
        logger.info(f"wellness_check_agent: Retrieved user_data for user_id={user_id}: {user_data}")
    except Exception as e:
        logger.error(f"wellness_check_agent: Failed to get user data for user_id={user_id}: {str(e)}", exc_info=True)
        raise

    last_assessment = user_data.get("last_assessment")
    recent_assessment_notified = user_data.get("recent_assessment_notified", False)
    
    # Check if enough time has passed since last assessment (24 hours), but only if we haven't already notified the user
    if last_assessment and not recent_assessment_notified:
        last_assessment_time = datetime.fromisoformat(last_assessment)
        if datetime.now(timezone.utc) - last_assessment_time < timedelta(hours=24):
            response = (
                "I notice we've recently done a wellness check. Instead, let’s focus on how you're feeling right now. "
                "What would you like to talk about?"
            )
            # Set the flag to indicate that the user has been notified
            try:
                user_data["recent_assessment_notified"] = True
                container.upsert_item(user_data)
                logger.info(f"wellness_check_agent: Set recent_assessment_notified for user_id={user_id}")
            except Exception as e:
                logger.error(f"wellness_check_agent: Failed to set recent_assessment_notified for user_id={user_id}: {str(e)}", exc_info=True)
                raise
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": response, "agent_name": "wellness_check_agent"}],
                "user_id": user_id,
                "next_agent": None,  # Route back to chat_agent
                "context": context
            }

    question_index = user_data.get("question_index", 0)
    logger.info(f"wellness_check_agent: question_index for user_id={user_id}: {question_index}")
    answers = user_data.get("answers", [])

    questions = [
        "How are you feeling today? On a scale of 1-10, where would you place your mood?",
        "In the past week, how often have you felt overwhelmed or anxious? (1 being rarely, 10 being constantly)",
        "Tell me about your energy levels and motivation lately.",
        "How well have you been sleeping? (1-10, where 10 is excellent)",
        "How connected do you feel to others around you?"
    ]

    if question_index == 0:
        response = "I'd like to understand how you're doing better. " + questions[0]
        try:
            store_user_response(user_id, user_message, question_index)
        except Exception as e:
            logger.error(f"wellness_check_agent: Failed to store response for user_id={user_id}, question_index={question_index}: {str(e)}", exc_info=True)
            raise
        next_agent = "wellness_check_agent"
    elif 0 < question_index < len(questions):
        try:
            store_user_response(user_id, user_message, question_index - 1)
        except Exception as e:
            logger.error(f"wellness_check_agent: Failed to store response for user_id={user_id}, question_index={question_index - 1}: {str(e)}", exc_info=True)
            raise
        response = questions[question_index]
        next_agent = "wellness_check_agent"
    else:
        # Calculate risk and provide personalized feedback
        try:
            store_user_response(user_id, user_message, question_index - 1)
        except Exception as e:
            logger.error(f"wellness_check_agent: Failed to store response for user_id={user_id}, question_index={question_index - 1}: {str(e)}", exc_info=True)
            raise
        
        # Calculate average score from numeric responses
        numeric_answers = [int(a) if str(a).isdigit() else 5 for a in answers if a is not None]
        avg_score = sum(numeric_answers) / len(numeric_answers) if numeric_answers else 5

        if avg_score < 4:
            response = (
                "I hear you, and I can sense you're going through a challenging time. "
                "Based on our conversation, I think it would be really beneficial to speak with a mental health professional. "
                "Would you like to help you explore some professional support options?"
            )
        elif avg_score <= 6:
            response = (
                "Thank you for sharing with me. While you're managing, I can tell there's room for more support. "
                "Would you like to explore some self-care techniques or discuss creating a personalized wellness plan?"
            )
        else:
            response = (
                "It's great to see you're maintaining good mental well-being! "
                "Would you like to learn about some additional strategies to help maintain this positive state?"
            )
        
        next_agent = None  # Route back to chat_agent
        # Reset assessment state
        try:
            user_data["question_index"] = 0
            user_data["answers"] = []
            user_data["recent_assessment_notified"] = False
            container.upsert_item(user_data)
        except Exception as e:
            logger.error(f"wellness_check_agent: Failed to reset state for user_id={user_id}: {str(e)}", exc_info=True)
            raise

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response, "agent_name": "wellness_check_agent"}],
        "user_id": user_id,
        "next_agent": next_agent,
        "context": context
    }

def therapy_agent(state: AgentState) -> AgentState:
    """
    An adaptive therapy planning agent for SoulSync AI.
    Focuses on:
    1. Crafting personalized therapy plans based on user input and history.
    2. Offering actionable, context-sensitive recommendations.
    3. Setting realistic, user-aligned goals for relief and growth.
    4. Providing supportive, trust-building communication.
    """
    logger.info(f"[THERAPY_AGENT] Processing state for user_id={state['user_id']}")

    # Extract the latest user message
    user_message = None
    for message in reversed(state["messages"]):
        if message["role"] == "user":
            user_message = message["content"].lower().strip()
            break
    
    if not user_message:
        logger.warning(f"therapy_agent: No user message found for user_id={state['user_id']}")
        response = "I couldn’t find your latest message—could you share what’s on your mind?"
        return {
            "messages": state["messages"] + [{"role": "assistant", "content": response, "agent_name": "personalized_therapy_agent"}],
            "user_id": state["user_id"],
            "next_agent": "personalized_therapy_agent",
            "context": state.get("context", {})
        }

    user_id = state["user_id"]
    context = state.get("context", {})

    # Fetch user data with fallback
    try:
        user_data = get_user_data(user_id) or "No prior user data available."
    except Exception as e:
        logger.error(f"therapy_agent: Failed to get user data for user_id={user_id}: {str(e)}", exc_info=True)
        user_data = "User data unavailable due to a glitch—we’ll work with what you tell me now."

    # Build conversation context
    recent_history = state["messages"][-5:]
    conversation_summary = "\n".join([f"{m['role']}: {m['content']}" for m in recent_history])
    full_context = f"User History: {user_data}\nRecent Conversation:\n{conversation_summary}\nCurrent Message: {user_message}"

    # System prompt for natural, adaptive reasoning
    system_prompt = """
    You are the Therapy Planning specialist of SoulSync AI. Your role is to deeply understand the user’s emotional state, needs, and context, then craft a personalized therapy plan. Focus on:
    - Personalization: Tailor the plan to the user’s unique situation, drawing from their history and current input.
    - Actionable steps: Suggest practical, specific actions they can take now.
    - Realistic goals: Set short-term wins and long-term growth targets that feel achievable.
    - Supportive tone: Build trust with warm, empathetic communication.
    Interpret the user’s intent and emotions naturally—don’t rely on rigid rules or keywords. If the user seems to want a different focus (e.g., wellness, rehab), flag it. Respond with a therapy plan and a conversational message.
    """

    # Generate the therapy plan dynamically
    messages = [{"role": "user", "content": full_context}]
    therapy_response = invoke_azure_openai(messages, system_prompt)

    # Parse the response (assuming it’s structured naturally by the AI, e.g., plan + message)
    try:
        # For simplicity, assume therapy_response contains both plan and user message; in practice, you might split these via AI or formatting
        if "therapy plan" not in therapy_response.lower():
            therapy_plan = therapy_response  # Fallback: treat the whole response as the plan
            user_message_response = "Here’s a plan I’ve put together for you—let me know how it feels or if you’d like to tweak it!"
        else:
            # Split if the AI naturally separates plan and message (e.g., with a delimiter or clear sections)
            plan_start = therapy_response.lower().index("therapy plan")
            therapy_plan = therapy_response[plan_start:]
            user_message_response = therapy_response[:plan_start].strip() or "Here’s your therapy plan—how does it sit with you?"
    except Exception as e:
        logger.error(f"therapy_agent: Error parsing therapy response: {str(e)}")
        therapy_plan = therapy_response
        user_message_response = "I’ve got a plan for you—let’s see if it fits!"

    # Check for intent shift organically
    if any(phrase in user_message for phrase in ["switch", "change", "different", "not this", "wellness", "rehab"]):
        logger.info(f"therapy_agent: Detected potential intent shift for user_id={user_id}")
        response = "It sounds like you might want to explore something different—should I hand you back to the main chat agent to pivot?"
        return {
            "messages": state["messages"] + [{"role": "assistant", "content": response, "agent_name": "personalized_therapy_agent"}],
            "user_id": user_id,
            "next_agent": None,  # Route back to chat_agent
            "context": context
        }

    # Generate a concise recommendation as a summary
    recommendation_prompt = f"Based on this:\n{full_context}\nProvide a concise, actionable therapy recommendation (1-2 sentences)."
    recommendation = invoke_azure_openai(
        [{"role": "user", "content": recommendation_prompt}],
        "Summarize a therapy recommendation."
    )

    # Store the therapy plan
    try:
        store_therapy_plan(user_id, therapy_plan, recommendation)
        logger.info(f"therapy_agent: Therapy plan stored for user_id={user_id}")
    except Exception as e:
        logger.error(f"therapy_agent: Failed to store therapy plan for user_id={user_id}: {str(e)}", exc_info=True)
        therapy_plan += "\n\n(Note: We hit a snag saving this plan—our team’s on it.)"

    # Send email with fallback
    email_sent = send_therapy_recommendation_email(user_id, state["messages"], recommendation)
    if not email_sent:
        therapy_plan += "\n\nNote: Couldn’t send the plan to the therapist—our team’s been alerted."
    else:
        therapy_plan += "\n\nYour plan’s been sent to our therapist for a look."

    # Decide next steps intuitively
    follow_up_needed = any(phrase in therapy_response.lower() for phrase in ["discuss", "follow up", "what do you think", "let me know"])
    next_agent = "personalized_therapy_agent" if follow_up_needed else None

    # Combine response for the user
    full_response = f"{user_message_response}\n\n{therapy_plan}"

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": full_response, "agent_name": "personalized_therapy_agent"}],
        "user_id": user_id,
        "next_agent": next_agent,
        "context": context
    }

def rehab_agent(state: AgentState) -> AgentState:
    # Find the most recent user message
    user_message = None
    for message in reversed(state["messages"]):
        if message["role"] == "user":
            user_message = message["content"]
            break
    
    if user_message is None:
        logger.warning(f"rehab_agent: No user message found in messages for user_id={state['user_id']}")
        return state

    user_id = state["user_id"]
    context = state.get("context", {})

    system_prompt = """
    You are the Post-Rehabilitation specialist of SoulSync AI.
    Focus on:
    1. Relapse prevention
    2. Progress tracking
    3. Motivation maintenance
    4. Support system engagement
    """

    # Check if the user wants to switch intents
    intent_check_prompt = f"Analyze this message: {user_message}\nDoes the user want to switch to a different intent (e.g., wellness, therapy, general)? Respond with 'Yes' or 'No'."
    intent_switch = invoke_azure_openai(
        [{"role": "user", "content": intent_check_prompt}],
        "Determine if the user wants to switch intents."
    )
    if "Yes" in intent_switch:
        response = "I understand, let’s switch gears. I’ll pass you back to the main chat agent to assist with your request."
        return {
            "messages": state["messages"] + [{"role": "assistant", "content": response, "agent_name": "post_rehab_follow_up_agent"}],
            "user_id": user_id,
            "next_agent": None,  # Route back to chat_agent
            "context": context
        }

    try:
        user_data = get_user_data(user_id)
    except Exception as e:
        logger.error(f"rehab_agent: Failed to get user data for user_id={user_id}: {str(e)}", exc_info=True)
        raise

    conversation_history = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"][-5:]])
    
    messages = [{
        "role": "user",
        "content": f"User History: {user_data}\nRecent Conversation:\n{conversation_history}\nCurrent Message: {user_message}"
    }]
    
    response = invoke_azure_openai(messages, system_prompt)

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response, "agent_name": "post_rehab_follow_up_agent"}],
        "user_id": user_id,
        "next_agent": None,  # Route back to chat_agent
        "context": context
    }

# Create Workflow
workflow = StateGraph(AgentState)
workflow.add_node("chat_agent", chat_agent)
workflow.add_node("wellness_check_agent", wellness_agent)
workflow.add_node("personalized_therapy_agent", therapy_agent)
workflow.add_node("post_rehab_follow_up_agent", rehab_agent)

workflow.set_entry_point("chat_agent")

# Add conditional edges with logging
def log_and_route(state):
    next_agent = state["next_agent"]
    logger.info(f"Routing from chat_agent: next_agent={next_agent}")
    return next_agent

workflow.add_conditional_edges("chat_agent", log_and_route, {
    "wellness_check_agent": "wellness_check_agent",
    "personalized_therapy_agent": "personalized_therapy_agent",
    "post_rehab_follow_up_agent": "post_rehab_follow_up_agent",
    None: END
})

workflow.add_conditional_edges("wellness_check_agent", lambda state: state["next_agent"], {
    "wellness_check_agent": "wellness_check_agent",
    None: "chat_agent"
})
workflow.add_conditional_edges("personalized_therapy_agent", lambda state: state["next_agent"], {
    "personalized_therapy_agent": "personalized_therapy_agent",
    None: "chat_agent"
})
workflow.add_conditional_edges("post_rehab_follow_up_agent", lambda state: state["next_agent"], {
    "post_rehab_follow_up_agent": "post_rehab_follow_up_agent",
    None: "chat_agent"
})

app_workflow = workflow.compile()

# API Routes
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    required_fields = ["firstName", "lastName", "dob", "email", "password"]
    
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        query = f"SELECT * FROM c WHERE c.email = '{data['email']}'"
        existing_users = list(container.query_items(query, enable_cross_partition_query=True))
        if existing_users:
            return jsonify({"error": "Email already exists"}), 400

        user_doc = {
            "id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            **data,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "question_index": 0,
            "answers": [],
            "recent_assessment_notified": False
        }
        container.create_item(user_doc)

        return jsonify({
            "message": "Signup successful",
            "user_id": user_doc["user_id"]
        }), 201
    except Exception as e:
        logger.error(f"signup: Error for email={data.get('email')}: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred during signup. Please try again."}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    if not data.get("email") or not data.get("password"):
        return jsonify({"error": "Missing email or password"}), 400

    try:
        query = f"SELECT * FROM c WHERE c.email = '{data['email']}'"
        users = list(container.query_items(query, enable_cross_partition_query=True))
        
        if not users or users[0]["password"] != data["password"]:
            return jsonify({"error": "Invalid email or password"}), 401

        return jsonify({
            "message": "Login successful",
            "user_id": users[0]["user_id"]
        }), 200
    except Exception as e:
        logger.error(f"login: Error for email={data.get('email')}: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred during login. Please try again."}), 500

@app.route("/chat", methods=["POST"])
@app.route("/chat", methods=["POST"])
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id")
    user_message = data.get("message", "").strip()
    
    if not user_id or not user_message:
        logger.warning("Missing user_id or message")
        return jsonify({"error": "Missing user_id or message"}), 400

    try:
        user_data = get_user_data(user_id)
        messages = user_data.get("messages", [])
        context = user_data.get("context", {})
        next_agent = user_data.get("next_agent")

        logger.info(f"chat: Starting workflow for user_id={user_id}, message='{user_message}', next_agent={next_agent}, messages={messages}")

        # Append the new user message
        messages.append({"role": "user", "content": user_message})

        # Initialize the state
        state = {
            "messages": messages,
            "user_id": user_id,
            "next_agent": next_agent,
            "context": context
        }

        # Run the workflow until it reaches END or no further transitions are needed
        iteration = 0
        assistant_responses = []
        while True:
            iteration += 1
            logger.info(f"chat: Workflow iteration {iteration} for user_id={user_id}, state={state}")
            result = app_workflow.invoke(state)
            
            # Check if the workflow has reached the END node
            if result.get("__end__"):
                logger.info(f"chat: Workflow reached END for user_id={user_id}")
                break

            # Collect any new assistant responses
            last_message = result["messages"][-1]
            if last_message["role"] == "assistant":
                assistant_responses.append(last_message)
                logger.info(f"chat: Assistant response collected for user_id={user_id}: {last_message}")

            # Check if we should continue (i.e., if next_agent is set)
            if result["next_agent"] is None:
                logger.info(f"chat: No further transitions needed for user_id={user_id}, breaking loop")
                break

            # Update the state for the next iteration
            state = result

        logger.info(f"chat: Workflow result for user_id={user_id}: {result}")

        # Update user data with the final state
        user_data["messages"] = result["messages"]
        user_data["next_agent"] = result["next_agent"]
        user_data["context"] = result["context"]
        container.upsert_item(user_data)

        # Return the last assistant response (or an error if none were generated)
        if not assistant_responses:
            logger.warning(f"chat: No assistant responses generated for user_id={user_id}")
            return jsonify({"error": "No response generated. Please try again."}), 500

        final_message = assistant_responses[-1]
        logger.info(f"chat: Returning response for user_id={user_id}: {final_message}")
        return jsonify({
            "response": final_message["content"],
            "agent_used": final_message.get("agent_name", "chat_agent")
        })

    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"chat: CosmosDB Error for user_id={user_id}: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred accessing the database. Please try again later."}), 500
    except Exception as e:
        logger.error(f"chat: General Error for user_id={user_id}, message='{user_message}': {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred processing your message: {str(e)}. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)