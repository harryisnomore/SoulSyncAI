# import os
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from openai import AzureOpenAI
# from azure.cosmos import CosmosClient, exceptions
# from pathlib import Path
# from dotenv import load_dotenv
# import uuid
# from typing import TypedDict, List
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import InMemorySaver
# import logging

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Load environment variables
# dotenv_path = Path(__file__).resolve().parent / ".env"
# load_dotenv(dotenv_path=dotenv_path)

# app = Flask(__name__)
# CORS(app)

# # Azure OpenAI Configuration
# client = AzureOpenAI(
#     azure_endpoint=os.getenv("ENDPOINT"),
#     api_key=os.getenv("API_KEY"),
#     api_version="2023-05-15"
# )

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

# # Signup endpoint (unchanged)
# @app.route("/signup", methods=["POST"])
# def signup():
#     data = request.json
#     first_name = data.get("firstName")
#     last_name = data.get("lastName")
#     dob = data.get("dob")
#     email = data.get("email")
#     password = data.get("password")  # In production, hash this!

#     if not all([first_name, last_name, dob, email, password]):
#         return jsonify({"error": "Missing required fields"}), 400

#     user_id = str(uuid.uuid4())
#     user_data = {
#         "id": user_id,
#         "user_id": user_id,
#         "first_name": first_name,
#         "last_name": last_name,
#         "dob": dob,
#         "email": email,
#         "password": password,  # Store hashed password in production
#         "conversation_history": []
#     }

#     try:
#         container.upsert_item(user_data)
#         return jsonify({"user_id": user_id, "message": "Signup successful"}), 201
#     except exceptions.CosmosHttpResponseError as e:
#         return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500

# # Login endpoint (unchanged)
# @app.route("/login", methods=["POST"])
# def login():
#     data = request.json
#     email = data.get("email")
#     password = data.get("password")

#     if not email or not password:
#         return jsonify({"error": "Missing email or password"}), 400

#     try:
#         query = f"SELECT * FROM c WHERE c.email = '{email}'"
#         user_docs = list(container.query_items(query, enable_cross_partition_query=True))

#         if not user_docs:
#             return jsonify({"error": "Email not found"}), 404

#         user_doc = user_docs[0]
#         if user_doc["password"] != password:  # In production, compare hashed passwords
#             return jsonify({"error": "Incorrect password"}), 401

#         return jsonify({"user_id": user_doc["user_id"], "message": "Login successful"}), 200

#     except exceptions.CosmosHttpResponseError as e:
#         return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
#     except Exception as e:
#         return jsonify({"error": f"Error: {str(e)}"}), 500

# # Define tools for specialized agents
# def suggest_therapy_plan(input_text: str) -> str:
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "system", "content": "Suggest a therapy plan based on this input."}, {"role": "user", "content": input_text}]
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         logger.error(f"Error in suggest_therapy_plan: {str(e)}")
#         return "Sorry, I couldn’t suggest a therapy plan right now."

# def check_wellness(status: str) -> str:
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "system", "content": "Assess wellness and flag if urgent based on this status."}, {"role": "user", "content": status}]
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         logger.error(f"Error in check_wellness: {str(e)}")
#         return "Sorry, I couldn’t assess your wellness right now."

# def post_rehab_followup(input_text: str) -> str:
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "system", "content": "Provide post-rehab follow-up advice based on this input."}, {"role": "user", "content": input_text}]
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         logger.error(f"Error in post_rehab_followup: {str(e)}")
#         return "Sorry, I couldn’t provide post-rehab advice right now."

# # Custom AzureChatOpenAI wrapper
# class AzureChatOpenAI:
#     def __init__(self, client):
#         self.client = client

#     def invoke(self, messages):
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=messages
#             )
#             return {"role": "assistant", "content": response.choices[0].message.content.strip()}
#         except Exception as e:
#             logger.error(f"Error in AzureChatOpenAI invoke: {str(e)}")
#             raise

# # Define state schema
# class AgentState(TypedDict):
#     messages: List[dict]
#     next: str
#     selected_agent: str

# # Define custom agent logic
# def create_custom_agent(model, tool, name, prompt):
#     def agent_node(state: AgentState) -> AgentState:
#         messages = state["messages"]
#         system_prompt = [{"role": "system", "content": prompt}]
#         tool_prompt = [
#             {"role": "system", "content": f"Use the tool '{tool.__name__}' to process the user's input: {messages[-1]['content']}"}
#         ]
#         try:
#             response = model.invoke(system_prompt + messages + tool_prompt)
#             tool_result = tool(messages[-1]["content"])
#             logger.debug(f"Agent {name} processed input: {messages[-1]['content']} -> {tool_result}")
#             return {"messages": messages + [{"role": "assistant", "content": tool_result}], "selected_agent": name}
#         except Exception as e:
#             logger.error(f"Error in agent {name}: {str(e)}")
#             return {"messages": messages + [{"role": "assistant", "content": f"Sorry, something went wrong with {name}. Try again!"}], "selected_agent": name}

#     graph = StateGraph(AgentState)
#     graph.add_node(name, agent_node)
#     graph.set_entry_point(name)
#     graph.add_edge(name, END)
#     return graph.compile(name=name)

# # Custom supervisor logic
# def create_custom_supervisor(agents, model, prompt, checkpointer):
#     def supervisor_node(state: AgentState) -> AgentState:
#         messages = state["messages"]
#         last_message = messages[-1]["content"].lower().strip()
#         system_prompt = [{"role": "system", "content": prompt}]

#         # Prepare the routing prompt with examples for better intent understanding
#         routing_prompt = system_prompt + messages + [
#             {"role": "system", "content": (
#                 "You are a supervisor for a mental health assistant system. Based on the user's message and conversation history, decide which agent to route to: 'therapy_plan_agent', 'wellness_check_agent', or 'post_rehab_agent'. "
#                 "Return only the agent name. Use the following guidelines:\n"
#                 "- 'therapy_plan_agent' for requests related to therapy suggestions, emotional support, or greetings. Examples: 'I’m feeling anxious', 'I need a therapy plan', 'Hi', 'Hello'.\n"
#                 "- 'wellness_check_agent' for progress monitoring or status updates. Examples: 'How am I doing?', 'I’m feeling okay', 'Check my progress'.\n"
#                 "- 'post_rehab_agent' for follow-up after rehab, including post-rehab advice or questions about recovery after rehab. Examples: 'I finished rehab', 'I’m out of rehab', 'What’s next after rehab?', 'I’ve recovered from rehab'.\n"
#                 "Consider the intent and context of the message. If unsure, choose the most relevant agent based on the conversation history."
#             )}
#         ]
#         try:
#             response = model.invoke(routing_prompt)
#             next_agent = response["content"].strip()
#             logger.debug(f"Supervisor routing decision: {last_message} -> {next_agent}")
#         except Exception as e:
#             logger.error(f"Error in supervisor routing: {str(e)}")
#             return {"messages": messages + [{"role": "assistant", "content": "Sorry, I couldn’t decide which agent to use. Try again!"}], "selected_agent": "supervisor"}

#         # Route to the appropriate agent
#         if next_agent == "therapy_plan_agent":
#             return {"next": "therapy_plan_agent", "messages": messages, "selected_agent": "therapy_plan_agent"}
#         elif next_agent == "wellness_check_agent":
#             return {"next": "wellness_check_agent", "messages": messages, "selected_agent": "wellness_check_agent"}
#         elif next_agent == "post_rehab_agent":
#             return {"next": "post_rehab_agent", "messages": messages, "selected_agent": "post_rehab_agent"}
#         else:
#             logger.warning(f"Unknown agent selected: {next_agent}")
#             return {"messages": messages + [{"role": "assistant", "content": "Sorry, I couldn’t determine the right agent for your request."}], "selected_agent": "supervisor"}

#     # Build the supervisor graph
#     graph = StateGraph(AgentState)
#     graph.add_node("supervisor", supervisor_node)
#     for agent in agents:
#         graph.add_node(agent.name, agent)
#     graph.set_entry_point("supervisor")
#     graph.add_conditional_edges("supervisor", lambda state: state.get("next", END), {
#         "therapy_plan_agent": "therapy_plan_agent",
#         "wellness_check_agent": "wellness_check_agent",
#         "post_rehab_agent": "post_rehab_agent",
#         END: END
#     })
#     for agent in agents:
#         graph.add_edge(agent.name, END)

#     return graph.compile(checkpointer=checkpointer)

# # Initialize supervisor
# def initialize_supervisor():
#     model = AzureChatOpenAI(client)

#     # Create specialized agents
#     therapy_agent = create_custom_agent(
#         model=model,
#         tool=suggest_therapy_plan,
#         name="therapy_plan_agent",
#         prompt="You are an expert in creating personalized therapy plans. If the user greets you with 'hi', 'hello', or 'hii', respond with a friendly greeting and ask how they’re feeling."
#     )

#     wellness_agent = create_custom_agent(
#         model=model,
#         tool=check_wellness,
#         name="wellness_check_agent",
#         prompt="You are a wellness monitoring expert."
#     )

#     post_rehab_agent = create_custom_agent(
#         model=model,
#         tool=post_rehab_followup,
#         name="post_rehab_agent",
#         prompt="You are a post-rehab follow-up expert."
#     )

#     # Validate agents
#     agents = [therapy_agent, wellness_agent, post_rehab_agent]
#     for agent in agents:
#         if not hasattr(agent, "invoke"):
#             raise ValueError(f"Agent {agent} is not properly initialized with an invoke method.")

#     try:
#         checkpointer = InMemorySaver()
#         workflow = create_custom_supervisor(
#             agents=agents,
#             model=model,
#             prompt=(
#                 "You are SoulSync, a chill AI assistant from the future acting as a supervisor. "
#                 "Route patient messages to the appropriate agent based on their intent and context."
#             ),
#             checkpointer=checkpointer
#         )

#         return workflow
#     except Exception as e:
#         raise RuntimeError(f"Failed to initialize supervisor: {str(e)}")

# # Global supervisor instance
# app.supervisor = initialize_supervisor()

# # Updated chat endpoint
# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.json
#     user_id = data.get("user_id")
#     user_message = data.get("message", "").strip()

#     if not user_id or not user_message:
#         return jsonify({"error": "Missing user_id or message"}), 400

#     try:
#         logger.debug(f"Received chat request: user_id={user_id}, message={user_message}")
#         query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
#         user_docs = list(container.query_items(query, enable_cross_partition_query=True))

#         if not user_docs:
#             logger.warning(f"User not found: {user_id}")
#             return jsonify({"error": "User not found"}), 404

#         user_doc = user_docs[0]
#         conversation_history = user_doc.get("conversation_history", [])

#         # Invoke the supervisor with a thread_id
#         result = app.supervisor.invoke(
#             {"messages": conversation_history + [{"role": "user", "content": user_message}]},
#             config={"configurable": {"thread_id": user_id}}
#         )
#         logger.debug(f"Supervisor result: {result}")

#         # Extract the assistant's response and selected agent
#         ai_response = result["messages"][-1]["content"] if result["messages"] else "No response generated."
#         selected_agent = result.get("selected_agent", "unknown")
#         logger.debug(f"Assistant response from {selected_agent}: {ai_response}")

#         # Update conversation history
#         conversation_history.append({"role": "user", "content": user_message})
#         conversation_history.append({"role": "assistant", "content": ai_response})
#         user_doc["conversation_history"] = conversation_history
#         container.upsert_item(user_doc)

#         # Return both the response and the selected agent
#         return jsonify({"response": ai_response, "agent": selected_agent})

#     except exceptions.CosmosHttpResponseError as e:
#         logger.error(f"CosmosDB Error: {str(e)}")
#         return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
#     except Exception as e:
#         logger.error(f"Chat endpoint error: {str(e)}")
#         return jsonify({"error": f"Error: {str(e)}"}), 500

# @app.route("/")
# def home():
#     return "SoulSync API is running!"

# if __name__ == "__main__":
#     app.run(debug=True, host="127.0.0.1", port=5000)

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, exceptions
from pathlib import Path
from dotenv import load_dotenv
import uuid
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

app = Flask(__name__)
CORS(app)

# Azure OpenAI Configuration
client = AzureOpenAI(
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
        "password": password,
        "conversation_history": [],
        "baseline_emotional_state": "neutral"  # Initial state
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

# Define tools for specialized agents
def suggest_therapy_plan(input_text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": (
                    "You are an expert in creating personalized, readable therapy plans. Respond with a warm, supportive, and casual tone without using Markdown (no **, ##, ---, etc.). "
                    "Break the response into short sections with clear titles using natural text and bullet points with indentation (two spaces) for sub-items. Use simple language, add emojis for personality, and include examples to guide the user. "
                    "Provide a customizable therapy plan template with sections for: what’s on your mind, duration, past therapy, goals, daily routine, hobbies, chat frequency, support crew, and next steps. "
                    "Encourage the user to share details and emphasize a collaborative approach."
                )}, {"role": "user", "content": input_text}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in suggest_therapy_plan: {str(e)}")
        return "Sorry, I couldn’t suggest a therapy plan right now."

def check_wellness(status: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Assess wellness , perform Psychometric assesment that is ask Compulsory 5 different questions one by one and then analyze the situation and flag if urgent based on this status."}, {"role": "user", "content": status}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in check_wellness: {str(e)}")
        return "Sorry, I couldn’t assess your wellness right now."

def post_rehab_followup(input_text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Provide post-rehab follow-up advice based on this input."}, {"role": "user", "content": input_text}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in post_rehab_followup: {str(e)}")
        return "Sorry, I couldn’t provide post-rehab advice right now."

# Custom AzureChatOpenAI wrapper
class AzureChatOpenAI:
    def __init__(self, client):
        self.client = client

    def invoke(self, messages):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return {"role": "assistant", "content": response.choices[0].message.content.strip()}
        except Exception as e:
            logger.error(f"Error in AzureChatOpenAI invoke: {str(e)}")
            raise


# Define state schema
class AgentState(TypedDict):
    messages: List[dict]
    next: str
    selected_agent: str
    topics: List[str]  # Added to track extracted topics
    sentiment: str     # Added to track sentiment

# Chat Agent (Conversation Partner)
def create_chat_agent(model):
    def chat_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        last_message = messages[-1]["content"]

        # Generate empathetic response
        empathetic_prompt = [
            {"role": "system", "content": (
                "You are a warm, empathetic conversational partner. Respond to the user's message in a supportive and understanding way. "
                "If the user greets you (e.g., 'hi', 'hello'), respond with a friendly greeting and ask how they’re feeling. "
                "For other messages, provide a supportive response and extract the main topics and sentiment."
            )}
        ] + messages
        try:
            empathetic_response = model.invoke(empathetic_prompt)
            empathetic_message = empathetic_response["content"]
        except Exception as e:
            logger.error(f"Error in chat agent response: {str(e)}")
            empathetic_message = "I’m here for you! Let’s talk about how you’re feeling."

        # Extract topics and sentiment
        analysis_prompt = [
            {"role": "system", "content": (
                "Analyze the user's message and extract the main topics (e.g., 'depression', 'focus issues', 'post-rehab') and sentiment (e.g., 'positive', 'negative', 'neutral'). "
                "Return the result in the format: Topics: [topic1, topic2], Sentiment: sentiment"
            )}
        ] + messages
        try:
            analysis_response = model.invoke(analysis_prompt)
            analysis_result = analysis_response["content"]
            # Parse the result (assuming the model returns it in the expected format)
            topics = analysis_result.split("Topics: ")[1].split(", Sentiment: ")[0].strip("[]").split(", ")
            sentiment = analysis_result.split("Sentiment: ")[1].strip()
        except Exception as e:
            logger.error(f"Error in chat agent analysis: {str(e)}")
            topics = ["unknown"]
            sentiment = "neutral"

        logger.debug(f"Chat Agent extracted: Topics={topics}, Sentiment={sentiment}")

        # Add the empathetic response to the conversation
        updated_messages = messages + [{"role": "assistant", "content": empathetic_message, "agent": "chat_agent"}]
        return {
            "messages": updated_messages,
            "topics": topics,
            "sentiment": sentiment,
            "next": "supervisor",  # Indicate next step for the parent graph
            "selected_agent": "chat_agent"
        }

    graph = StateGraph(AgentState)
    graph.add_node("chat_agent", chat_node)
    graph.set_entry_point("chat_agent")
    # Remove the edge to "supervisor" since it will be handled by the parent graph
    return graph.compile(name="chat_agent")

# Define custom agent logic for specialized agents
def create_custom_agent(model, tool, name, prompt):
    def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        topics = state["topics"]
        sentiment = state["sentiment"]
        system_prompt = [{"role": "system", "content": prompt}]
        tool_prompt = [
            {"role": "system", "content": (
                f"Process the user's input using the tool '{tool.__name__}'. "
                f"User input: {messages[-1]['content']}, Topics: {topics}, Sentiment: {sentiment}"
            )}
        ]
        try:
            response = model.invoke(system_prompt + messages + tool_prompt)
            tool_result = tool(messages[-1]["content"])
            logger.debug(f"Agent {name} processed input: {messages[-1]['content']} -> {tool_result}")
            return {
                "messages": messages + [{"role": "assistant", "content": tool_result, "agent": name}],
                "selected_agent": name
            }
        except Exception as e:
            logger.error(f"Error in agent {name}: {str(e)}")
            return {
                "messages": messages + [{"role": "assistant", "content": f"Sorry, something went wrong with {name}. Try again!"}],
                "selected_agent": name
            }

    graph = StateGraph(AgentState)
    graph.add_node(name, agent_node)
    graph.set_entry_point(name)
    graph.add_edge(name, END)
    return graph.compile(name=name)

# Custom supervisor logic
def create_custom_supervisor(agents, model, prompt, checkpointer):
    def supervisor_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        topics = state["topics"]
        sentiment = state["sentiment"]
        system_prompt = [{"role": "system", "content": prompt}]

        # Prepare the routing prompt with topics and sentiment
        routing_prompt = system_prompt + messages + [
            {"role": "system", "content": (
                "You are a supervisor for a mental health assistant system. Based on the conversation history, extracted topics, and sentiment, decide which agent to route to: "
                "'personalized_plan_update_agent', 'wellness_check_agent', or 'post_rehab_follow_up_agent'. "
                "Return only the agent name. Use the following guidelines:\n"
                "- 'personalized_plan_update_agent' for requests related to therapy suggestions or emotional support. "
                "Examples: Topics=['depression', 'anxiety'], Sentiment='negative'; Topics=['therapy', 'support'], Sentiment='neutral'.\n"
                "- 'wellness_check_agent' for progress monitoring or status updates and or emotional support. "
                "Examples: Topics=['progress', 'wellness','Support'], Sentiment='neutral'; Topics=['focus issues'], Sentiment='negative'.\n"
                "- 'post_rehab_follow_up_agent' for follow-up after rehab, including post-rehab advice or recovery questions. "
                "Examples: Topics=['post-rehab', 'recovery'], Sentiment='positive'; Topics=['rehab'], Sentiment='neutral'.\n"
                f"Current topics: {topics}, Sentiment: {sentiment}. Consider the intent and context of the conversation."
            )}
        ]
        try:
            response = model.invoke(routing_prompt)
            next_agent = response["content"].strip()
            logger.debug(f"Supervisor routing decision: Topics={topics}, Sentiment={sentiment} -> {next_agent}")
        except Exception as e:
            logger.error(f"Error in supervisor routing: {str(e)}")
            return {"messages": messages + [{"role": "assistant", "content": "Sorry, I couldn’t decide which agent to use. Try again!"}], "selected_agent": "supervisor"}

        # Route to the appropriate agent
        if next_agent == "personalized_plan_update_agent":
            return {"next": "personalized_plan_update_agent", "messages": messages, "selected_agent": "personalized_plan_update_agent"}
        elif next_agent == "wellness_check_agent":
            return {"next": "wellness_check_agent", "messages": messages, "selected_agent": "wellness_check_agent"}
        elif next_agent == "post_rehab_follow_up_agent":
            return {"next": "post_rehab_follow_up_agent", "messages": messages, "selected_agent": "post_rehab_follow_up_agent"}
        else:
            logger.warning(f"Unknown agent selected: {next_agent}")
            return {"messages": messages + [{"role": "assistant", "content": "Sorry, I couldn’t determine the right agent for your request."}], "selected_agent": "supervisor"}

    # Build the supervisor graph
    graph = StateGraph(AgentState)
    graph.add_node("chat_agent", agents[0])  # Chat Agent
    graph.add_node("supervisor", supervisor_node)
    for agent in agents[1:]:  # Specialized agents
        graph.add_node(agent.name, agent)
    graph.set_entry_point("chat_agent")
    graph.add_edge("chat_agent", "supervisor")
    graph.add_conditional_edges("supervisor", lambda state: state.get("next", END), {
        "personalized_plan_update_agent": "personalized_plan_update_agent",
        "wellness_check_agent": "wellness_check_agent",
        "post_rehab_follow_up_agent": "post_rehab_follow_up_agent",
        END: END
    })
    for agent in agents[1:]:
        graph.add_edge(agent.name, END)

    return graph.compile(checkpointer=checkpointer)
# Initialize supervisor
def initialize_supervisor():
    model = AzureChatOpenAI(client)

    # Create Chat Agent
    chat_agent = create_chat_agent(model)

    # Create specialized agents
    personalized_plan_update_agent = create_custom_agent(
        model=model,
        tool=suggest_therapy_plan,
        name="personalized_plan_update_agent",
        prompt="You are an expert in creating and updating personalized therapy plans based on user feedback and insights."
    )

    wellness_check_agent = create_custom_agent(
        model=model,
        tool=check_wellness,
        name="wellness_check_agent",
        prompt="You are a wellness monitoring expert. Assess the user's emotional state and provide guidance or escalate if necessary."
    )

    post_rehab_follow_up_agent = create_custom_agent(
        model=model,
        tool=post_rehab_followup,
        name="post_rehab_follow_up_agent",
        prompt="You are a post-rehab follow-up expert. Provide advice and support for users after rehab, focusing on relapse prevention and recovery."
    )

    # Validate agents
    agents = [chat_agent, personalized_plan_update_agent, wellness_check_agent, post_rehab_follow_up_agent]
    for agent in agents:
        if not hasattr(agent, "invoke"):
            raise ValueError(f"Agent {agent} is not properly initialized with an invoke method.")

    try:
        checkpointer = InMemorySaver()
        workflow = create_custom_supervisor(
            agents=agents,
            model=model,
            prompt=(
                "You are SoulSync, a chill AI assistant from the future acting as a supervisor. "
                "Route patient messages to the appropriate agent based on extracted topics and sentiment."
            ),
            checkpointer=checkpointer
        )

        return workflow
    except Exception as e:
        raise RuntimeError(f"Failed to initialize supervisor: {str(e)}")

# Global supervisor instance
app.supervisor = initialize_supervisor()

# Updated chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id")
    user_message = data.get("message", "").strip()

    if not user_id or not user_message:
        return jsonify({"error": "Missing user_id or message"}), 400

    try:
        logger.debug(f"Received chat request: user_id={user_id}, message={user_message}")
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
        user_docs = list(container.query_items(query, enable_cross_partition_query=True))

        if not user_docs:
            logger.warning(f"User not found: {user_id}")
            return jsonify({"error": "User not found"}), 404

        user_doc = user_docs[0]
        conversation_history = user_doc.get("conversation_history", [])

        # Invoke the supervisor with a thread_id
        result = app.supervisor.invoke(
            {"messages": conversation_history + [{"role": "user", "content": user_message}]},
            config={"configurable": {"thread_id": user_id}}
        )
        logger.debug(f"Supervisor result: {result}")

        # Extract the assistant's response and selected agent
        # The last message could be from chat_agent or a specialized agent
        last_message = result["messages"][-1]
        ai_response = last_message["content"]
        selected_agent = last_message.get("agent", "unknown")
        logger.debug(f"Assistant response from {selected_agent}: {ai_response}")

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": ai_response, "agent": selected_agent})
        user_doc["conversation_history"] = conversation_history
        container.upsert_item(user_doc)

        # Return both the response and the selected agent
        return jsonify({"response": ai_response, "agent": selected_agent})

    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"CosmosDB Error: {str(e)}")
        return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route("/")
def home():
    return "SoulSync API is running!"

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)