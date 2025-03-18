# import os
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from openai import AzureOpenAI
# from azure.cosmos import CosmosClient, exceptions
# from pathlib import Path
# from dotenv import load_dotenv
# import uuid

# # Load environment variables (unchanged)
# dotenv_path = Path(__file__).resolve().parent / ".env"
# load_dotenv(dotenv_path=dotenv_path)

# app = Flask(__name__)
# CORS(app)

# # Azure OpenAI Configuration (unchanged)
# client = AzureOpenAI(
#     azure_endpoint=os.getenv("ENDPOINT"),
#     api_key=os.getenv("API_KEY"),
#     api_version="2023-05-15"
# )

# # Azure Cosmos DB Configuration (unchanged)
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

# # RESTORED: Generate summary function
# def generate_summary(conversation_history):
#     """Generate a brief summary of the conversation."""
#     if not conversation_history:
#         return "No previous conversation found."
    
#     summary_prompt = [
#         {"role": "system", "content": "Provide a concise 3-5 line summary of the following conversation."}
#     ]
#     summary_prompt.extend(conversation_history[-10:])  # Limit to last 10 messages

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=summary_prompt
#     )
#     return response.choices[0].message.content.strip()

# # RESTORED: Detect intent function
# def detect_intent(user_message, has_previous):
#     """Use LLM to detect user intent conversationally with improved handling of negations."""
#     intent_prompt = [
#         {"role": "system", "content": """Classify the user's intent based on their message. Respond with one of these categories:
#         - 'recall_past': User wants to know about or reference the previous conversation.
#         - 'continue': User wants to continue the previous conversation.
#         - 'start_fresh': User wants to start a new conversation.
#         - 'general': User is making a general statement or question with no clear intent.
#         Consider negations (e.g., 'no' followed by a request) as part of the intent. Provide only the category name."""},
#         {"role": "user", "content": user_message}
#     ]
    
#     if has_previous:
#         intent_prompt.append({"role": "system", "content": "Note: There is a previous conversation available."})
#     else:
#         intent_prompt.append({"role": "system", "content": "Note: There is no previous conversation available."})

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=intent_prompt,
#         temperature=0.3
#     )
#     intent = response.choices[0].message.content.strip()

#     if "no" in user_message.lower() and intent not in ["recall_past", "start_fresh"]:
#         fallback_prompt = intent_prompt + [
#             {"role": "system", "content": "Re-evaluate the intent, considering 'no' as a potential indicator to reverse the previous action or recall past context. Respond with the category name."}
#         ]
#         fallback_response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=fallback_prompt,
#             temperature=0.3
#         )
#         intent = fallback_response.choices[0].message.content.strip()

#     return intent

# # RESTORED: Get chat response function
# def get_chat_response(user_message, conversation_history, user_id, intent):
#     """Generate a response with a system prompt based on detected intent."""
#     if intent == "recall_past":
#         system_prompt = [
#             {"role": "system", "content": f"You are SoulSync, a chill AI assistant from the future. Summarize the user's past conversation if available, then assist. The user's ID is {user_id}. Use their chat history to recall past vibes. If no relevant history, focus on the new request."}
#         ]
#     elif intent == "continue":
#         system_prompt = [
#             {"role": "system", "content": f"You are SoulSync, a chill AI assistant from the future. Continue the user's previous conversation naturally. The user's ID is {user_id}. Use their chat history to pick up where we left off."}
#         ]
#     elif intent == "start_fresh":
#         system_prompt = [
#             {"role": "system", "content": f"You are SoulSync, a chill AI assistant from the future. Start a new conversation. The user's ID is {user_id}. Ignore past chat history."}
#         ]
#     else:  # general
#         system_prompt = [
#             {"role": "system", "content": f"You are SoulSync, a chill AI assistant from the future. Respond naturally to the user's new request. The user's ID is {user_id}. Only use chat history if they ask about it explicitly (e.g., 'what did we talk about?', 'remember last chat'). Avoid summarizing past conversations unless asked."}
#         ]

#     messages = system_prompt
#     if intent in ["recall_past", "continue"] and conversation_history:
#         messages.extend(conversation_history[-10:])
#     messages.append({"role": "user", "content": user_message})

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         temperature=0.7,
#         frequency_penalty=0.5,
#         presence_penalty=0.5
#     )
#     return response.choices[0].message.content.strip()

# # CHANGED: Updated chat endpoint to use intelligent responses
# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.json
#     user_id = data.get("user_id")
#     user_message = data.get("message", "").strip()

#     if not user_id or not user_message:
#         return jsonify({"error": "Missing user_id or message"}), 400

#     try:
#         query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
#         user_docs = list(container.query_items(query, enable_cross_partition_query=True))

#         if not user_docs:
#             return jsonify({"error": "User not found"}), 404

#         user_doc = user_docs[0]
#         conversation_history = user_doc.get("conversation_history", [])
#         has_previous = bool(conversation_history)

#         # Use intent detection
#         intent = detect_intent(user_message, has_previous)

#         # Generate intelligent response
#         ai_response = get_chat_response(user_message, conversation_history, user_id, intent)

#         # Append new messages to conversation history
#         conversation_history.append({"role": "user", "content": user_message})
#         conversation_history.append({"role": "assistant", "content": ai_response})

#         # Update user document with summary if recalling past
#         user_doc["conversation_history"] = conversation_history
#         user_doc["summary"] = generate_summary(conversation_history) if intent == "recall_past" else user_doc.get("summary", "No summary needed")
#         container.upsert_item(user_doc)

#         return jsonify({"response": ai_response})

#     except exceptions.CosmosHttpResponseError as e:
#         return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
#     except Exception as e:
#         return jsonify({"error": f"Error: {str(e)}"}), 500

# @app.route("/")
# def home():
#     return "SoulSync API is running!"

# if __name__ == "__main__":
#     app.run(debug=True, host="127.0.0.1", port=5000)






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
#             return {"messages": messages + [{"role": "assistant", "content": tool_result}]}
#         except Exception as e:
#             logger.error(f"Error in agent {name}: {str(e)}")
#             return {"messages": messages + [{"role": "assistant", "content": f"Sorry, something went wrong with {name}. Try again!"}]}

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

#         # Explicitly handle greetings
#         greetings = ["hi", "hello", "hii", "hey"]
#         if last_message in greetings:
#             logger.debug(f"Supervisor detected greeting: {last_message} -> therapy_plan_agent")
#             return {"next": "therapy_plan_agent", "messages": messages}

#         routing_prompt = system_prompt + messages + [
#             {"role": "system", "content": "Based on the last user message, decide which agent to route to: 'therapy_plan_agent', 'wellness_check_agent', or 'post_rehab_agent'. Return only the agent name."}
#         ]
#         try:
#             response = model.invoke(routing_prompt)
#             next_agent = response["content"].strip()
#             logger.debug(f"Supervisor routing decision: {last_message} -> {next_agent}")
#         except Exception as e:
#             logger.error(f"Error in supervisor routing: {str(e)}")
#             return {"messages": messages + [{"role": "assistant", "content": "Sorry, I couldn’t decide which agent to use. Try again!"}]}

#         # Route to the appropriate agent
#         if next_agent == "therapy_plan_agent":
#             return {"next": "therapy_plan_agent", "messages": messages}
#         elif next_agent == "wellness_check_agent":
#             return {"next": "wellness_check_agent", "messages": messages}
#         elif next_agent == "post_rehab_agent":
#             return {"next": "post_rehab_agent", "messages": messages}
#         else:
#             logger.warning(f"Unknown agent selected: {next_agent}")
#             return {"messages": messages + [{"role": "assistant", "content": "Sorry, I couldn’t determine the right agent for your request."}]}

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
#                 "Route patient messages to the appropriate agent: "
#                 "- 'therapy_plan_agent' for therapy suggestions, emotional support requests, or greetings like 'hi', 'hello', or 'hii'. "
#                 "- 'wellness_check_agent' for progress monitoring or status updates (e.g., 'how am I doing?', 'I’m feeling okay'). "
#                 "- 'post_rehab_agent' for follow-up after rehab (e.g., 'I finished rehab', 'what’s next after rehab?'). "
#                 "Use conversation history to maintain context."
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
#             config={"configurable": {"thread_id": user_id}}  # Use user_id as thread_id
#         )
#         logger.debug(f"Supervisor result: {result}")

#         # Extract the assistant's response
#         ai_response = result["messages"][-1]["content"] if result["messages"] else "No response generated."
#         logger.debug(f"Assistant response: {ai_response}")

#         # Update conversation history
#         conversation_history.append({"role": "user", "content": user_message})
#         conversation_history.append({"role": "assistant", "content": ai_response})
#         user_doc["conversation_history"] = conversation_history
#         container.upsert_item(user_doc)

#         return jsonify({"response": ai_response})

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
