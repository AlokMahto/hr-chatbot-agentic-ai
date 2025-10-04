import os
from typing import List, Optional

import redis
from pinecone import Pinecone
from dotenv import load_dotenv
from pydantic import BaseModel
from utils import get_all_tools

from langchain.tools import Tool
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- Load environment variables ---
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
REDIS_CHAT_HISTORY_KEY_PREFIX = os.getenv("REDIS_CHAT_HISTORY_KEY_PREFIX")
REDIS_CHAT_HISTORY_TTL_SECONDS = int(os.getenv("REDIS_CHAT_HISTORY_TTL_SECONDS"))
HOLIDAY_API_KEY = os.getenv("HOLIDAY_API_KEY")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME"))

# --- LLM Initialization ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

# --- Redis Initialization ---
redis_client_utility: Optional[redis.Redis] = None

try:
    if REDIS_URL:
        redis_client_utility = redis.from_url(REDIS_URL)
        redis_client_utility.ping()
        print("Redis connection successful!")
except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_client_utility = None

# --- Pydantic Models for Request and Response ---
class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

# --- Helper function to format retrieved documents ---
def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No context found."
    return "\n\n".join(doc.page_content for doc in docs)

# --- RAG Tool for Policy Retrieval ---
def search_hr_policies(query: str) -> str:
    """
    Search the HR policy knowledge base for relevant information.
    
    Args:
        query: The search query
        
    Returns:
        str: Relevant policy information
    """
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 3}
        )
        docs = retriever.invoke(query)
        return format_docs(docs)
    except Exception as e:
        return f"Error searching policies: {str(e)}"

# Create RAG tool
policy_search_tool = Tool(
    name="search_hr_policies",
    func=search_hr_policies,
    description="Search the HR policy knowledge base for information about company policies, leave policies, benefits, procedures, and guidelines. Use this when the user asks about HR policies, leave rules, company procedures, or any policy-related questions."
)

# --- Agent Prompt Template ---
agent_prompt_template_str = """
    You are an advanced HR policy assistant with access to multiple tools. Your role is to:

    1. Answer HR policy questions using the policy knowledge base
    2. Provide current date and holiday information when asked
    3. Help HR professionals with policy interpretation and compliance

    **Decision Making Process:**
    - For questions about HR policies, procedures, leave rules, benefits → Use the 'search_hr_policies' tool
    - For questions about today's date, current date → Use the 'get_current_date' tool
    - For questions about holidays, public holidays, holiday list → Use the 'check_holidays' tool
    - For questions about whether today is a holiday → Use the 'check_today_holiday' tool
    - For questions about upcoming holidays, next holidays → Use the 'get_upcoming_holidays' tool

    **Important Guidelines:**
    - Always think step-by-step before choosing a tool
    - Use the most specific tool for the query
    - If a question requires multiple tools, use them sequentially
    - Always provide clear, professional, and accurate responses
    - If information is not available, clearly state that and suggest alternatives
    - Cite specific policies when using information from the knowledge base

    Be helpful, accurate, and professional in all interactions.
"""

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", agent_prompt_template_str),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- Initialize Agent ---
conversation_runnable_with_history: Optional[RunnableWithMessageHistory] = None

if llm and REDIS_URL:
    # Get all tools
    all_tools = get_all_tools() + [policy_search_tool]
    
    # Create agent
    agent = create_tool_calling_agent(llm, all_tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    print("Initialized Agent with tools:")
    for tool in all_tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")
    
    def get_redis_session_history(session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            session_id=session_id,
            url=REDIS_URL,
            key_prefix=REDIS_CHAT_HISTORY_KEY_PREFIX,
            ttl=REDIS_CHAT_HISTORY_TTL_SECONDS
        )
    
    conversation_runnable_with_history = RunnableWithMessageHistory(
        runnable=agent_executor,
        get_session_history=get_redis_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    print("Initialized RunnableWithMessageHistory with Agent and RedisChatMessageHistory.")
else:
    if not llm:
        print("CRITICAL: LLM not initialized. Conversation runnable cannot be created.")
    if not REDIS_URL:
        print("CRITICAL: Redis not configured or unavailable. Conversation runnable cannot be created.")
    conversation_runnable_with_history = None