import os
import uuid
import redis
import traceback
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from fastapi import FastAPI, HTTPException, Body
from langchain_community.chat_message_histories import RedisChatMessageHistory
from main import ChatResponse, ChatRequest, llm, redis_client_utility, conversation_runnable_with_history

# --- Load environment variables ---
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
REDIS_CHAT_HISTORY_KEY_PREFIX = os.getenv("REDIS_CHAT_HISTORY_KEY_PREFIX")
REDIS_CHAT_HISTORY_TTL_SECONDS = int(os.getenv("REDIS_CHAT_HISTORY_TTL_SECONDS"))

# --- Initialize FastAPI App ---
app = FastAPI(
    title="HR Chatbot API",
    description="API endpoint for HR Assistant with Agentic AI",
    version="1.0.0"
)

# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest = Body(...)):
    if not llm:
        raise HTTPException(status_code=503, detail="LLM service is unavailable.")
    if not REDIS_URL:
        raise HTTPException(status_code=503, detail="Redis service is unavailable.")
    if not conversation_runnable_with_history:
         raise HTTPException(status_code=503, detail="Conversation runnable not initialized.")

    session_id_to_use = request.session_id or str(uuid.uuid4())

    try:
        # Agent returns a dict with 'output' key
        result = conversation_runnable_with_history.invoke(
            input={"input": request.query},
            config={"configurable": {"session_id": session_id_to_use}}
        )

        ai_response_content: str
        
        # Agent executor returns a dict with 'output' key
        if isinstance(result, dict) and 'output' in result:
            ai_response_content = result['output']
        elif isinstance(result, AIMessage):
            ai_response_content = result.content
        elif isinstance(result, str):
            ai_response_content = result
        else:
            print(f"Unexpected AI response format: {type(result)} - {result}")
            raise HTTPException(status_code=500, detail="Error parsing AI response.")

        return ChatResponse(
            response=ai_response_content,
            session_id=session_id_to_use,
        )
    
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error during conversation: {type(e).__name__} - {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=503, detail=f"Chat history service error: {str(e)}")
    except Exception as e:
        print(f"Error during conversation: {type(e).__name__} - {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error during conversation: {str(e)}")

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    llm_ok = bool(llm)
    redis_ok = False
    redis_reason = "Redis not configured or initial connection failed."

    if REDIS_URL and redis_client_utility:
        try:
            if redis_client_utility.ping():
                redis_ok = True
                redis_reason = "Connected successfully."
            else:
                redis_reason = "Ping returned false."
        except redis.exceptions.ConnectionError as e:
            redis_reason = f"Connection test failed: {str(e)}"
        except Exception as e:
            redis_reason = f"Health check failed: {str(e)}"
    
    overall_status = "healthy" if llm_ok and redis_ok else "unhealthy"
        
    return {
        "status": overall_status,
        "details": {
            "llm_service": {
                "status": "healthy" if llm_ok else "unhealthy",
                "reason": "LLM initialized" if llm_ok else "LLM not initialized"
            },
            "redis_service": {
                "status": "healthy" if redis_ok else "unhealthy",
                "reason": redis_reason
            }
        }
    }

@app.delete("/chat_history/{session_id}")
async def clear_chat_history_endpoint(session_id: str):
    if not REDIS_URL or not redis_client_utility:
        raise HTTPException(status_code=503, detail="Redis service not configured or unavailable.")
    
    key_to_check = f"{REDIS_CHAT_HISTORY_KEY_PREFIX}{session_id}"

    try:
        if redis_client_utility.exists(key_to_check):
            history_to_clear = RedisChatMessageHistory(
                session_id=session_id,
                url=REDIS_URL,
                key_prefix=REDIS_CHAT_HISTORY_KEY_PREFIX
            )
            history_to_clear.clear()
            return {"message": f"Chat history for session {session_id} cleared."}
        else:
            raise HTTPException(status_code=404, detail="Session ID not found in chat history.")
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error while clearing history for session {session_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=503, detail=f"Chat history service error: {str(e)}")
    except Exception as e:
        print(f"Error clearing Redis history for session {session_id}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interacting with Redis: {str(e)}")

# --- Run Instructions ---
# 1. Set up environment variables in .env file:
#    - REDIS_URL
#    - HOLIDAY_API_KEY (from calendarific.com)
#    - PINECONE_API_KEY
#    - INDEX_NAME
#    - GOOGLE_API_KEY
#    - REDIS_CHAT_HISTORY_KEY_PREFIX
#    - REDIS_CHAT_HISTORY_TTL_SECONDS
#
# 2. Start Redis using Docker:
#    docker volume create redis-data
#    docker network create agent-network
#    docker run -d --network agent-network -p 6379:6379 --name chatbot_memory -v redis-data:/data redis:latest
#
# 3. Run the FastAPI app:
#    uvicorn api:app --reload
#
# Example usage with curl:
#
# 1. Ask about current date:
#    curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"query": "What is todays date?"}'