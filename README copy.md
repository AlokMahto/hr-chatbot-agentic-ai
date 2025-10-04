# HR Chatbot with Agentic AI

An intelligent HR assistant powered by Google's Gemini AI, featuring RAG (Retrieval-Augmented Generation) for policy queries, holiday information, and conversational memory using Redis.

## Features

- 🤖 **Agentic AI** - Multi-tool agent that intelligently selects the right tool for each query
- 📚 **RAG System** - Vector-based search through HR policy documents using Pinecone
- 📅 **Holiday Integration** - Real-time holiday information via Calendarific API
- 💬 **Conversational Memory** - Redis-backed chat history for contextual conversations
- 🌐 **Dual Interface** - FastAPI REST API and Streamlit web interface
- 🐳 **Docker Support** - Complete containerization with Docker Compose

## Architecture

The chatbot uses LangChain's agent framework with the following tools:

1. **search_hr_policies** - Searches the HR policy knowledge base using semantic search
2. **get_current_date** - Returns the current date
3. **check_holidays** - Fetches all holidays for a specific year and country
4. **check_today_holiday** - Checks if today is a holiday
5. **get_upcoming_holidays** - Gets the next 5 upcoming holidays

## Tech Stack

- **LLM**: Google Gemini 2.0 Flash (via LangChain)
- **Vector Database**: Pinecone
- **Embeddings**: Google text-embedding-004
- **Memory**: Redis
- **API Framework**: FastAPI
- **UI**: Streamlit
- **Orchestration**: LangChain Agents
- **Containerization**: Docker & Docker Compose

## Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- API Keys:
  - Google AI API Key (for Gemini)
  - Pinecone API Key
  - Calendarific API Key (for holidays)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd hr-chatbot
```

### 2. Set Up Environment Variables

Create or Rename `.env.example` to `.env` file in the project root:

```env
# Google AI
GOOGLE_API_KEY=your_google_api_key

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=hr-vector-search-index

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_CHAT_HISTORY_KEY_PREFIX=chat_history:
REDIS_CHAT_HISTORY_TTL_SECONDS=3600

# Langsmith
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=hr-chatbot

# Holiday API
HOLIDAY_API_KEY=your_calendarific_api_key
```

### 3. Install Dependencies

```bash
pip install -r requirement.txt
```

### 4. Prepare HR Policy Documents

Place your OCR HR policy documents (`.txt` files) in the `Data/Pages_Text/` directory.

### 5. Create Vector Embeddings

Run the embedding script to index your HR policy documents:

```bash
python Embeddings.py
```

This will:
- Connect to Pinecone
- Create the index if it doesn't exist
- Split documents into chunks
- Generate embeddings and upload to Pinecone

## Running the Application

### Option 1: Local Development

#### Start Redis

```bash
docker run -d -p 6379:6379 --name chatbot_memory redis:latest
```

#### Run FastAPI Server

```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

#### Run Streamlit UI

```bash
streamlit run app.py
```

The UI will be available at `http://localhost:8501`

### Option 2: Docker Compose (Recommended)

```bash
docker-compose up --build
```

This will start:
- Redis on port 6379
- FastAPI on port 8000
- Streamlit on port 8501

## API Usage

### Chat Endpoint

**POST** `/chat`

Request:
```json
{
  "query": "What is the leave policy?",
  "session_id": "optional-session-id"
}
```

Response:
```json
{
  "response": "According to our leave policy...",
  "session_id": "session-id-used"
}
```

### Health Check

**GET** `/health`

Returns the status of LLM and Redis services.

### Clear Chat History

**DELETE** `/chat_history/{session_id}`

Clears the conversation history for a specific session.

### Example cURL Commands

```bash
# Chat request
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is todays date?"}'

# Health check
curl http://localhost:8000/health

# Clear chat history
curl -X DELETE "http://localhost:8000/chat_history/your-session-id"
```

## Project Structure

```
hr-chatbot/
│
├── api.py                  # FastAPI application
├── app.py                  # Streamlit UI
├── main.py                 # Core agent logic and initialization
├── utils.py                # Tool implementations (date, holidays)
├── Embeddings.py           # Vector embedding creation script
├── requirement.txt         # Python dependencies
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile             # Docker image definition
├── .env                   # Environment variables
├── .dockerignore          # Docker ignore patterns
│
└── Data/
    └── Pages_Text/        # OCR HR policy documents (.txt files)
    └── HR Policy PDF/        # HR policy documents (.pdf files)
```

## Key Components

### Agent System (`main.py`)

The core agent uses LangChain's `create_tool_calling_agent` with:
- Custom prompt template for HR assistance
- Tool selection logic
- Message history integration
- Error handling

### Tools (`utils.py`)

Four utility tools providing:
- Current date information
- Holiday checking capabilities
- Upcoming holiday forecasting
- Country-specific holiday data

### Vector Store (`Embeddings.py`)

RAG implementation using:
- RecursiveCharacterTextSplitter for chunking
- Google's text-embedding-004 model
- Pinecone for vector storage and similarity search

### Memory (`api.py`, `main.py`)

Redis-backed conversation history:
- Session-based storage
- Configurable TTL
- Automatic cleanup

## Example Queries

Try asking:

- "What is the leave policy?"
- "What's today's date?"
- "Is today a holiday?"
- "Show me upcoming holidays"
- "What are the public holidays in India?"
- "Tell me about the company's remote work policy"
- "How many sick leaves can I take?"

## Configuration

### Agent Settings

Modify in `main.py`:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", 
    temperature=0.7
)
```

### Vector Search

Adjust retrieval parameters in `main.py`:
```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3}  # Number of documents to retrieve
)
```

### Text Chunking

Configure in `Embeddings.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
```