# LangGraph Chatbot UI 🤖💬

A comprehensive conversational AI application demonstrating **LangGraph state management** with multiple **Streamlit frontend** implementations. Features both standard and streaming chat interfaces powered by the **Qwen 2.5-7B** language model via Hugging Face Inference API.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
  - [langgraph_backend.py](#langgraph_backendpy)
  - [streamlit_frontend.py](#streamlit_frontendpy)
  - [streamlit_frontend_streaming.py](#streamlit_frontend_streamingpy)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [System Architecture](#system-architecture)
- [Configuration](#configuration)
- [API Functions](#api-functions)

---

## 🎯 Project Overview

This project demonstrates a production-ready chatbot architecture using **LangGraph** for backend state management and **Streamlit** for frontend presentation. It includes two distinct frontend implementations:

1. **Standard Chat Interface** - Basic conversation with message streaming
2. **Advanced Streaming Interface** - Enhanced UX with real-time token-by-token response display

The backend uses LangGraph's `StateGraph` to manage conversation state, message history, and LLM inference through a cleanly structured graph workflow.

### Key Objectives
- ✨ Demonstrate LangGraph state management in production applications
- 🧠 Implement multi-turn conversations with full context awareness
- ⚡ Support both standard and streaming inference modes
- 🎨 Provide multiple UI implementations for different use cases
- 🔐 Secure API token handling through environment variables

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────┐     ┌──────────────────────────┐   │
│  │  streamlit_         │     │  streamlit_frontend_     │   │
│  │  frontend.py        │     │  streaming.py            │   │
│  │                     │     │                          │   │
│  │  • Basic Chat UI    │     │  • Advanced Streaming    │   │
│  │  • Message History  │     │  • Token-level Updates   │   │
│  │  • Simple Display   │     │  • Spinner Feedback      │   │
│  │  • Error Handling   │     │  • Helper Functions      │   │
│  └──────────┬──────────┘     └──────────────┬───────────┘   │
│             │                               │               │
│             └───────────┬───────────────────┘               │
│                         │                                    │
└─────────────────────────┼────────────────────────────────────┘
                          │
                    invoke_chatbot()
                  stream_chatbot(messages)
                          │
┌─────────────────────────▼────────────────────────────────────┐
│              BACKEND LOGIC LAYER                            │
├──────────────────────────────────────────────────────────────┤
│                  (langgraph_backend.py)                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. LLM Configuration                               │   │
│  │     └─ HuggingFaceEndpoint (Qwen 2.5-7B)           │   │
│  │     └─ ChatHuggingFace wrapper                      │   │
│  │     └─ Temperature: 0.7 (balanced)                  │   │
│  │                                                      │   │
│  │  2. State Definition                                │   │
│  │     └─ ChatState TypedDict                          │   │
│  │     └─ Messages with add_messages reducer           │   │
│  │                                                      │   │
│  │  3. Graph Nodes                                     │   │
│  │     └─ chat_node: Processes messages & calls LLM   │   │
│  │                                                      │   │
│  │  4. Graph Structure                                 │   │
│  │     └─ START → chat_node → END                     │   │
│  │     └─ InMemorySaver checkpointer                  │   │
│  │                                                      │   │
│  │  5. API Functions                                   │   │
│  │     └─ invoke_chatbot(): Standard mode              │   │
│  │     └─ stream_chatbot(): Streaming mode             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────┬─────────────────────────────────────┘
                          │
                   HTTP/REST API Call
                          │
┌─────────────────────────▼─────────────────────────────────────┐
│              EXTERNAL LLM SERVICE                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Hugging Face Inference API                               │
│  ├─ Model: Qwen/Qwen2.5-7B-Instruct                       │
│  ├─ Provider: Hugging Face Hub                           │
│  ├─ Auth: HUGGINGFACEHUB_API_TOKEN                       │
│  ├─ Capabilities: Text generation, context awareness    │
│  └─ Output: Generated text tokens                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
C12_LANGGRAPG_CHATBOT_UI/
│
├── langgraph_backend.py              # Backend: LangGraph state & LLM logic
├── streamlit_frontend.py             # Frontend: Basic chat interface
├── streamlit_frontend_streaming.py   # Frontend: Advanced streaming interface
├── README.md                         # This file
└── .env (not included, create it)    # Environment variables
```

---

## 📄 File Descriptions

### **langgraph_backend.py**

**Purpose**: Core backend logic implementing LangGraph state management and LLM integration.

**Key Components**:

#### 1. **LLM Setup**
```python
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=hf_token,
    temperature=0.7
)
llm = ChatHuggingFace(llm=llm)
```
- **HuggingFaceEndpoint**: Direct API access to the Qwen 2.5-7B model
- **ChatHuggingFace**: Wrapper that handles message formatting and responses
- **temperature=0.7**: Balanced creativity (0=deterministic, 1=creative)

#### 2. **State Definition**
```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```
- **TypedDict**: Type-safe state structure
- **Annotated**: Adds metadata for message reducer
- **add_messages**: LangChain's built-in reducer that intelligently merges new messages into conversation history
- **BaseMessage**: LangChain's message type supporting HumanMessage, AIMessage, SystemMessage, etc.

#### 3. **Chat Node Function**
```python
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}
```
- **Input**: Current conversation state with all previous messages
- **Processing**: Passes full message history to LLM for context-aware responses
- **Output**: Returns LLM's response as a new message
- **Reducer Logic**: The `add_messages` reducer automatically appends this to the conversation

#### 4. **Graph Definition**
```python
checkpointer = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)
```
- **StateGraph**: Defines the conversation workflow
- **Nodes**: Named processing steps (here: "chat_node")
- **Edges**: Control flow between nodes
- **InMemorySaver**: Stores conversation state by thread_id for multi-session support

#### 5. **API Functions**

**`invoke_chatbot(messages, thread_id="default")`**
- **Type**: Blocking/synchronous
- **Input**: List of BaseMessage objects and unique thread identifier
- **Output**: Full response object with updated state
- **Use Case**: When you need the complete response at once
- **Config**: Uses thread_id for state persistence across calls

**`stream_chatbot(messages, thread_id="default") → Iterator[str]`**
- **Type**: Streaming/generator-based
- **Yields**: Individual text chunks as strings (non-empty only)
- **Processing**: 
  - Iterates through `llm.stream(messages)`
  - Extracts `.content` attribute from chunk objects
  - Filters out empty deltas
- **Use Case**: Real-time token-by-token display for UX feedback
- **Note**: This version uses stateless streaming (no state persistence in stream)

---

### **streamlit_frontend.py**

**Purpose**: Basic Streamlit chat interface with standard message streaming.

**Key Features**:

#### 1. **Page Setup**
```python
st.title("LangGraph Chatbot")
```
- Simple, clear title for the application

#### 2. **Session State Management**
```python
if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []
```
- **Streamlit Session State**: Persists data during user's session (survives reruns)
- **Structure**: List of dictionaries with `{"role": "user"|"assistant", "content": str}`
- **Purpose**: Maintains conversation history across page reruns

#### 3. **Message Display**
```python
for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
```
- **st.chat_message()**: Streamlit's built-in component for chat bubbles
- **Roles**: "user" (left-aligned) or "assistant" (right-aligned)
- **Format**: Displays each message in a chat bubble with appropriate styling

#### 4. **User Input**
```python
user_input = st.chat_input("Type your message here...")
if user_input:
    # Process and display
```
- **st.chat_input()**: Fixed input box at bottom of chat
- **Triggers**: On Enter key press
- **Flow**: Only proceeds if user_input is not empty

#### 5. **Message History Building**
```python
prior_messages = []
for msg in st.session_state["messages_history"]:
    if msg["role"] == "user":
        prior_messages.append(HumanMessage(content=msg["content"]))
    else:
        prior_messages.append(AIMessage(content=msg["content"]))
```
- **Conversion**: Transforms session history into LangChain message objects
- **Type Safety**: Uses proper LangChain message types for LLM compatibility
- **Purpose**: Preserves multi-turn conversation context

#### 6. **Streaming Response**
```python
for delta in stream_chatbot(prior_messages + [HumanMessage(content=user_input)]):
    full_response += delta
    message_placeholder.markdown(full_response + "▌")
```
- **Streaming Loop**: Iterates through text deltas from backend
- **Live Display**: Updates placeholder with accumulated text + cursor symbol "▌"
- **Message_placeholder**: st.empty() container that gets updated without rerun
- **Final Render**: Shows complete response without cursor

#### 7. **Error Handling**
```python
except Exception as e:
    st.error(f"Error getting response: {str(e)}")
```
- **User Feedback**: Displays error messages in red error box
- **Graceful Degradation**: Prevents app crash on LLM errors

---

### **streamlit_frontend_streaming.py**

**Purpose**: Advanced Streamlit interface with enhanced streaming UX, helper functions, and better error handling.

**Key Enhancements Over Basic Version**:

#### 1. **Helper Function: Message Conversion**
```python
def history_to_langchain_messages(history: list[dict]) -> list[BaseMessage]:
    """Convert our session history into LangChain BaseMessage objects."""
    messages: list[BaseMessage] = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages
```
- **Separation of Concerns**: Reusable function instead of inline logic
- **Robustness**: Uses `.get()` with defaults to handle missing keys
- **Maintainability**: Centralized conversion logic
- **Type Safety**: Ensures only recognized roles are converted

#### 2. **Title and State**
```python
st.title("LangGraph Chatbot (Streaming)")
if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []
```
- **Distinct Title**: Differentiates from basic version
- **Identical State**: Uses same session state structure for consistency

#### 3. **Improved Message Display**
```python
for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message.get("content", ""))
```
- **Safe Access**: Uses `.get()` with default empty string
- **Defensive Coding**: Prevents KeyError if content missing

#### 4. **Enhanced Streaming with Spinner**
```python
with st.spinner("Thinking..."):
    for delta in stream_chatbot(prior_messages):
        if not isinstance(delta, str):
            delta = str(delta)
        full_response += delta
        message_placeholder.markdown(full_response + "▌")
```
- **st.spinner()**: Shows "Thinking..." while waiting for LLM
- **Type Checking**: Ensures delta is string (defensive programming)
- **User Feedback**: Provides visual indication of processing
- **Smooth Updates**: Real-time character-by-character display

#### 5. **Final Render Optimization**
```python
message_placeholder.markdown(full_response or " ")
```
- **Fallback**: Displays space if response is empty (prevents empty message)
- **No Cursor**: Final display without "▌" cursor symbol

#### 6. **Comprehensive Error Handling**
```python
except Exception as e:
    st.error(f"Error getting response: {str(e)}")
    st.session_state["messages_history"].append({
        "role": "assistant", 
        "content": f"[Error] {str(e)}"
    })
```
- **Error Logging**: Records error in message history
- **User Awareness**: Shows error details for debugging
- **History Consistency**: Maintains valid state even on failure
- **Recovery**: App remains usable after error

---

## ✨ Key Features

### **1. Multi-Turn Conversations**
- Full context awareness through message history
- All previous messages passed to LLM for coherent responses
- Conversation thread separation via thread_id

### **2. Dual Frontend Implementation**
- **Basic**: Streamlined, straightforward interface
- **Advanced**: Enhanced UX with helper functions and better feedback

### **3. Streaming Support**
- Token-by-token display in both frontends
- Real-time cursor feedback in basic version
- Spinner feedback in advanced version
- Better perceived performance and responsiveness

### **4. LangGraph Integration**
- Professional state management with StateGraph
- In-memory checkpointing for session persistence
- Extensible graph architecture for future enhancements
- Message reducer for intelligent history management

### **5. Error Resilience**
- Try-catch blocks protect against LLM failures
- Graceful error messages to users
- Session state consistency maintenance
- Non-breaking error handling

### **6. Secure Configuration**
- Environment variable-based API token management
- Token loaded via `python-dotenv`
- No hardcoded credentials in source code

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit 1.0+ | Web UI and chat interface |
| **Backend** | LangGraph | State management and workflow |
| **LLM Framework** | LangChain | Message abstraction and LLM interface |
| **LLM Model** | Qwen 2.5-7B-Instruct | Language model provider |
| **LLM API** | Hugging Face Inference | Model hosting and API |
| **Config** | python-dotenv | Environment variable management |
| **Language** | Python 3.8+ | Core implementation |

---

## 🚀 Installation & Setup

### **Prerequisites**
- Python 3.8+
- Hugging Face API token ([Get one here](https://huggingface.co/settings/tokens))
- pip package manager

### **Step 1: Clone Repository**
```bash
cd C12_LANGGRAPG_CHATBOT_UI
```

### **Step 2: Create Virtual Environment (Optional)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install streamlit langchain-huggingface langgraph python-dotenv langchain-core
```

### **Step 4: Configure Environment**
Create a `.env` file in the project root:
```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

### **Step 5: Run Application**

**Option A: Basic Frontend**
```bash
streamlit run streamlit_frontend.py
```

**Option B: Advanced Streaming Frontend**
```bash
streamlit run streamlit_frontend_streaming.py
```

The application will open at `http://localhost:8501`

---

## 💬 Usage Guide

### **Basic Workflow**
1. Start the Streamlit app
2. Type your message in the input box
3. Press Enter to send
4. Watch the response stream in real-time
5. Continue the conversation with full context

### **Example Interactions**
```
User: "What is machine learning?"
Assistant: [Detailed explanation of ML concepts]

User: "Can you give me an example?"
Assistant: [Specific examples building on previous explanation]

User: "How is that different from deep learning?"
Assistant: [Comparison leveraging full conversation context]
```

### **Tips**
- **Context**: Earlier messages influence later responses
- **Adjustment**: Change temperature in backend for different response styles
- **Threading**: Modify thread_id in backend functions to separate conversations
- **Limits**: Be aware of model's context window (typically ~2000-4000 tokens)

---

## ⚙️ Configuration

### **LLM Parameters** (in `langgraph_backend.py`)

```python
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=hf_token,
    temperature=0.7  # Adjust this value
)
```

**Temperature Tuning**:
- **0.0** - Deterministic, repetitive, focused
- **0.5** - Balanced, coherent, creative
- **0.7** - Default, good for general chat (current)
- **1.0** - Very creative, potentially inconsistent
- **>1.0** - Highly unpredictable

### **Model Options**
Replace `repo_id` to use different Hugging Face models:
- `"meta-llama/Llama-2-7b-chat"` - Meta's Llama 2
- `"mistralai/Mistral-7B-Instruct"` - Mistral 7B
- `"NousResearch/Nous-Hermes-2-Mixtral"` - Nous Hermes 2

---

## 🔌 API Functions

### **Backend Functions** (from `langgraph_backend.py`)

#### **1. invoke_chatbot()**
```python
from langgraph_backend import invoke_chatbot
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="Hello!")]
result = invoke_chatbot(messages, thread_id="user_123")
response = result['messages'][-1]  # Last message is AI response
```

**Parameters**:
- `messages` (list[BaseMessage]): Conversation messages
- `thread_id` (str): Unique identifier for conversation thread

**Returns**: Dictionary with keys:
- `messages`: Updated message list with LLM response appended

**Use Cases**: Batch processing, non-interactive applications

---

#### **2. stream_chatbot()**
```python
from langgraph_backend import stream_chatbot
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="Tell me a story")]
for chunk in stream_chatbot(messages, thread_id="user_123"):
    print(chunk, end="", flush=True)
```

**Parameters**:
- `messages` (list[BaseMessage]): Conversation messages
- `thread_id` (str): Unique identifier for conversation thread

**Yields**: Strings (non-empty text deltas only)

**Use Cases**: Interactive chat, real-time display, responsive UI

---

## 🔒 Security Considerations

1. **API Token**: Never commit `.env` file to version control
2. **Token Rotation**: Regenerate tokens periodically
3. **Rate Limiting**: Be aware of Hugging Face API rate limits
4. **Cost**: API calls may incur charges - monitor usage
5. **Data Privacy**: User inputs are sent to external API

---

## 🚀 Future Enhancements

- Add system prompts for persona customization
- Implement conversation export/import
- Add token usage tracking and cost estimation
- Support for local LLMs (Ollama, LM Studio)
- Database persistence instead of in-memory state
- Multi-user support with user authentication
- Advanced retrieval-augmented generation (RAG)

---

## 📝 Notes

- **Streaming Behavior**: The `stream_chatbot()` function yields strings only, filtering empty chunks
- **State Persistence**: Current implementation uses in-memory checkpointer; data is lost on restart
- **Context Window**: Model's maximum context is approximately 2000-4000 tokens depending on model version
- **API Dependency**: Requires active internet connection and valid Hugging Face API token
- **Thread Safety**: Use unique thread_ids for concurrent conversations

---

## 📧 Support & Troubleshooting

### **Common Issues**

**Issue**: "HUGGINGFACEHUB_API_TOKEN not found"
- **Solution**: Create `.env` file with valid token

**Issue**: "Model not found" or "Connection error"
- **Solution**: Check internet connection and Hugging Face API status

**Issue**: Streamlit app not responding
- **Solution**: Check LLM API rate limits; may need to wait or upgrade plan

**Issue**: Slow response times
- **Solution**: Large models are slower; consider smaller variant or reduce context
┌────────────────────▼─────────────────────────────────────────┐
│        EXTERNAL: HUGGING FACE INFERENCE API                │
├──────────────────────────────────────────────────────────────┤
│  • Qwen 2.5-7B Model Inference                            │
│  • Endpoint: api-inference.huggingface.co                 │
│  • Latency: 1-5 seconds per request                       │
│  • Support: Multi-turn conversations                      │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🎨 Frontend Interface
- **Streamlit Dashboard** (`streamlit_frontend.py`)
  - Clean, modern chat interface
  - Real-time message input/output
  - Session-based message history
  - User-friendly message display
  - Error handling and status messages
  - Responsive design for all devices

### 🧠 Backend Capabilities
- **LangGraph Integration**
  - State-based conversation management
  - Message queue processing
  - In-memory state persistence
  - Graph-based workflow execution

- **LLM Processing**
  - Qwen 2.5-7B Instruct model
  - Hugging Face API integration
  - Temperature-controlled responses (0.7)
  - Context-aware answer generation
  - Multi-turn conversation support

- **Message Handling**
  - User message processing
  - AI response generation
  - Message history tracking
  - Error recovery and logging

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Frontend** | Streamlit | 1.58.0 |
| **Backend** | Python, LangGraph | 3.14+, 1.2.4 |
| **LLM** | LangChain, Hugging Face | 1.4.1, 1.2.2 |
| **State Management** | LangGraph StateGraph | 1.2.4 |
| **Message Processing** | LangChain Core | 1.4.1 |
| **Environment** | Python dotenv | 1.2.2 |
| **API Client** | HuggingFace Inference | Native |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Hugging Face API Token ([Get one here](https://huggingface.co/settings/tokens))
- Windows/Linux/Mac

### Step 1: Clone Repository
```bash
git clone https://github.com/roylaxmikanta/HF-ChatHub.git
cd HF-ChatHub
```

### Step 2: Create Virtual Environment
```bash
python -m venv lkr
```

### Step 3: Activate Environment
**Windows (Batch):**
```bash
.\lkr\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
$env:PYTHONIOENCODING='utf-8'
.\lkr\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source lkr/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install streamlit langgraph langchain-core langchain-huggingface python-dotenv
```

### Step 5: Configure API Token
Create a `.env` file in the project root:
```env
HUGGINGFACEHUB_API_TOKEN="your_hf_token_here"
```

---

## 🚀 Usage

### Running the Chatbot

From the **project root** (the folder that contains `C12_LANGGRAPG_CHATBOT_UI/`), run:

**Start Streamlit Application (main UI):**
```bash
streamlit run C12_LANGGRAPG_CHATBOT_UI/streamlit_frontend.py
```

**Access the application:**
```text
http://localhost:8501
```

**(Optional) Start Streamlit Application (streaming-focused UI):**
```bash
streamlit run C12_LANGGRAPG_CHATBOT_UI/streamlit_frontend_streaming.py
```
http://localhost:8501
streamlit run C12_LANGGRAPG_CHATBOT_UI/streamlit_frontend.py

### Features:
✅ Real-time chat interface  
✅ Message history within session  
✅ Instant AI responses  
✅ Error handling and recovery  
✅ Clean, intuitive UI

---

## 📂 Project Structure

```
HF-ChatHub/
├── 📄 streamlit_frontend.py       # Main Streamlit application
├── 📄 langgraph_backend.py        # Backend logic & LLM integration
├── 📄 requirements.txt            # Python dependencies
├── 📄 .env                        # Environment variables (API token)
├── 📄 .gitignore                  # Git ignore rules
├── 📄 LICENSE                     # Project license
├── 📄 README.md                   # This file
├── lkr/                           # Python virtual environment
│   ├── Scripts/
│   │   ├── activate.bat
│   │   ├── Activate.ps1
│   │   └── python
│   └── Lib/
│       └── site-packages/
└── __pycache__/                   # Python cache files

### File Descriptions

**streamlit_frontend.py**
- Streamlit UI component
- Message input/output handling
- Session state management
- Real-time chat display

**langgraph_backend.py**
- LangGraph StateGraph setup
- ChatState definition
- chat_node function (LLM interface)
- invoke_chatbot function (main API)
- Hugging Face LLM configuration

**requirements.txt**
- Project dependencies
- Python package versions
- Installation manifest
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the **project root** (same folder where you run Streamlit):

```env
# Required: Hugging Face API Token
HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"
```
HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"

**How to get your API token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with read access
3. Copy the token
4. Paste it in `.env` file
5. Restart the application

### LLM Parameters

Edit `langgraph_backend.py` to customize:

```python
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",  # Change model
    huggingfacehub_api_token=hf_token,
    temperature=0.7                       # Adjust creativity (0-1)
)
```

**Temperature Explanation:**
- `0.0` - Deterministic, always same response
- `0.5` - Balanced creativity and consistency
- `0.7` - Default, good for general chat
- `1.0` - Maximum creativity and randomness

### Supported Models

You can use other Hugging Face models. Some alternatives:

```python
# Meta's Llama
repo_id="meta-llama/Llama-2-7b-chat"

# Mistral
repo_id="mistralai/Mistral-7B-Instruct-v0.1"

# Other Qwen models
repo_id="Qwen/Qwen2-72B-Instruct"
repo_id="Qwen/Qwen-1.8B-Chat"
```

---

## 👨‍💻 Development

### Project Architecture

```
streamlit_frontend.py
├── Import Backend Functions
│   └── invoke_chatbot(messages)
│
├── Streamlit UI Components
│   ├── st.title()
│   ├── st.chat_message()
│   ├── st.chat_input()
│   └── st.error()
│
└── Session State Management
    └── st.session_state (message history)

langgraph_backend.py
├── LLM Configuration
│   └── HuggingFaceEndpoint
│       └── ChatHuggingFace wrapper
│
├── ChatState Definition
│   └── TypedDict with messages
│
├── chat_node Function
│   ├── Extract messages
│   ├── Call LLM
│   └── Return response
│
├── StateGraph Setup
│   ├── Create graph
│   ├── Add chat_node
│   ├── Define edges
│   └── Compile graph
│
└── invoke_chatbot Function
    ├── Build message history
    ├── Call graph.invoke()
    └── Return AI response
```

### Adding Custom Logic

**Example: Add message preprocessing**

```python
def preprocess_message(text: str) -> str:
    """Clean and prepare user input"""
    return text.strip().lower()

# In streamlit_frontend.py
if user_input:
    cleaned_input = preprocess_message(user_input)
    response = invoke_chatbot([HumanMessage(content=cleaned_input)])
```

**Example: Add response filtering**

```python
def filter_response(response: str) -> str:
    """Clean AI response"""
    return response.strip()

# In streamlit_frontend.py
ai_response = filter_response(response["messages"][-1].content)
```

### Testing the Application

Create a test file `test_chatbot.py`:

```python
from langgraph_backend import invoke_chatbot
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

# Test the chatbot
messages = [HumanMessage(content="What is machine learning?")]
response = invoke_chatbot(messages)
print(response["messages"][-1].content)
```

Run tests:
```bash
python test_chatbot.py
```

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError` for langgraph_backend
**Solution:** Ensure you're running from the project root and the module is imported correctly
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

### Issue: Hugging Face API Token Error
**Solution:** 
1. Generate token: https://huggingface.co/settings/tokens
2. Copy token to `.env` file
3. Verify file is in project root
4. Restart the application

### Issue: No module named 'streamlit'
**Solution:** Install Streamlit
```bash
pip install streamlit
```

### Issue: Port 8501 already in use
**Solution:** Run Streamlit on a different port
```bash
streamlit run streamlit_frontend.py --server.port 8502
```

### Issue: Model loading timeout
**Solution:** The first request may take 1-2 minutes for model loading. This is normal. Subsequent requests will be faster.

### Issue: Empty response from LLM
**Solution:**
1. Check your API token is valid
2. Check internet connection
3. Check Hugging Face service status
4. Try with a simpler prompt

### Issue: Memory errors with large conversations
**Solution:** The session history grows with each message. For long sessions:
- Clear session state: Press "C" in Streamlit to clear cache
- Or restart the application
- Or implement conversation pruning

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 📞 Support

For issues, questions, or suggestions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Architecture](#architecture) diagram
3. Check LangGraph documentation: https://langchain-ai.github.io/langgraph/
4. Hugging Face API docs: https://huggingface.co/inference-api

---

## 🌟 Future Enhancements

- [ ] Persistent conversation storage (JSON/Database)
- [ ] User authentication system
- [ ] Multiple LLM model support with switching
- [ ] Conversation export (PDF/CSV/TXT)
- [ ] Conversation search and filtering
- [ ] Docker containerization
- [ ] Web deployment (Hugging Face Spaces)
- [ ] Voice input/output support
- [ ] Advanced analytics dashboard
- [ ] Rate limiting and usage tracking
- [ ] Custom system prompts
- [ ] Conversation sharing feature
- [ ] Multi-language support

---

## 📞 Support & Resources

For issues, questions, or contributions:

1. **Documentation**
   - [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
   - [Streamlit Docs](https://docs.streamlit.io/)
   - [Hugging Face API Docs](https://huggingface.co/docs/api-inference)

2. **Troubleshooting**
   - Check the [Troubleshooting](#troubleshooting) section
   - Review the [Configuration](#configuration) guide
   - Verify `.env` file setup

3. **Community**
   - GitHub Issues: [Report bugs](https://github.com/roylaxmikanta/HF-ChatHub/issues)
   - GitHub Discussions: [Ask questions](https://github.com/roylaxmikanta/HF-ChatHub/discussions)

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📊 Project Stats

- **Language**: Python 3.14+
- **Frontend**: Streamlit
- **Backend**: LangGraph
- **LLM**: Hugging Face (Qwen 2.5-7B)
- **License**: MIT
- **Status**: Active Development

---

**Made with ❤️ using LangGraph, LangChain, and Streamlit**

**Last Updated**: June 6, 2026
