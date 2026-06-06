# HF-ChatHub 🤖💬

A modern, lightweight chatbot application powered by **LangGraph** and **Hugging Face LLMs** with real-time chat interaction.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Development](#development)

---

## 🎯 Overview

**HF-ChatHub** is a conversational AI application built with **Streamlit** that provides an intuitive user interface for interacting with the **Qwen 2.5-7B** large language model through the Hugging Face API. The application uses **LangGraph** for advanced state management and message processing.

### Key Highlights
- ✨ **Streamlit Interface**: Modern, responsive UI with real-time updates
- 🧠 **Advanced LLM**: Qwen 2.5-7B via Hugging Face Inference API
- 🔄 **State Management**: LangGraph-powered conversation management
- ⚡ **Real-time Chat**: Instant message processing and responses
- 🔐 **Secure**: Environment-based API token configuration

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Streamlit Dashboard (streamlit_frontend.py)               │
│   ├─ Chat Interface                                        │
│   ├─ Message History                                       │
│   ├─ Session Management                                   │
│   └─ Real-time Message Display                            │
│                                                              │
│   URL: http://localhost:8501                              │
│                                                              │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     │ invoke_chatbot(message)
                     │
┌────────────────────▼─────────────────────────────────────────┐
│              BACKEND LOGIC LAYER                            │
├──────────────────────────────────────────────────────────────┤
│                  (langgraph_backend.py)                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │    LangGraph State Management                        │   │
│  │                                                      │   │
│  │  ┌────────────────────────────────────────────────┐ │   │
│  │  │  ChatState (TypedDict)                        │ │   │
│  │  │  • messages: List[BaseMessage]               │ │   │
│  │  └────────────────────────────────────────────────┘ │   │
│  │                                                      │   │
│  │  ┌────────────────────────────────────────────────┐ │   │
│  │  │  StateGraph Configuration                    │ │   │
│  │  │  • START → chat_node → END                  │ │   │
│  │  │  • In-memory checkpointer                   │ │   │
│  │  │  • Message history tracking                 │ │   │
│  │  └────────────────────────────────────────────────┘ │   │
│  │                                                      │   │
│  │  ┌────────────────────────────────────────────────┐ │   │
│  │  │  Chat Node (Inference)                       │ │   │
│  │  │  • Process user messages                    │ │   │
│  │  │  • Call LLM for response generation         │ │   │
│  │  │  • Return AI response                       │ │   │
│  │  └────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     │ LLM API Call
                     │ (HuggingFaceEndpoint)
                     │
┌────────────────────▼─────────────────────────────────────────┐
│              LLM SERVICE LAYER                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  LangChain + Hugging Face Integration                      │
│  ├─ Model: Qwen/Qwen2.5-7B-Instruct                       │
│  ├─ Provider: Hugging Face Inference API                 │
│  ├─ Temperature: 0.7 (balanced output)                   │
│  ├─ Auth: HUGGINGFACEHUB_API_TOKEN                       │
│  └─ Response: Context-aware AI text generation           │
│                                                              │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     │ HTTPS/REST Call
                     │
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

**Start Streamlit Application:**
```bash
streamlit run streamlit_frontend.py
```

**Access the application:**
```
http://localhost:8501
```

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

Create a `.env` file in the project root:

```env
# Required: Hugging Face API Token
HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"
```

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
