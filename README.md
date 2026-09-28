# 🤖 AI Model Advisor

A RAG-based assistant that helps you **understand, compare and choose** between 6 leading AI chatbots: **ChatGPT, Claude, Gemini, DeepSeek, Grok and Perplexity**.

Most chatbots only answer questions. AI Model Advisor also **compares** two chatbots side by side and **recommends** the best one for your use case, with sources for every answer.

## ✨ Features

| Tab | What it does |
|---|---|
| 💬 **Ask** | Ask anything about the 6 chatbots: models, API, pricing, strengths. Answers are grounded only in the knowledge base. |
| ⚖️ **Compare** | Pick any 2 chatbots (and optionally an aspect like "pricing" or "coding") and get a comparison table, key differences and a verdict. Uses metadata-filtered retrieval so each side gets its own context. |
| 🎯 **Recommend** | Describe your use case (budget, language, task) and get the best choice, 2 alternatives and things to keep in mind. |
| 📚 **Sources** | Every answer shows which files and chunks were used, to reduce hallucination. |
| 🎬 **Intro video** | A short animated intro with voice-over (`intro.mp4`). Can be swapped for a HeyGen avatar video. |

## 🏗️ Architecture

```
data/*.txt (6 chatbots + overall comparison)
      │  load + add metadata (bot name, source)
      ▼
RecursiveCharacterTextSplitter (800 chars, 120 overlap)
      ▼
HuggingFace embeddings (all-MiniLM-L6-v2, free, local)
      ▼
FAISS vector store ──► retrieval (normal / filtered by chatbot)
      ▼
Prompt (Ask / Compare / Recommend template) ──► Groq LLM (free API)
      ▼
Streamlit UI (answer + sources)
```

## 🛠️ Tech Stack
Python · LangChain · FAISS · Sentence-Transformers (HuggingFace) · Groq API · Streamlit

## 🚀 How to run

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. install packages
pip install -r requirements.txt

# 3. run
streamlit run app.py
```

Then paste your free **Groq API key** (https://console.groq.com/keys) in the sidebar.
Or set it once: `set GROQ_API_KEY=your_key` (Windows) / `export GROQ_API_KEY=your_key` (Mac/Linux).

> First run downloads the embedding model (~90 MB), so it takes a minute. After that it is fast.
> If an LLM model gives an error, pick another one from the sidebar dropdown.

**Intro video:** `intro.mp4` (animated intro) plays at the top of the app. To use a HeyGen avatar instead, replace it with your HeyGen video using the same file name.

## 🧪 Demo questions

- **Ask:** "What models does Claude offer, and how are they different?"
- **Ask:** "What are the peak hours of the DeepSeek API?"
- **Ask:** "Which chatbots have a free API tier?"
- **Compare:** ChatGPT vs Claude → aspect: "pricing"
- **Compare:** Gemini vs Perplexity → aspect: "web search"
- **Recommend:** "I'm a student learning to code and need a free or very cheap AI API."
- **Recommend:** "I want to build a customer support bot for 10,000 users on a low budget."
- **Out of scope test:** "Who won the IPL?" → bot should say it doesn't have this info (no hallucination).

## 📁 Project structure

```
ai-model-advisor/
├── app.py              # Streamlit app + RAG pipeline
├── requirements.txt
├── README.md
├── intro.mp4           # animated intro video
├── .streamlit/config.toml  # dark theme
├── AI_Model_Advisor_Documentation.pdf
└── data/
    ├── chatgpt.txt
    ├── claude.txt
    ├── gemini.txt
    ├── deepseek.txt
    ├── grok.txt
    ├── perplexity.txt
    └── overview_comparison.txt
```

## 📌 Data note
Knowledge base compiled from the official docs and pricing pages of each provider (links inside each file), **as of 25 September 2026**. AI models and prices change often; update the `.txt` files to refresh the knowledge base. No code change needed.

## 🔮 Future improvements
- Auto-refresh data by scraping official docs weekly
- Cost calculator with sliders
- RAGAS evaluation of answer quality
- Real-time talking avatar (HeyGen LiveAvatar) that speaks the answers
