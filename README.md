# Text Summarization App

This is a simple and efficient Streamlit application that summarizes YouTube videos and web pages using Groq's language models with the help of LangChain. It's designed to be fast, easy to use, and works well even with longer content.

---

## Features

- Summarize YouTube videos (requires captions)
- Summarize content from websites
- Automatically breaks down long content into manageable chunks
- Uses Groq LLMs for fast and accurate summarization

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo-url>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
### 3. Run the app

```bash
streamlit run app.py
```

---

##  Usage

1. **Enter your Groq API Key** in the sidebar (get one at [Groq Console](https://console.groq.com/keys)).
2. **Paste a YouTube video URL** (with captions) or a website URL.
3. Click **Summarize**.
4. Read your summary!

---
## Notes & Requirements

- **YouTube videos** must have captions/transcripts enabled.
- Private, age-restricted, or region-blocked videos will not work.
- Most readable websites are supported.
- For best results, use with English-language content.

---
