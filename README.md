🎓 Course Advisor Chatbot (Django + LangChain + Ollama)

An intelligent Course Recommendation Chatbot built using Django, LangChain, and Ollama local LLMs.
This project uses Retrieval-Augmented Generation (RAG) to provide accurate suggestions from your custom course dataset.

🚀 Features
✅ AI Chatbot with Course Intelligence

Answers user queries about courses

Provides recommendations based on skills, goals, and course descriptions

✅ RAG Pipeline

Embeds course dataset using Ollama Embeddings

Stores vectors in ChromaDB

Retrieves the most relevant courses for each query

✅ Django-Powered Full Stack App

Session-based chat

Persistent chat history

Clean UI with typing indicator

AJAX-powered smooth conversations

✅ Local LLM (Privacy + Speed)

Uses Ollama Llama 3.2 model

No external API needed

Fast and cost-free

🧠 Tech Stack
Backend	Django
AI Engine	LangChain + Ollama LLM
Vector DB	Chroma
Embeddings	mxbai-embed-large
Frontend	HTML, CSS, JavaScript
Data Format	CSV Course Dataset
🛠️ How It Works

User sends a message

Django backend stores message in database

LangChain retrieves relevant documents using embeddings

LLM (Ollama) generates an answer using:

Chat history

Retrieved course data

Custom system prompt

Response is displayed in UI
