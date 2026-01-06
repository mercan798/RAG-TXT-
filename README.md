# RAG-TXT-

RAG-Enabled AI ChatBot

A Streamlit-based AI assistant that utilizes Retrieval-Augmented Generation (RAG) to reference past conversations. The application automatically indexes local .txt chat logs into a vector database to provide context-aware responses.
Key Features

    Persistent Memory: Automatically reads and indexes saved chat history files.

    Vector Search: Uses OpenAI text-embedding-3-large and ChromaDB for semantic search.

    Dynamic Context Injection: Injects relevant snippets from past logs into the LLM prompt.

    Chat Management: Save current sessions to .txt files or clear history instantly.

    Efficiency: Includes an indexing state tracker to ensure files are only processed once.

The .txt Indexing System Explained

The core functionality of this application is its ability to learn from previous interactions stored as text files. Here is the technical breakdown of how the text processing works:
1. File Detection and State Tracking

The system monitors the base directory for files matching the pattern chat_history_*.txt. To prevent redundant processing and manage API usage, it maintains an index_state.json file.

    New Files: Processed and added to the database.

    Indexed Files: Skipped during the startup boot sequence.

2. Text Chunking

Raw text files are often too large for single embeddings or prompt limits. The chunking_text function breaks these files into manageable segments:

    Chunk Size: 500 characters.

    Overlap: 50 characters. This ensures that semantic context is preserved even if a sentence is split between two chunks.

3. Embedding and Vector Storage

Each text chunk is processed through OpenAI's embedding model, which converts string data into a high-dimensional vector. These vectors are stored in a Persistent ChromaDB client. This allows the bot to perform mathematical similarity searches to find the most relevant past conversations based on the user's current query.
4. Retrieval and Inference

When a user sends a message:

    The application generates an embedding for the user's prompt.

    It queries the ChromaDB collection for the top 3 most similar text chunks.

    These chunks are compiled into a Context block.

    The Context, along with the current message and the recent conversation history, is sent to GPT-4 to generate a final response.

Technical Stack

    Frontend: Streamlit

    Large Language Model: OpenAI GPT-4

    Embeddings: OpenAI text-embedding-3-large

    Vector Database: ChromaDB

    Environment Management: Python-dotenv

Installation and Setup
1. Prerequisites

    Python 3.9 or higher

    OpenAI API Key

2. Installation
Bash

# Clone the repository
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install streamlit openai chromadb python-dotenv requests
