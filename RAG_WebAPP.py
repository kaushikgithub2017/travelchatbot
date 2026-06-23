import os
import sys
from flask import Flask, request, jsonify, render_template_string
from anthropic import AnthropicFoundry
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------------------------------------------------------
# Configuration
RAG_FILE = "travel_rag.txt"          # your travel knowledge file
TOP_K = 3                            # number of chunks to retrieve
MAX_TOKENS = 1024

# Azure AI Foundry / Anthropic endpoint details
ENDPOINT = "https://kaushikai-23june2026.services.ai.azure.com/anthropic"
DEPLOYMENT_NAME = "claude-sonnet-4-6"

# ----------------------------------------------------------------------
# Step 1: Load and chunk the RAG file
def load_and_chunk(filepath, min_chunk_length=50):
    """Read the file and split into paragraphs, filtering short ones."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = [c for c in chunks if len(c) >= min_chunk_length]
    if not chunks:
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    return chunks

# ----------------------------------------------------------------------
# Step 2: Build TF-IDF index
class SimpleRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query, k=TOP_K):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-k:][::-1]
        return [self.chunks[i] for i in top_indices]

# ----------------------------------------------------------------------
# Step 3: Build the chatbot with Anthropic
class TravelChatbot:
    def __init__(self, retriever):
        self.retriever = retriever
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://ai.azure.com/.default"
        )
        self.client = AnthropicFoundry(
            azure_ad_token_provider=token_provider,
            base_url=ENDPOINT
        )
        self.model = DEPLOYMENT_NAME

    def ask(self, user_query):
        context_chunks = self.retriever.retrieve(user_query)
        context = "\n\n".join(context_chunks)

        system_prompt = (
            "You are a knowledgeable travel assistant. "
            "Answer the user's question **only** using the provided context. "
            "If the context does not contain enough information, say 'I don't know' "
            "and do not make up an answer."
        )

        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {user_query}\n"
            "Answer:"
        )

        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=MAX_TOKENS,
        )
        return response.content[0].text

# ----------------------------------------------------------------------
# Flask Web App
app = Flask(__name__)

# Global objects (initialized on first request or startup)
retriever = None
bot = None

def init_rag():
    global retriever, bot
    if not os.path.exists(RAG_FILE):
        raise FileNotFoundError(f"RAG file '{RAG_FILE}' not found.")
    chunks = load_and_chunk(RAG_FILE)
    print(f"Loaded {len(chunks)} chunks.")
    retriever = SimpleRetriever(chunks)
    bot = TravelChatbot(retriever)

# HTML template (embedded)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Travel RAG Chatbot</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f4f8;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            max-width: 700px;
            width: 100%;
            background: white;
            border-radius: 16px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
            padding: 30px;
            box-sizing: border-box;
        }
        h1 {
            color: #1e293b;
            font-weight: 600;
            margin-top: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        h1 small {
            font-size: 0.6rem;
            font-weight: 400;
            color: #64748b;
        }
        .input-area {
            display: flex;
            gap: 10px;
            margin: 20px 0;
        }
        input[type="text"] {
            flex: 1;
            padding: 14px 18px;
            border: 2px solid #e2e8f0;
            border-radius: 30px;
            font-size: 1rem;
            transition: border-color 0.2s;
            outline: none;
        }
        input[type="text"]:focus {
            border-color: #3b82f6;
        }
        button {
            padding: 14px 28px;
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 30px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
        }
        button:hover {
            background: #2563eb;
        }
        button:active {
            transform: scale(0.97);
        }
        button:disabled {
            background: #94a3b8;
            cursor: not-allowed;
        }
        #response-area {
            margin-top: 25px;
            padding: 20px;
            background: #f8fafc;
            border-radius: 12px;
            border-left: 5px solid #3b82f6;
            white-space: pre-wrap;
            word-wrap: break-word;
            min-height: 80px;
            color: #0f172a;
            line-height: 1.6;
        }
        .loading {
            color: #64748b;
            font-style: italic;
        }
        .error {
            color: #b91c1c;
            background: #fee2e2;
            border-left-color: #b91c1c;
        }
        .footer {
            margin-top: 25px;
            font-size: 0.8rem;
            color: #94a3b8;
            text-align: center;
        }
        .context-info {
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>
            ✈️ Travel Assistant
            <small>RAG with Claude</small>
        </h1>
        <div class="input-area">
            <input type="text" id="queryInput" placeholder="Ask about destinations, culture, visas..." autofocus>
            <button id="askBtn">Ask</button>
        </div>
        <div id="response-area">Your answer will appear here.</div>
        <div class="context-info" id="contextInfo"></div>
        <div class="footer">Powered by Azure AI Foundry · TF‑IDF retrieval</div>
    </div>
    <script>
        const queryInput = document.getElementById('queryInput');
        const askBtn = document.getElementById('askBtn');
        const responseArea = document.getElementById('response-area');
        const contextInfo = document.getElementById('contextInfo');

        async function askQuestion() {
            const query = queryInput.value.trim();
            if (!query) {
                responseArea.textContent = 'Please enter a question.';
                responseArea.className = '';
                return;
            }

            askBtn.disabled = true;
            responseArea.textContent = 'Thinking...';
            responseArea.className = 'loading';
            contextInfo.textContent = '';

            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const data = await res.json();
                if (res.ok) {
                    responseArea.textContent = data.answer;
                    responseArea.className = '';
                    if (data.context_chunks) {
                        contextInfo.textContent = `📚 Retrieved ${data.context_chunks.length} chunks.`;
                    }
                } else {
                    responseArea.textContent = `Error: ${data.error || 'Unknown error'}`;
                    responseArea.className = 'error';
                }
            } catch (err) {
                responseArea.textContent = `Network error: ${err.message}`;
                responseArea.className = 'error';
            } finally {
                askBtn.disabled = false;
                queryInput.focus();
            }
        }

        askBtn.addEventListener('click', askQuestion);
        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') askQuestion();
        });

        // Focus input on load
        queryInput.focus();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask():
    if bot is None:
        return jsonify({'error': 'Chatbot not initialized.'}), 500
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing query.'}), 400
    query = data['query'].strip()
    if not query:
        return jsonify({'error': 'Query cannot be empty.'}), 400

    try:
        answer = bot.ask(query)
        # Also return the retrieved context chunks for display (optional)
        context_chunks = bot.retriever.retrieve(query)
        return jsonify({
            'answer': answer,
            'context_chunks': context_chunks
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------------------------------------------------------------
# Main – start Flask server
def main():
    # Initialize RAG components before starting server
    try:
        init_rag()
        print("✅ RAG chatbot initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)

    # Run Flask (use 0.0.0.0 for external access)
    print("\n🌐 Starting web server at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()