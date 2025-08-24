import os
import re
import requests
import openai
import google.generativeai as genai
import spacy
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ======================
# Load API keys
# ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY", "")
GOOGLE_SEARCH_CX_RESTRICTED = os.getenv("GOOGLE_SEARCH_CX_RESTRICTED", "")
GOOGLE_SEARCH_CX_BROAD = os.getenv("GOOGLE_SEARCH_CX_BROAD", "")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ======================
# Initialize Flask
# ======================
app = Flask(__name__, static_folder="static")
CORS(app)

# ======================
# Load Biomedical NER Model
# ======================
try:
    nlp = spacy.load("en_ner_bc5cdr_md")
except OSError:
    import subprocess
    subprocess.run([
        "python", "-m", "pip", "install",
        "en-ner-bc5cdr-md @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
    ], check=True)
    nlp = spacy.load("en_ner_bc5cdr_md")

# ======================
# Helper Variables
# ======================
ABUSIVE_WORDS = ["idiot", "stupid", "dumb", "hate", "shut up", "fool", "damn", "bastard", "crap"]
GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

# ======================
# Helper Functions
# ======================
def contains_abuse(text):
    text = text.lower()
    return any(word in text for word in ABUSIVE_WORDS)

def google_search_with_citations(query, num_results=10, broad=False):
    if not GOOGLE_SEARCH_KEY:
        return []
    cx = GOOGLE_SEARCH_CX_BROAD if broad else GOOGLE_SEARCH_CX_RESTRICTED
    if not cx:
        return []
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": GOOGLE_SEARCH_KEY, "cx": cx, "q": query, "num": num_results},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Google Search API error: {e}")
        return []
    results = []
    for i, item in enumerate(data.get("items", []), start=1):
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", "")
        })
    return results[:6]  # always return top 6

def is_answer_incomplete(answer_text, user_query):
    answer_lower = answer_text.lower()
    if any(phrase in answer_lower for phrase in ["sorry", "don't know", "cannot find", "need more information"]):
        return True
    question_keywords = ["type", "types", "kind", "kinds", "explain", "list", "what are"]
    if any(word in user_query.lower() for word in question_keywords):
        if "type" not in answer_lower and "kind" not in answer_lower and "explain" not in answer_lower:
            return True
    return False

def generate_answer_with_sources(messages, results, last_topic=None):
    formatted_results_text = ""
    for idx, item in enumerate(results, start=1):
        formatted_results_text += f"[{idx}] {item['title']}\n{item['snippet']}\nSource: {item['link']}\n\n"

    system_prompt = (
        "You are a helpful medical and health assistant. Provide concise, clear answers. "
        "Use the search results provided below to answer the user's question. "
        "Cite the most relevant sources as [1], [2], etc., based on their order in the list. "
        "You may cite multiple sources per fact. "
        "If the user uses pronouns like 'it', 'those', 'these', 'that', 'this disease', 'the condition', infer they mean the most recent topic mentioned by the user. "
        "Answer based on the search results. "
        "Do not mention missing sources or include notes about lacking search results. "
        "You can answer general health topics, not just diseases.\n\n"
    )
    if last_topic:
        system_prompt += f"Focus on the topic: {last_topic}\n\n"
    system_prompt += formatted_results_text

    openai_messages = [{"role": "system", "content": system_prompt}]
    openai_messages.extend(messages)

    if OPENAI_API_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=openai_messages,
                temperature=0.3,
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            if "quota" not in str(e).lower():
                return f"OpenAI error: {e}"

    if GEMINI_API_KEY:
        try:
            conversation_text = system_prompt + "\nConversation:\n"
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_text += f"{role}: {msg['content']}\n"
            conversation_text += "Assistant:"
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(conversation_text)
            return resp.text
        except Exception as e:
            return f"Gemini error: {e}"

    return "I don't know. Please consult a medical professional."

def get_last_topic(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = msg.get("content", "")
            doc = nlp(text)
            disease_entities = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
            if disease_entities:
                return disease_entities[0].lower()
            # fallback: first noun/proper noun
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"]:
                    return token.text.lower()
    return None

def rewrite_query(query, last_topic):
    if not last_topic or any(ent.label_ == "DISEASE" for ent in nlp(query).ents):
        return query
    pattern = re.compile(r"\b(it|this|that|these|those|them|the disease|the condition)\b", flags=re.IGNORECASE)
    return pattern.sub(last_topic, query)

# ======================
# API Route
# ======================
@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    try:
        data = request.get_json()
        messages = data.get("messages")
        if not messages or not isinstance(messages, list):
            return jsonify({"answer": "Please provide conversation history as a list of messages.", "sources": []})

        latest_user_message = next((msg.get("content", "").strip() for msg in reversed(messages) if msg.get("role") == "user"), None)
        if not latest_user_message:
            return jsonify({"answer": "No user message found in conversation.", "sources": []})

        if contains_abuse(latest_user_message):
            return jsonify({"answer": "I am here to help with medical questions. Please keep the conversation respectful.", "sources": []})

        if latest_user_message.lower() in GREETINGS:
            return jsonify({"answer": "Hi! How may I help you with your medical questions today?", "sources": []})

        # Determine last topic only for pronouns
        last_topic = get_last_topic(messages)
        search_query = rewrite_query(latest_user_message, last_topic) if last_topic else latest_user_message

        # Always perform a fresh search for the current question
        results = google_search_with_citations(search_query, num_results=10)
        answer = generate_answer_with_sources(messages, results, last_topic=last_topic)

        # Fallback for types/kinds questions: always search the current question fresh
        if any(word in latest_user_message.lower() for word in ["type", "types", "kind", "kinds"]):
            fallback_results = google_search_with_citations(latest_user_message, num_results=15)
            answer = generate_answer_with_sources(messages, fallback_results)
            return jsonify({"answer": answer, "sources": fallback_results})

        # Fallback if answer incomplete
        if is_answer_incomplete(answer, latest_user_message):
            fallback_results = google_search_with_citations(search_query, num_results=15)
            answer = generate_answer_with_sources(messages, fallback_results, last_topic=last_topic)
            return jsonify({"answer": answer, "sources": fallback_results})

        return jsonify({"answer": answer, "sources": results})

    except Exception as e:
        print(f"Error in /api/v1/search_answer: {e}")
        return jsonify({"answer": "Internal server error", "sources": []}), 500

@app.route("/")
def serve_index():
    try:
        return send_from_directory(app.static_folder, "medibot.html")
    except Exception as e:
        print(f"Error serving static file: {e}")
        return "Index file not found", 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)
