import os
import re
import requests
import openai
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY", "")
GOOGLE_SEARCH_CX_RESTRICTED = os.getenv("GOOGLE_SEARCH_CX_RESTRICTED", "")
GOOGLE_SEARCH_CX_BROAD = os.getenv("GOOGLE_SEARCH_CX_BROAD", "")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__, static_folder="static")
CORS(app)

# Basic abusive words list (expand as needed)
ABUSIVE_WORDS = ["idiot", "stupid", "dumb", "hate", "shut up", "fool", "damn", "bastard", "crap"]

def contains_abuse(text):
    text = text.lower()
    for word in ABUSIVE_WORDS:
        if word in text:
            return True
    return False

def google_search_with_citations(query, num_results=5, cx=None):
    """Perform Google Custom Search and return results with formatted citations."""
    if not GOOGLE_SEARCH_KEY or not cx:
        return [], ""

    params = {
        "key": GOOGLE_SEARCH_KEY,
        "cx": cx,
        "q": query,
        "num": num_results
    }
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Google Search API error: {e}")
        return [], ""

    results = []
    for i, item in enumerate(data.get("items", []), start=1):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        results.append({"title": title, "snippet": snippet, "link": link})
    return results, ""

def is_answer_incomplete(answer_text, user_query):
    """
    Simple heuristic to check if answer is incomplete:
    - If answer contains apology phrases or "I don't know"
    - Or if key question words are missing in answer
    """
    answer_lower = answer_text.lower()
    if any(phrase in answer_lower for phrase in ["sorry", "don't know", "cannot find", "need more information"]):
        return True

    question_keywords = ["type", "types", "explain", "list", "what are", "different kinds", "kinds"]
    if any(word in user_query.lower() for word in question_keywords):
        if "type" not in answer_lower and "kind" not in answer_lower and "explain" not in answer_lower:
            return True

    return False

def generate_answer_with_sources(messages, results):
    """Generate an answer using OpenAI or Gemini based on search results and conversation."""
    # Compose system prompt with web results
    formatted_results_text = ""
    for idx, item in enumerate(results, start=1):
        formatted_results_text += f"[{idx}] {item['title']}\n{item['snippet']}\nSource: {item['link']}\n\n"

    system_prompt = (
        "You are a helpful and knowledgeable medical assistant chatbot. "
        "When the user uses pronouns like 'it', 'those', 'these', or says 'explain that', "
        "infer that they mean the most recent medical topic or condition discussed earlier in the conversation. "
        "Always keep track of conversational context carefully. "
        "Answer the user's questions based on the following web search results. "
        "If you cannot find a clear answer, politely say you don't know and recommend consulting a healthcare professional. "
        "Cite your sources with numbers like [1], [2], etc.\n\n"
        f"{formatted_results_text}\n"
    )

    openai_messages = [{"role": "system", "content": system_prompt}]
    openai_messages.extend(messages)

    if OPENAI_API_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=openai_messages,
                temperature=0.3,
            )
            answer = resp.choices[0].message["content"]
            return answer
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

def get_last_medical_topic(messages):
    medical_terms = [
        "diabetes", "asthma", "cancer", "migraine", "hypertension",
        "glaucoma", "stroke", "arthritis", "depression", "anxiety",
        "covid", "influenza", "pneumonia", "eczema", "alzheimer", "parkinson"
    ]

    for msg in reversed(messages):
        text = msg.get("content", "").lower()
        for term in medical_terms:
            if term in text:
                return term
    return None

def rewrite_query(query, last_topic):
    if not last_topic:
        return query

    pronouns = ["it", "those", "these", "that", "them"]
    pattern = re.compile(r"\b(" + "|".join(pronouns) + r")\b", flags=re.IGNORECASE)
    new_query = pattern.sub(last_topic, query)
    return new_query

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    data = request.get_json()
    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"answer": "Please provide conversation history as a list of messages.", "sources": []})

    latest_user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_user_message = msg.get("content", "").strip()
            break

    if not latest_user_message:
        return jsonify({"answer": "No user message found in conversation.", "sources": []})

    if contains_abuse(latest_user_message):
        polite_response = (
            "I am here to help with medical questions. "
            "Please keep the conversation respectful. How can I assist you today?"
        )
        return jsonify({"answer": polite_response, "sources": []})

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if latest_user_message.lower() in greetings:
        return jsonify({"answer": "Hi! How may I help you with your medical questions today?", "sources": []})

    last_topic = get_last_medical_topic(messages)
    search_query = rewrite_query(latest_user_message, last_topic)

    # 1. Search restricted trusted medical sites first
    results, _ = google_search_with_citations(
        search_query, num_results=5, cx=GOOGLE_SEARCH_CX_RESTRICTED
    )
    answer = generate_answer_with_sources(messages, results)

    # 2. If incomplete answer or no results, search broader trusted medical sites only
    if is_answer_incomplete(answer, latest_user_message) or len(results) == 0:
        fallback_results, _ = google_search_with_citations(
            search_query, num_results=10, cx=GOOGLE_SEARCH_CX_BROAD
        )
        answer = generate_answer_with_sources(messages, fallback_results)
        return jsonify({"answer": answer, "sources": fallback_results})

    return jsonify({"answer": answer, "sources": results})

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "medibot.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)
