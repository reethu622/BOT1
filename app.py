import os
import re
import requests
import openai
import google.generativeai as genai
import spacy
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

# Load scispaCy model once
nlp = spacy.load("en_core_sci_sm")

ABUSIVE_WORDS = ["idiot", "stupid", "dumb", "hate", "shut up", "fool", "damn", "bastard", "crap"]

# -----------------------------------
# Helper functions for topic and query handling
# -----------------------------------

def contains_abuse(text):
    text = text.lower()
    return any(word in text for word in ABUSIVE_WORDS)


def get_last_topic(messages):
    """
    Detect last mentioned medical or insurance topic from user messages using NER.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = msg.get("content", "")
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents if ent.label_ in {"DISEASE","DISORDER","SYMPTOM","CONDITION","ORG"}]
            if entities:
                return entities[0].lower()
    return None


def rewrite_query(query, last_topic):
    """
    Replace pronouns with last_topic and handle 'types of it' queries.
    """
    if not last_topic:
        return query

    doc = nlp(query)
    # If query already has a medical/insurance entity, don't replace
    if any(ent.label_ in {"DISEASE","DISORDER","SYMPTOM","CONDITION","ORG"} for ent in doc.ents):
        return query

    pronouns = ["it","those","these","that","them"]
    pattern = re.compile(r"\\b(" + "|".join(pronouns) + r")\\b", flags=re.IGNORECASE)
    rewritten = pattern.sub(last_topic, query)

    # Handle 'types of it' explicitly
    if "type" in rewritten.lower() and last_topic:
        rewritten = f"types of {last_topic}"

    return rewritten


def google_search_with_citations(query, num_results=5, broad=False):
    if not GOOGLE_SEARCH_KEY:
        return [], ""
    cx = GOOGLE_SEARCH_CX_BROAD if broad else GOOGLE_SEARCH_CX_RESTRICTED
    if not cx:
        return [], ""
    params = {"key": GOOGLE_SEARCH_KEY, "cx": cx, "q": query, "num": num_results}
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Google Search API error: {e}")
        return [], ""
    results = []
    for i, item in enumerate(data.get("items", []), start=1):
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", "")
        })
    return results, ""

# -----------------------------------
# Flask route
# -----------------------------------

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    data = request.get_json()
    messages = data.get("messages")

    if not messages or not isinstance(messages, list):
        return jsonify({"answer": "Please provide conversation history as a list of messages.", "sources": []})

    latest_user_message = next((msg.get("content", "").strip() for msg in reversed(messages) if msg.get("role") == "user"), None)
    if not latest_user_message:
        return jsonify({"answer": "No user message found in conversation.", "sources": []})

    if contains_abuse(latest_user_message):
        return jsonify({"answer": "I am here to help with medical or insurance questions. Please keep the conversation respectful.", "sources": []})

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if latest_user_message.lower() in greetings:
        return jsonify({"answer": "Hi! How may I help you with your medical or insurance questions today?", "sources": []})

    # Detect last topic
    last_topic = get_last_topic(messages)

    # Rewrite query to handle pronouns + types
    search_query = rewrite_query(latest_user_message, last_topic)

    # Search using Google CSE / SearchKey
    results, _ = google_search_with_citations(search_query, num_results=5, broad=False)

    # Generate answer using OpenAI (or Gemini fallback)
    sources_text = "\n".join([f"[{i+1}] {r['title']}\n{r['snippet']}\nSource: {r['link']}" for i, r in enumerate(results)])
    prompt = f"""
    You are a helpful medical + insurance assistant chatbot.
    User asked: {latest_user_message}

    Use the following search results to answer, and cite sources [1],[2], etc.:
    {sources_text}
    """

    answer = "I don't know. Please consult a professional."
    if OPENAI_API_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are Medibot."}, {"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = resp.choices[0].message["content"]
        except Exception as e:
            print(f"OpenAI error: {e}")

    return jsonify({"answer": answer, "sources": results})

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "medibot.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)


