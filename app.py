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

# Basic abusive words list
ABUSIVE_WORDS = ["idiot", "stupid", "dumb", "hate", "shut up", "fool", "damn", "bastard", "crap"]

def contains_abuse(text):
    text = text.lower()
    return any(word in text for word in ABUSIVE_WORDS)

def google_search_with_citations(query, num_results=5, broad=False):
    """Perform Google Custom Search and return results with formatted citations."""
    if not GOOGLE_SEARCH_KEY:
        return [], ""  # Skip search if keys missing

    cx = GOOGLE_SEARCH_CX_BROAD if broad else GOOGLE_SEARCH_CX_RESTRICTED
    if not cx:
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
    for item in data.get("items", []):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        results.append({"title": title, "snippet": snippet, "link": link})
    return results, ""

def extract_types_from_snippets(results, topic=None):
    """Extract types from snippets that mention the topic."""
    types_texts = []
    pattern = re.compile(r"(types|kinds|subtypes|categories) of ([\w\s,]+)", re.IGNORECASE)
    for res in results:
        text_block = (res.get("title", "") + " " + res.get("snippet", "")).lower()
        if topic and topic.lower() not in text_block:
            continue  # skip irrelevant results
        for match in pattern.finditer(res.get("snippet", "")):
            types_str = match.group(2).strip()
            types_texts.append(types_str)
    return "\n".join(types_texts)

def generate_answer_with_sources(messages, results, last_topic=None):
    extracted_types = extract_types_from_snippets(results, topic=last_topic)
    formatted_results_text = ""
    for idx, item in enumerate(results, start=1):
        formatted_results_text += f"[{idx}] {item['title']}\n{item['snippet']}\nSource: {item['link']}\n\n"

    system_prompt = (
        "You are a helpful and knowledgeable medical assistant chatbot. "
        "Provide concise, clear, and medically relevant answers strictly based on the search results. "
        "Replace pronouns like 'it', 'those', 'these' with the last medical topic discussed. "
        "Cite sources like [1], [2], etc. "
        "If the answer is unclear, recommend consulting a healthcare professional.\n\n"
    )
    if extracted_types:
        system_prompt += f"Types or categories extracted from search results:\n{extracted_types}\n\n"

    system_prompt += f"{formatted_results_text}\n"
    openai_messages = [{"role": "system", "content": system_prompt}] + messages

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

def get_last_medical_topic(messages):
    """Return the most recent medical entity discussed."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = msg.get("content", "")
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents if ent.label_ in {"DISEASE","DISORDER","SYMPTOM","CONDITION"}]
            if entities:
                return entities[0].lower()
    return None

def contains_medical_entity(text):
    doc = nlp(text)
    return any(ent.label_ in {"DISEASE","DISORDER","SYMPTOM","CONDITION"} for ent in doc.ents)

def rewrite_query(query, last_topic):
    """Replace pronouns with last_topic if safe."""
    if not last_topic or contains_medical_entity(query):
        return query
    pronouns = ["it","those","these","that","them"]
    pattern = re.compile(r"\b(" + "|".join(pronouns) + r")\b", flags=re.IGNORECASE)
    return pattern.sub(last_topic, query)

@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    data = request.get_json()
    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"answer":"Please provide conversation history as a list of messages.","sources":[]})

    latest_user_message = next((msg.get("content","").strip() for msg in reversed(messages) if msg.get("role")=="user"), None)
    if not latest_user_message:
        return jsonify({"answer":"No user message found in conversation.","sources":[]})

    if contains_abuse(latest_user_message):
        return jsonify({"answer":"I am here to help with medical questions. Please be respectful.","sources":[]})

    greetings = ["hi","hello","hey","good morning","good afternoon","good evening"]
    if latest_user_message.lower() in greetings:
        return jsonify({"answer":"Hi! How may I help you with your medical questions today?","sources":[]})

    last_topic = get_last_medical_topic(messages)
    search_query = rewrite_query(latest_user_message, last_topic)

    results, _ = google_search_with_citations(search_query, num_results=5, broad=False)
    # Strict filter: only keep results mentioning the last topic
    if last_topic:
        results = [r for r in results if last_topic.lower() in (r.get("title","") + r.get("snippet","")).lower()]

    extracted_types = extract_types_from_snippets(results, topic=last_topic)
    answer = generate_answer_with_sources(messages, results, last_topic=last_topic)

    # Handle "types" queries if nothing extracted
    if "type" in latest_user_message.lower() and not extracted_types and last_topic:
        fallback_query = f"types of {last_topic}"
        fallback_results, _ = google_search_with_citations(fallback_query, num_results=10, broad=False)
        fallback_results = [r for r in fallback_results if last_topic.lower() in (r.get("title","")+r.get("snippet","")).lower()]
        answer = generate_answer_with_sources(messages, fallback_results, last_topic=last_topic)
        return jsonify({"answer": answer, "sources": fallback_results})

    return jsonify({"answer": answer, "sources": results})

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "medibot.html")

if __name__=="__main__":
    port = int(os.environ.get("PORT",7000))
    app.run(host="0.0.0.0", port=port)





