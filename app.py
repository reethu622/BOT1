import os
import re
import subprocess
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

def google_search_with_citations(query, num_results=5, broad=False):
    if not GOOGLE_SEARCH_KEY:
        return [], ""
    cx = GOOGLE_SEARCH_CX_BROAD if broad else GOOGLE_SEARCH_CX_RESTRICTED
    if not cx:
        return [], ""
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
        return [], ""
    results = [
        {"title": item.get("title", ""), "snippet": item.get("snippet", ""), "link": item.get("link", "")}
        for item in data.get("items", [])
    ]
    return results, ""

def is_answer_incomplete(answer_text, user_query):
    answer_lower = answer_text.lower()
    if any(phrase in answer_lower for phrase in ["sorry", "don't know", "cannot find", "need more information"]):
        return True
    question_keywords = ["type", "types", "explain", "list", "what are", "different kinds", "kinds"]
    if any(word in user_query.lower() for word in question_keywords):
        if "type" not in answer_lower and "kind" not in answer_lower and "explain" not in answer_lower:
            return True
    return False

def extract_types_from_snippets(results):
    types_texts = []
    pattern = re.compile(r"(?:types|kinds|subtypes|forms|categories|variants|main types|main forms) (?:of|for)? ([\w\s,/-]+)", re.IGNORECASE)
    for res in results:
        snippet = res.get("snippet", "")
        for match in pattern.finditer(snippet):
            types_texts.append(match.group(1).strip())
    return "\n".join(list(dict.fromkeys(types_texts)))

def generate_answer_with_sources(messages, results, last_topic=None):
    extracted_types = extract_types_from_snippets(results)
    formatted_results_text = ""
    for idx, item in enumerate(results, start=1):
        formatted_results_text += f"[{idx}] {item['title']}\n{item['snippet']}\nSource: {item['link']}\n\n"
    
    system_prompt = (
        "You are a helpful medical assistant. Provide concise, clear, and medically relevant answers. "
        "Cite the most relevant sources from the list below for each fact. You may cite multiple sources per fact if appropriate. "
        "Use the sources provided below and cite them as [1], [2], etc., based on their order in the list. "
        "If the user uses pronouns like 'it', 'those', 'these', 'that', 'this disease', 'the condition', infer they mean the most recent medical topic. "
        "Answer strictly based on the search results.\n\n"
    )
    if last_topic:
        system_prompt += f"Focus on the medical topic: {last_topic}\n\n"
    if extracted_types:
        system_prompt += f"Here are types/categories extracted from search results:\n{extracted_types}\n\n"
    system_prompt += formatted_results_text + "\n"
    
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

def get_last_medical_topic(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = msg.get("content", "")
            doc = nlp(text)
            disease_entities = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
            if disease_entities:
                return disease_entities[0].lower()
    return None

def contains_medical_entity(text):
    doc = nlp(text)
    return any(ent.label_ == "DISEASE" for ent in doc.ents)

def rewrite_query(query, last_topic):
    if not last_topic or contains_medical_entity(query):
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

        # Determine last medical topic
        last_topic = get_last_medical_topic(messages)

        # Decide query: use NER if there’s a medical entity, else just the user query
        if last_topic or contains_medical_entity(latest_user_message):
            search_query = rewrite_query(latest_user_message, last_topic)
        else:
            search_query = latest_user_message  # general health query

        # Google search top 10, filter relevance if medical topic exists
        results, _ = google_search_with_citations(search_query, num_results=10, broad=False)
        if last_topic:
            results = [r for r in results if last_topic.lower() in r["title"].lower() or last_topic.lower() in r["snippet"].lower()]
        results = results[:6]  # top 6

        answer = generate_answer_with_sources(messages, results, last_topic=last_topic)

        # Fallback for types/kinds questions
        if any(word in latest_user_message.lower() for word in ["type", "types", "kind", "kinds"]):
            fallback_query = f"types of {last_topic}" if last_topic else latest_user_message
            fallback_results, _ = google_search_with_citations(fallback_query, num_results=15, broad=True)
            if last_topic:
                fallback_results = [r for r in fallback_results if last_topic.lower() in r["title"].lower() or last_topic.lower() in r["snippet"].lower()]
            fallback_results = fallback_results[:6]
            answer = generate_answer_with_sources(messages, fallback_results, last_topic=last_topic)
            return jsonify({"answer": answer, "sources": fallback_results})

        # Fallback if answer incomplete
        if is_answer_incomplete(answer, latest_user_message):
            fallback_results, _ = google_search_with_citations(search_query, num_results=15, broad=True)
            if last_topic:
                fallback_results = [r for r in fallback_results if last_topic.lower() in r["title"].lower() or last_topic.lower() in r["snippet"].lower()]
            fallback_results = fallback_results[:6]
            answer = generate_answer_with_sources(messages, fallback_results, last_topic=last_topic)
            return jsonify({"answer": answer, "sources": fallback_results})

        # Return answer with all top 6 sources
        return jsonify({"answer": answer, "sources": results})

    except Exception as e:
        print(f"Error in /api/v1/search_answer: {e}")
        return jsonify({"answer": "Internal server error", "sources": []}), 500

# ======================
# Serve static file
# ======================
@app.route("/")
def serve_index():
    try:
        return send_from_directory(app.static_folder, "medibot.html")
    except Exception as e:
        print(f"Error serving static file: {e}")
        return "Index file not found", 404

# ======================
# Run App
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)






