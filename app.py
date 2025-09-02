import os
import re
import requests
import openai
import google.generativeai as genai
import spacy
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from better_profanity import profanity  # Lightweight automatic profanity detection

# ------------------ API Keys ------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY", "")
GOOGLE_SEARCH_CX_RESTRICTED = os.getenv("GOOGLE_SEARCH_CX_RESTRICTED", "")
GOOGLE_SEARCH_CX_BROAD = os.getenv("GOOGLE_SEARCH_CX_BROAD", "")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ------------------ Flask ------------------
app = Flask(__name__, static_folder="static")
CORS(app)

# ------------------ NLP ------------------
nlp = spacy.load("en_core_sci_sm")

# ------------------ Profanity ------------------
profanity.load_censor_words()

def contains_abuse(text):
    return profanity.contains_profanity(text)

# ------------------ Google Search ------------------
def google_search_with_citations(query, num_results=5, broad=False, topic=None):
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
    for item in data.get("items", []):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        # 🔥 filter: must mention topic if provided
        if topic and topic.lower() not in (title + snippet + link).lower():
            continue
        results.append({
            "title": title,
            "snippet": snippet,
            "link": link
        })
    return results, ""

# ------------------ Utility Functions ------------------
def is_answer_incomplete(answer_text, user_query):
    answer_lower = answer_text.lower()
    if any(phrase in answer_lower for phrase in ["sorry", "don't know", "cannot find", "need more information"]):
        return True
    question_keywords = ["type", "types", "explain", "list", "what are", "different kinds", "kinds"]
    if any(word in user_query.lower() for word in question_keywords):
        if "type" not in answer_lower and "kind" not in answer_lower and "explain" not in answer_lower:
            return True
    return False

def extract_types_from_snippets(results, topic=None):
    types_texts = []
    pattern = re.compile(r"\b(type|types|kind|kinds|subtype|subtypes|category|categories) of ([\w\s,]+?)(?:[.;]|$)", re.IGNORECASE)
    for res in results:
        snippet = res.get("snippet", "")
        for match in pattern.finditer(snippet):
            types_str = match.group(2).strip()
            if not topic or topic.lower() in types_str.lower():
                types_texts.append(types_str)
    return "\n".join(types_texts)

def generate_answer_with_sources(messages, results, last_topic=None):
    extracted_types = extract_types_from_snippets(results, topic=last_topic)
    formatted_results_text = ""
    for idx, item in enumerate(results, start=1):
        formatted_results_text += f"[{idx}] {item['title']}\n{item['snippet']}\nSource: {item['link']}\n\n"

    system_prompt = (
        "You are a helpful and knowledgeable medical assistant chatbot. "
        "Provide concise, clear, and medically relevant answers based strictly on the following web search results. "
        "Avoid unnecessary details and focus on directly answering the user's question. "
        "When the user uses pronouns like 'it', 'those', 'these', or says 'explain that', "
        "infer that they mean the most recent medical topic or condition discussed earlier in the conversation. "
        "Always keep track of conversational context carefully. "
        "Answer the user's questions based on the following web search results. "
        "If you cannot find a clear answer, politely say you don't know and recommend consulting a healthcare professional. "
        "Cite your sources with numbers like [1], [2], etc.\n\n"
    )
    if extracted_types:
        system_prompt += f"Here are some types or categories extracted from the search results:\n{extracted_types}\n\n"
    system_prompt += f"{formatted_results_text}\n"

    openai_messages = [{"role": "system", "content": system_prompt}]
    openai_messages.extend(messages)

    if OPENAI_API_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=openai_messages,
                temperature=0.3,
            )
            return resp.choices[0].message["content"], "openai"
        except Exception as e:
            if "quota" not in str(e).lower():
                return f"OpenAI error: {e}", "openai"

    if GEMINI_API_KEY:
        try:
            conversation_text = system_prompt + "\nConversation:\n"
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_text += f"{role}: {msg['content']}\n"
            conversation_text += "Assistant:"
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(conversation_text)
            return resp.text, "gemini"
        except Exception as e:
            return f"Gemini error: {e}", "gemini"

    return "I don't know. Please consult a medical professional.", "none"

def get_last_medical_topic(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = msg.get("content", "")
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents if ent.label_ in {"DISEASE", "DISORDER", "SYMPTOM", "CONDITION"}]
            if entities:
                return entities[0].lower()
    return None

def contains_medical_entity(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"DISEASE", "DISORDER", "SYMPTOM", "CONDITION"}:
            return True
    return False

def rewrite_query(query, last_topic):
    if not last_topic:
        return query
    if contains_medical_entity(query):
        return query
    pronouns = ["it", "those", "these", "that", "them"]
    pattern = re.compile(r"\b(" + "|".join(pronouns) + r")\b", flags=re.IGNORECASE)
    return pattern.sub(last_topic, query)

# ------------------ API Endpoint ------------------
@app.route("/api/v1/search_answer", methods=["POST"])
def search_answer():
    data = request.get_json()
    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({
            "answer": "Please provide conversation history as a list of messages.",
            "sources": [],
            "restricted": False,
            "fallback": False,
            "model_used": "none"
        })

    latest_user_message = next((msg.get("content", "").strip() for msg in reversed(messages) if msg.get("role") == "user"), None)
    if not latest_user_message:
        return jsonify({
            "answer": "No user message found in conversation.",
            "sources": [],
            "restricted": False,
            "fallback": False,
            "model_used": "none"
        })

    if contains_abuse(latest_user_message):
        return jsonify({
            "answer": "I am here to help with medical questions. Please keep the conversation respectful. How can I assist you today?",
            "sources": [],
            "restricted": True,
            "fallback": False,
            "model_used": "none"
        })

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if any(latest_user_message.lower().startswith(greet) for greet in greetings):
        return jsonify({
            "answer": "Hi! How may I help you with your medical questions today?",
            "sources": [],
            "restricted": True,
            "fallback": False,
            "model_used": "none"
        })

    last_topic = get_last_medical_topic(messages)
    search_query = rewrite_query(latest_user_message, last_topic)

    results, _ = google_search_with_citations(search_query, num_results=5, broad=False, topic=last_topic)
    extracted_types = extract_types_from_snippets(results, topic=last_topic)
    answer, model_used = generate_answer_with_sources(messages, results, last_topic=last_topic)

    if "type" in latest_user_message.lower() and not extracted_types:
        fallback_query = f"types of {last_topic}" if last_topic else latest_user_message
        fallback_results, _ = google_search_with_citations(fallback_query, num_results=10, broad=False, topic=last_topic)
        fallback_results_broad, _ = google_search_with_citations(fallback_query, num_results=10, broad=True, topic=last_topic)
        combined_results = fallback_results + fallback_results_broad
        answer, model_used = generate_answer_with_sources(messages, combined_results, last_topic=last_topic)
        return jsonify({
            "answer": answer,
            "sources": combined_results,
            "restricted": False,
            "fallback": True,
            "model_used": model_used
        })

    if is_answer_incomplete(answer, latest_user_message):
        fallback_results, _ = google_search_with_citations(search_query, num_results=15, broad=True, topic=last_topic)
        answer, model_used = generate_answer_with_sources(messages, fallback_results, last_topic=last_topic)
        return jsonify({
            "answer": answer,
            "sources": fallback_results,
            "restricted": False,
            "fallback": True,
            "model_used": model_used
        })

    return jsonify({
        "answer": answer,
        "sources": results,
        "restricted": True,
        "fallback": False,
        "model_used": model_used
    })

# ------------------ Serve Frontend ------------------
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "medibot.html")

# ------------------ Run ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)








