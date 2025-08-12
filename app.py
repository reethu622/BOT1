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

# Basic abusive words list (expand as needed)
ABUSIVE_WORDS = ["idiot", "stupid", "dumb", "hate", "shut up", "fool", "damn", "bastard", "crap"]

def contains_abuse(text):
    text = text.lower()
    for word in ABUSIVE_WORDS:
        if word in text:
            return True
    return False

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

def extract_types_from_snippets(results, topic=None):
    """
    Look for patterns like 'types of', 'kinds of', 'subtypes' in snippets to extract types.
    Returns a string summary of types found or empty string.
    """
    types_texts = []
    pattern = re.compile(r"(types|kinds|subtypes|categories) of ([\w\s,]+)", re.IGNORECASE)
    for res in results:
        for match in pattern.finditer(res.get("snippet", "")):
            types_str = match.group(2).strip()
            # Optional: filter only those that mention the topic
            if topic:
                if topic.lower() in types_str.lower():
                    types_texts.append(types_str)
                else:
                    types_texts.append(types_str)
            else:
                types_texts.append(types_str)
    return "\n".join(types_texts)

def generate_answer_with_sources(messages, results, last_topic=None):
    """Generate an answer using OpenAI or Gemini based on search results and conversation."""

    extracted_types = extract_types_from_snippets(results, topic=last_topic)
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
    )
    if extracted_types:
        system_prompt += f"Here are some types or categories extracted from the search results:\n{extracted_types}\n\n"

    system_prompt += f"{formatted_results_text}\n"

    openai_messages = [{"role": "system", "content": system_prompt}]
    openai_messages.extend(messages)

    # Try OpenAI first
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

    # Fallback to Gemini
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

    # If no LLM keys, return fallback message
    return "I don't know. Please consult a medical professional."

def get_last_medical_topic(messages):
    """
    Extract medical entities from user messages using scispaCy NLP,
    return the most recent relevant entity as last topic.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        text = msg.get("content", "")
        doc = nlp(text)
        # Extract entities labeled as DISEASE, DISORDER, SYMPTOM, CONDITION in scispaCy
        entities = [ent.text for ent in doc.ents if ent.label_ in {"DISEASE", "DISORDER", "SYMPTOM", "CONDITION"}]
        if entities:
            return entities[0].lower()
    return None

def rewrite_query(query, last_topic):
    """
    Replace ambiguous pronouns with last_topic if found.
    """
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

    results, _ = google_search_with_citations(search_query, num_results=5, broad=False)
    extracted_types = extract_types_from_snippets(results, topic=last_topic)
    answer = generate_answer_with_sources(messages, results, last_topic=last_topic)

    # If the user asks about "types" but no types info found, do fallback search forcing "types of <topic>"
    if "type" in latest_user_message.lower() and not extracted_types:
        fallback_query = f"types of {last_topic}" if last_topic else latest_user_message
        fallback_results, _ = google_search_with_citations(fallback_query, num_results=10, broad=False)
        answer = generate_answer_with_sources(messages, fallback_results, last_topic=last_topic)
        return jsonify({"answer": answer, "sources": fallback_results})

    # If answer is incomplete, fallback to broader search
    if is_answer_incomplete(answer, latest_user_message):
        fallback_results, _ = google_search_with_citations(search_query, num_results=15, broad=True)
        answer = generate_answer_with_sources(messages, fallback_results, last_topic=last_topic)
        return jsonify({"answer": answer, "sources": fallback_results})

    return jsonify({"answer": answer, "sources": results})

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "medibot.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)





