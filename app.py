from flask import Flask, request, jsonify, render_template
from langdetect import detect
from dotenv import load_dotenv
import os

from store import get_retriever
from src.prompt import get_prompt, get_owner_context
from src.helper import get_llm

load_dotenv()

# ======================
# FLASK APP
# ======================
app = Flask(__name__)

# ======================
# LLM SETUP
# ======================
llm = get_llm()
translator_llm = llm

# ======================
# RETRIEVER + PROMPT
# ======================
retriever = get_retriever()
prompt = get_prompt()


# ======================
# TRANSLATION
# ======================
def translate_to_en(text):
    return translator_llm.invoke(
        f"Translate this to English:\n{text}"
    ).content


def translate_to_vi(text):
    return translator_llm.invoke(
        f"Translate this to Vietnamese:\n{text}"
    ).content


# ======================
# RAG PIPELINE
# ======================
def run_rag(user_message: str):
    try:
        lang = detect(user_message)
    except:
        lang = "en"

    # Translate input nếu không phải EN
    query = translate_to_en(user_message) if lang != "en" else user_message

    # Retrieve context
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs]) if docs else ""

    # Build prompt
    final_prompt = prompt.format(
        context=context,
        chat_history="",
        question=query,
        owner_context=get_owner_context()
    )

    # Generate answer
    answer = llm.invoke(final_prompt).content

    # Translate back nếu user là VI
    if lang == "vi":
        answer = translate_to_vi(answer)

    return answer


# ======================
# ROUTES
# ======================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = data.get("message", "").strip()

    if not msg:
        return jsonify({"error": "Empty message"}), 400

    response = run_rag(msg)

    return jsonify({"response": response})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ======================
# GUNICORN ENTRY (IMPORTANT)
# ======================
# Render sẽ dùng gunicorn app:app nên KHÔNG cần app.run()
# Nhưng giữ lại để chạy local

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)