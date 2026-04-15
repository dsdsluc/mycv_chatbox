from flask import Flask, request, jsonify, render_template
from langdetect import detect
from dotenv import load_dotenv
import os

from store import get_retriever
from src.prompt import get_prompt, get_owner_context
from src.helper import get_llm

load_dotenv()

app = Flask(__name__)

# ======================
# LLM SETUP
# ======================
llm = get_llm()

translator_llm = get_llm()

# ======================
# RETRIEVER
# ======================
retriever = get_retriever()

prompt = get_prompt()


# ======================
# TRANSLATION FUNCTIONS
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
def run_rag(user_message):
    lang = detect(user_message)

    # 1. Translate input nếu cần
    query = translate_to_en(user_message) if lang != "en" else user_message

    # 2. Retrieve context
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    # 3. Build prompt
    final_prompt = prompt.format(
        context=context,
        chat_history="",
        question=query,
        owner_context=get_owner_context()
    )

    # 4. Generate answer
    answer = llm.invoke(final_prompt).content

    # 5. Translate back nếu user là Vietnamese
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
    msg = request.json.get("message", "").strip()

    if not msg:
        return jsonify({"error": "Empty message"}), 400

    response = run_rag(msg)

    return jsonify({
        "response": response
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ======================
# RUN SERVER
# ======================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)