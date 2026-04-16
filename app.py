from flask import Flask, request, jsonify, render_template
from langdetect import detect
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

llm = None
translator_llm = None
retriever = None
prompt = None
_initialized = False


def init_services():
    global llm, translator_llm, retriever, prompt

    try:
        from src.helper import get_llm
        from store import get_retriever
        from src.prompt import get_prompt

        llm = get_llm()
        translator_llm = llm
        retriever = get_retriever()
        prompt = get_prompt()

        print("✅ INIT SUCCESS")

    except Exception as e:
        print("❌ INIT ERROR:", str(e))


# ✅ Lazy init - doesn't block Gunicorn startup
@app.before_request
def lazy_init():
    global _initialized
    if not _initialized:
        init_services()
        _initialized = True


def translate_to_en(text):
    if not translator_llm:
        return text
    return translator_llm.invoke(
        f"Translate this to English:\n{text}"
    ).content


def translate_to_vi(text):
    if not translator_llm:
        return text
    return translator_llm.invoke(
        f"Translate this to Vietnamese:\n{text}"
    ).content


def run_rag(user_message: str):
    if not llm or not prompt:
        return "Service is still initializing, please try again in a moment."

    try:
        lang = detect(user_message)
    except:
        lang = "en"

    query = translate_to_en(user_message) if lang != "en" else user_message

    docs = retriever.invoke(query) if retriever else []
    context = "\n".join([d.page_content for d in docs]) if docs else ""

    final_prompt = prompt.format(
        context=context,
        chat_history="",
        question=query,
        owner_context="AI Assistant"
    )

    answer = llm.invoke(final_prompt).content

    if lang == "vi":
        answer = translate_to_vi(answer)

    return answer


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
    return jsonify({
        "status": "ok",
        "llm": llm is not None,
        "retriever": retriever is not None
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)