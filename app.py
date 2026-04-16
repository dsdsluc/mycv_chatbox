from flask import Flask, request, jsonify, render_template
from langdetect import detect
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# ======================
# GLOBAL (KHÔNG CRASH IMPORT)
# ======================
llm = None
translator_llm = None
retriever = None
prompt = None


# ======================
# INIT SERVICES SAFELY
# ======================
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


init_services()


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

    query = translate_to_en(user_message) if lang != "en" else user_message

    docs = retriever.invoke(query) if retriever else []
    context = "\n".join([d.page_content for d in docs]) if docs else ""

    final_prompt = prompt.format(
        context=context,
        chat_history="",
        question=query,
        owner_context="AI Assistant"
    )

    answer = llm.invoke(final_prompt).content if llm else "Service not ready"

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
    return jsonify({
        "status": "ok",
        "llm": llm is not None,
        "retriever": retriever is not None
    })


# ======================
# LOCAL RUN ONLY
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)