import os
from google import genai
from app.config import GOOGLE_API_KEY, GEMINI_MODEL

client = None

def get_client():
    global client
    if client is not None:
        return client
    api_key = GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            print("\nInitializing Gemini Client...")
            client = genai.Client(api_key=api_key)
            print("Gemini Ready")
            return client
        except Exception as e:
            print(f"Warning: Failed to initialize Gemini Client: {e}")
            return None
    return None

get_client()

SYSTEM_PROMPT = """
You are a financial document analyst.
Answer ONLY using the provided context.
Rules:
1. Use only information present in the context.
2. Do not use outside knowledge.
3. If information is missing say: "I could not find that information in the filing."
4. Cite the most relevant sources.
5. Keep answers concise but complete.
6. If multiple sources support the answer, use the strongest source.
7. Never invent financial figures.
8. Use source references exactly as provided.
9. Prefer table values for financial questions.
10. If answering a financial question, include the exact value reported.
"""

def build_context(retrieval_result):
    context_parts = []
    results = retrieval_result["results"]
    for idx, result in enumerate(results, start=1):
        source_header = f"""
SOURCE {idx}
Type: {result['type']}
Section: {result['section']}
"""
        body = result["context"] if result["type"] == "text" else result["content"]
        context_parts.append(source_header + "\n" + body)
    return "\n\n" + ("\n" + "=" * 80 + "\n").join(context_parts)

def generate_answer(query, retrieval_result):
    print("\n" + "=" * 80)
    print("GENERATING ANSWER")
    print("=" * 80)
    context = build_context(retrieval_result)
    prompt = f"""
{SYSTEM_PROMPT}
QUESTION:
{query}
CONTEXT:
{context}
ANSWER:
"""
    ai_client = get_client()
    if not ai_client:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is not configured in backend/python/.env. Please provide a valid Gemini API key to generate answers.")
    models_to_try = list(dict.fromkeys([GEMINI_MODEL, "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]))
    response = None
    last_err = None
    for model_name in models_to_try:
        try:
            print(f"Calling Gemini model: {model_name}...")
            response = ai_client.models.generate_content(model=model_name, contents=prompt)
            if response and response.text:
                break
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            last_err = e
            continue

    if response is None:
        raise last_err

    answer = response.text
    print("\nANSWER GENERATED")
    return {
        "query": query,
        "answer": answer,
        "sources": [
            {"section": r["section"], "page": r.get("page"), "type": r["type"]}
            for r in retrieval_result["results"]
        ]
    }