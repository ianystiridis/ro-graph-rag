import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def build_prompt(question: str, vector_results: list, graph_results: list) -> str:
    vector_context = "\n\n".join(
        [
            f"[VECTOR {i+1}] Doc: {r.get('document_id', 'unknown')}\n{r.get('text', '')}"
            for i, r in enumerate(vector_results)
        ]
    )

    graph_context = "\n\n".join(
        [
            f"[GRAPH {i+1}] Doc: {r.get('document_id', 'unknown')}\n{r.get('text', '')}"
            for i, r in enumerate(graph_results)
        ]
    )

    return f"""
Ești un asistent care răspunde în limba română folosind doar dovezile oferite.

Întrebare:
{question}

Dovezi din metoda vectorială:
{vector_context}

Dovezi din metoda bazată pe graf:
{graph_context}

Instrucțiuni:
- Răspunde clar și concis în limba română.
- Folosește doar informațiile din dovezile de mai sus.
- Dacă dovezile nu sunt suficiente, spune că nu există suficiente informații în dovezile recuperate.
- Menționează documentele relevante, de exemplu: rowiki_admin_000.
- Nu inventa informații externe.

Răspuns:
""".strip()


def generate_answer_openai(
    question: str,
    vector_results: list,
    graph_results: list,
    model: str = "gpt-4.1-mini",
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")

    client = OpenAI(api_key=api_key)

    prompt = build_prompt(question, vector_results, graph_results)

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return response.output_text