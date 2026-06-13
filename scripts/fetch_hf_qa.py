import json
from transformers import pipeline, set_seed

def fetch_hf_qa(model_name: str = "google/flan-t5-base", prompt: str = None):
    """Fetch a QA pair from a Hugging Face text‑generation model.
    Returns a JSON string with keys: question, context, answer.
    """
    generator = pipeline("text2text-generation", model=model_name, device=-1)
    if not prompt:
        prompt = (
            "Question: Is ibuprofen safe for children under 12 years old?\n"
            "Context: Clinical trial showed ibuprofen 5‑10 mg/kg reduces fever in children.\n"
            "Answer:"
        )
    set_seed(42)
    result = generator(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]
    answer = result.strip()
    payload = {
        "question": "Is ibuprofen safe for children under 12 years old?",
        "context": "Clinical trial showed ibuprofen 5‑10 mg/kg reduces fever in children.",
        "answer": answer,
    }
    return json.dumps(payload, indent=2)

if __name__ == "__main__":
    print(fetch_hf_qa())
