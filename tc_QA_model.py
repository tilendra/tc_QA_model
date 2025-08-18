
"""
Asks multiple questions, optionally with a given context, using OpenAI Chat API.

Args:
    api_key (str): Your OpenAI API key.
    questions (List[str]): A list of questions to ask.
    context (str, optional): The input text or passage. Defaults to None.
    model (str): The OpenAI model to use.
    temperature (float): Sampling temperature (lower = more factual).

    Use low temperature (0–0.3) for: precise, factual, consistent output
    Use moderate (0.4–0.7) for: balance between accuracy and variety
    Use high (0.8+) for: creativity, ideation, or fun


Returns:
    List[str]: Answers returned from the model.
"""
from openai import OpenAI
from typing import List, Optional

def ask_questions_with_context(
    api_key: str,
    questions: List[str],
    context: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3
) -> List[str]:
    """
    Asks multiple questions, optionally with a given context (e.g., text or code).
    If the context is code, the model outputs:
      1. Explanation
      2. Modified Code
    Also prints Q&A pairs automatically.
    """
    client = OpenAI(api_key=api_key)
    answers = []

    for question in questions:
        if context:
            # Special instructions if context looks like code
            if "def " in context or "class " in context or any(sym in context for sym in [";", "{", "}"]):
                user_content = (
                    f"The following is code:\n\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Please respond with two sections:\n"
                    "1. Explanation: Explain what the code does in simple terms.\n"
                    "2. Modified Code: Provide updated code based on the question or improvements.\n"
                )
            else:
                user_content = (
                    f"Read the following passage and answer the question.\n\n"
                    f"Passage:\n{context}\n\n"
                    f"Question:\n{question}\n\nAnswer:"
                )
        else:
            user_content = f"Answer the following question:\n\nQuestion:\n{question}\n\nAnswer:"

        messages = [
            {"role": "system", "content": "You are a helpful assistant and skilled code interpreter."},
            {"role": "user", "content": user_content}
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        answer = response.choices[0].message.content.strip()
        answers.append(answer)

    # Auto-print Q&A
    for q, a in zip(questions, answers):
        print(f"Q: {q}\n{a}\n")

    return answers
