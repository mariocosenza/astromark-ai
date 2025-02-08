import os

import google.generativeai as genai

model_config = genai.types.GenerationConfig(
    temperature=2,
    top_p=0.6,
    top_k=1,
    max_output_tokens=150
)


def get_llm(mark_text):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash", generation_config=model_config)
    prompt = f"Potresti darmi alcuni consisgli sul mio orientamento universitario considera che devi rivolgerti in modo formale e deve essere come un paragrafo non rivolto a me direttamente. Cerca di essere sintentico ma obbiettivo e formale. Questi sono i miei voti: {mark_text}"
    response = model.generate_content(prompt)
    return response.text
