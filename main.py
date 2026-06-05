from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datetime
import wikipedia
from zoneinfo import ZoneInfo

wikipedia.set_lang("en")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model... Please wait.")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

print("=== ChatGPT Style Search Engine ===")
print("Type 'exit' to stop\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    q = question.lower()

    # Date
    if "date" in q:
        today = datetime.datetime.now().strftime("%d %B %Y")
        print("\nAI:")
        print("Today's date is", today)
        print("\n" + "-" * 50 + "\n")
        continue

    # Time in India
    if "time" in q:
        india_time = datetime.datetime.now(
            ZoneInfo("Asia/Kolkata")
        ).strftime("%I:%M %p")

        print("\nAI:")
        print("Current time in India is", india_time)
        print("\n" + "-" * 50 + "\n")
        continue

    # Fixed location answers
    if "where is bengaluru" in q:
        print("\nAI:")
        print("Bengaluru is the capital city of Karnataka in southern India.")
        print("\n" + "-" * 50 + "\n")
        continue

    if "where is coimbatore" in q:
        print("\nAI:")
        print("Coimbatore is a city in Tamil Nadu in southern India.")
        print("\n" + "-" * 50 + "\n")
        continue

    # Wikipedia Search
    try:
        wiki_answer = wikipedia.summary(
            question,
            sentences=2,
            auto_suggest=False
        )

        print("\nAI:")
        print(wiki_answer)
        print("\n" + "-" * 50 + "\n")

    except:
        prompt = f"""
<|system|>
You are a helpful AI assistant.
<|user|>
{question}
<|assistant|>
"""

        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        print("\nAI:")
        print(response)
        print("\n" + "-" * 50 + "\n")
