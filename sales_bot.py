import pandas as pd
import spacy
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load sales data
try:
    df = pd.read_csv("sales.csv")
except FileNotFoundError:
    print("Error: sales.csv file not found. Please ensure it exists in the same directory.")
    exit()

# Load fine-tuned T5 model and tokenizer
try:
    model = T5ForConditionalGeneration.from_pretrained("./sales_qa_model")
    tokenizer = T5Tokenizer.from_pretrained("./sales_qa_model")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you've trained the model with fine_tune_t5.py.")
    exit()

def parse_question(text):
    doc = nlp(text.lower())
    product = None
    quarter = None
    for token in doc:
        if token.text in ["q1", "q2", "q3", "q4"]:
            quarter = token.text.upper()
        elif token.text in ["laptop", "laptops", "tablet", "tablets", "smartphone", "smartphones"]:
            product = token.lemma_.capitalize()
    return product, quarter

def answer_math_question(text):
    product, quarter = parse_question(text)
    filtered = df.copy()

    if quarter:
        filtered = filtered[filtered["Quarter"] == quarter]
    if product:
        filtered = filtered[filtered["Product"] == product]

    if "revenue" in text.lower():
        total = filtered["Revenue"].sum()
        return f"Revenue for {product or 'all products'} in {quarter or 'all quarters'} is ${total:,.2f}."

    elif "sold" in text.lower() or "units" in text.lower():
        total = filtered["Units Sold"].sum()
        return f"{total} units of {product or 'all products'} were sold in {quarter or 'all quarters'}."

    return None

def answer_with_t5(question):
    input_text = f"question: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)

    outputs = model.generate(
        input_ids,
        max_new_tokens=64,
        num_beams=4,
        early_stopping=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if not decoded:
        return "I'm not sure how to answer that yet."
    return decoded

# Main loop
print("Sales Math Bot ready! Ask your question or type 'exit'.")
while True:
    user_input = input("> ").strip()
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Try math-based answer first
    answer = answer_math_question(user_input)

    if answer:
        print("Answer:", answer)
    else:
        print("Falling back to language model...")
        try:
            result = answer_with_t5(user_input)
            print("Answer (T5):", result)
        except Exception as e:
            print("Error using T5 model:", e)
            print("Sorry, I couldn’t understand the question.")
