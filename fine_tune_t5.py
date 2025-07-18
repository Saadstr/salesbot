from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Define base model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Your training data
data = {
    "question": [
        "What is the total revenue?",
        "How much revenue was generated in Q1?",
        "Tell me the revenue for laptops in Q3.",
        "How many tablets were sold in Q2?",
        "What does revenue mean?",
        "Explain sales in simple terms.",
        "Units sold for smartphones in Q4?",
        "What is the total number of units sold?",
        "How much did laptops sell for in Q2?",
        "What is revenue?",
        "Define sales.",
        "How many products were sold in Q1?",
        "Revenue for Q4?",
        "Tell me about tablet sales.",
        "How many smartphones sold in Q1?",
        "Total sales in all quarters?",
        "What are units sold?",
        "Explain the term revenue.",
        "Sales data for laptops.",
        "Total revenue for Q2?",
        "Do we sell laptops?",
        "What is revenue for smartphones in Q3?",
        "How many laptops were sold in Q1?",
        "Revenue comparison between Q1 and Q2?",
        "Which quarter had the highest revenue?",
        "What was the total units sold in Q3?",
        "How many units of tablets sold in Q1?",
        "Smartphone revenue for Q4?",
        "Did tablet sales increase in Q4?",
        "Are laptops the best selling product?",
        "What is the revenue for tablets in Q2?",
        "How many smartphones sold in Q2?",
        "Do we sell tablets?",
        "How much revenue was generated from laptops in Q4?",
        "How many units sold in total for smartphones?",
        "Which product had the highest revenue in Q1?",
        "What is the revenue for Q3?",
        "How many laptops sold in Q4?",
        "Total units sold for tablets across all quarters?",
        "What is the revenue from smartphones in Q1?",
        "Did revenue increase from Q2 to Q3?",
        "How many units of laptops sold in Q2?",
        "Explain units sold.",
        "What are the total tablets sold?",
        "What was the revenue for smartphones in Q2?",
        "Do we sell smartphones?",
        "Total revenue generated in Q3?",
        "How many units were sold in Q4?",
        "Revenue generated from tablets in Q4?",
        "How many laptops sold in total?",
        "What is the total revenue for Q2?",
        "Number of units sold in Q1?",
        "Do tablets generate more revenue than laptops?",
        "Explain sales data.",
        "Which product had the most units sold?",
        "Revenue for laptops in Q1?",
        "What is the total number of units sold in Q2?",
        "How much did tablets sell for in Q3?",
        "Units sold for laptops in Q4?",
        "Revenue for smartphones in Q2?",
        "Did smartphone sales increase in Q3?",
        "What does units sold mean?",
        "Total revenue by product?",
        "How many units were sold across all products?",
        "What was the revenue for Q1?",
        "What are sales?",
        "Explain revenue.",
        "How many tablets sold in Q4?",
        "What was the total units sold in Q2?",
        "Which quarter had the lowest revenue?",
        "How many laptops sold in Q3?",
        "Revenue for tablets in Q1?",
        "Are smartphones profitable?",
        "How many smartphones sold in Q3?",
        "What is the total revenue for laptops?",
        "Do we sell smartphones in Q1?",
        "What is the revenue for tablets in Q4?",
        "How many units sold in Q3?",
        "Did revenue increase in Q4?",
        "Explain revenue in simple terms.",
        "How many total units sold for laptops?",
        "What was the revenue for Q2?",
        "Do we sell tablets in Q1?",
        "What was the total revenue in Q4?",
        "How many units of smartphones sold in Q1?",
        "Is Q4 the best quarter for sales?",
        "How much revenue did laptops generate in Q1?",
        "What is the total number of units sold in Q4?",
        "Do we have sales data for Q3?",
        "Explain the term sales.",
        "What is the revenue for Q2?",
        "How many tablets sold in Q3?"
    ],
    "answer": [
        "The total revenue is $2,114,771.95.",
        "Revenue in Q1 was $618,047.13.",
        "Laptops generated $62,888.78 in revenue in Q3.",
        "There were 280 tablets sold in Q2.",
        "Revenue is the total money earned from sales.",
        "Sales refer to the total units of products sold.",
        "Smartphones sold 249 units in Q4.",
        "Total units sold are 2,535 across all products.",
        "Laptops sold for $166,434.85 in Q2.",
        "Revenue is income generated from selling products.",
        "Sales mean the total number of products sold.",
        "In Q1, 746 products were sold.",
        "Revenue for Q4 is $638,873.74.",
        "Tablet sales totaled $684,880.42 across all quarters.",
        "246 smartphones were sold in Q1.",
        "Total sales across all quarters are $2,114,771.95.",
        "Units sold means the quantity of products sold.",
        "Revenue represents the earnings from sales.",
        "Laptops had 975 units sold and $717,184.61 revenue.",
        "Q2 revenue totaled $611,507.93.",
        "Yes we do, we sell laptops.",
        "Smartphone revenue for Q3 was $180,573.05.",
        "215 laptops were sold in Q1.",
        "Q1 revenue was $618,047.13 and Q2 revenue was $611,507.93.",
        "Q4 had the highest revenue with $638,873.74.",
        "A total of 320 units were sold in Q3.",
        "Tablets sold 180 units in Q1.",
        "Smartphone revenue in Q4 was $180,573.05.",
        "Yes, tablet sales increased in Q4 compared to Q3.",
        "Yes, laptops are the best selling product.",
        "Tablet revenue in Q2 was $175,000.00.",
        "230 smartphones were sold in Q2.",
        "Yes, we sell tablets.",
        "Laptops generated $190,000.00 in revenue in Q4.",
        "A total of 665 smartphones were sold across all quarters.",
        "Laptops had the highest revenue in Q1.",
        "Revenue for Q3 was $246,343.15.",
        "210 laptops were sold in Q4.",
        "A total of 895 tablets were sold across all quarters.",
        "Smartphone revenue in Q1 was $170,000.00.",
        "Revenue decreased from Q2 to Q3.",
        "200 laptops sold in Q2.",
        "Units sold refer to the number of products purchased or sold.",
        "A total of 895 tablets were sold.",
        "Smartphone revenue in Q2 was $175,000.00.",
        "Yes, smartphones are part of our product line.",
        "Revenue generated in Q3 was $246,343.15.",
        "320 units were sold in Q4 across all products.",
        "Tablet revenue in Q4 was $180,000.00.",
        "A total of 975 laptops were sold.",
        "Total revenue for Q2 was $611,507.93.",
        "746 units sold in Q1.",
        "No, tablets generate less revenue than laptops.",
        "Sales data represents the record of product sales including units and revenue.",
        "Laptops had the most units sold.",
        "Laptops generated $150,000.00 in revenue in Q1.",
        "1,050 units were sold in Q2 across all products.",
        "Tablets sold for $230,000.00 in Q3.",
        "215 laptops were sold in Q4.",
        "Smartphone revenue in Q2 was $175,000.00.",
        "Smartphone sales increased in Q3.",
        "Units sold means the quantity of products sold.",
        "Revenue by product is: Laptops $717,184.61, Tablets $684,880.42, Smartphones $712,706.92.",
        "2,535 units sold across all products.",
        "Revenue in Q1 was $618,047.13.",
        "Sales are the total units or amount of products sold.",
        "Revenue is the income earned from selling products.",
        "280 tablets sold in Q4.",
        "A total of 1,050 units sold in Q2.",
        "Q3 had the lowest revenue with $246,343.15.",
        "210 laptops sold in Q3.",
        "Tablet revenue in Q1 was $180,000.00.",
        "Yes, smartphones are profitable.",
        "230 smartphones sold in Q3.",
        "Total laptop revenue is $717,184.61.",
        "Yes, smartphones were sold in Q1.",
        "Tablet revenue in Q4 was $180,000.00.",
        "320 units sold in Q3 across all products.",
        "Revenue increased in Q4.",
        "Revenue is the total income generated from sales.",
        "975 laptops sold in total.",
        "Revenue in Q2 was $611,507.93.",
        "Yes, tablets were sold in Q1.",
        "Total revenue in Q4 was $638,873.74.",
        "246 smartphones sold in Q1.",
        "Q4 is the best quarter for sales.",
        "Laptops generated $150,000.00 revenue in Q1.",
        "715 units sold in Q4.",
        "Yes, sales data for Q3 is available.",
        "Sales is the total number or value of products sold.",
        "Revenue in Q2 was $611,507.93.",
        "280 tablets sold in Q3."
    ]
}


# Convert to Hugging Face Dataset
dataset = Dataset.from_dict(data)

# Tokenization function
def preprocess_function(examples):
    inputs = ["question: " + q for q in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["answer"], max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./sales_qa_model",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Train and save
trainer.train()
trainer.save_model("./sales_qa_model")
tokenizer.save_pretrained("./sales_qa_model")

print("Model and tokenizer saved to ./sales_qa_model")
