# generate_sales.py
import pandas as pd
import random

products = ["Laptop", "Tablet", "Smartphone"]
quarters = ["Q1", "Q2", "Q3", "Q4"]

data = []
for _ in range(100):
    product = random.choice(products)
    quarter = random.choice(quarters)
    units_sold = random.randint(1, 50)
    revenue = round(units_sold * random.uniform(200, 1500), 2)
    data.append({"Product": product, "Quarter": quarter, "Units Sold": units_sold, "Revenue": revenue})

df = pd.DataFrame(data)
df.to_csv("sales.csv", index=False)
print("Generated sales.csv")
