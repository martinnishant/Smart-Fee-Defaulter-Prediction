import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker("en_IN")

np.random.seed(42)
random.seed(42)

NUM_STUDENTS = 500
MONTHS = pd.date_range("2022-01", "2024-12", freq="MS").strftime("%Y-%m").tolist()

students = []

for _ in range(NUM_STUDENTS):
    sid = fake.uuid4()[:8]
    cls = random.randint(1, 12)
    section = random.choice(["A", "B", "C", "D", "E"])
    income = random.choices(["Low", "Mid", "High"], weights=[0.3, 0.5, 0.2])[0]
    scholarship = 1 if income == "Low" and random.random() < 0.4 else 0
    siblings = random.choices([0, 1, 2, 3], weights=[0.4, 0.35, 0.15, 0.1])[0]
    transport = random.randint(0, 1)

    base_risk = (
        (0.4 if income == "Low" else 0.1 if income == "High" else 0.25)
        + siblings * 0.05
        - scholarship * 0.15
    )

    for month in MONTHS:
        fee_due = random.randint(3000, 15000)
        delay = max(0, int(np.random.exponential(base_risk * 60)))
        fee_paid = fee_due if delay < 30 else int(fee_due * random.uniform(0.5, 0.95))

        students.append({
            "student_id": sid,
            "student_name": fake.name(),
            "class": cls,
            "section": section,
            "month": month,
            "fee_due": fee_due,
            "fee_paid": fee_paid,
            "days_delayed": delay,
            "defaulted": int(delay > 30),
            "sibling_count": siblings,
            "transport_opted": transport,
            "scholarship": scholarship,
            "parent_income_band": income
        })

df = pd.DataFrame(students)
df.to_csv("data/raw/fee_data.csv", index=False)

print(f"Generated {len(df)} rows")
print(f"Default rate: {df.defaulted.mean():.2%}")
print("Saved to data/raw/fee_data.csv")