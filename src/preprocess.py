import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/raw/fee_data.csv")

df = df.sort_values(["student_id", "month"])

df["payment_ratio"] = df["fee_paid"] / df["fee_due"]

df["avg_delay_3m"] = (
    df.groupby("student_id")["days_delayed"]
    .transform(lambda x: x.rolling(3, min_periods=1).mean())
)

df["delay_streak"] = (
    df.groupby("student_id")["defaulted"]
    .transform(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
)

df["total_arrears"] = (
    df.groupby("student_id")
    .apply(lambda g: (g["fee_due"] - g["fee_paid"]).cumsum())
    .reset_index(level=0, drop=True)
)

df["class_section_default_rate"] = (
    df.groupby(["class", "section"])["defaulted"].transform("mean")
)

le = LabelEncoder()
df["income_enc"] = le.fit_transform(df["parent_income_band"])

df.to_csv("data/processed/features.csv", index=False)

print("Preprocessing done")
print("Shape:", df.shape)
print("Saved to data/processed/features.csv")