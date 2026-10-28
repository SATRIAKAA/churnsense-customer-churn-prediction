
import pandas as pd

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("Ukuran dataset:", df.shape)

print("\n5 baris pertama:")
print(df.head())

print("\nInformasi kolom:")
df.info()

print("\nDistribusi target Churn:")
print(df["Churn"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
