
import pandas as pd

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

# 1. Periksa TotalCharges sebelum konversi
print("Contoh nilai TotalCharges:")
print(df["TotalCharges"].head(10).tolist())

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing setelah konversi TotalCharges:")
print(df["TotalCharges"].isna().sum())

# 2. Periksa baris duplikat
print("\nJumlah baris duplikat:", df.duplicated().sum())

# 3. Periksa duplikasi customerID
print(
    "Jumlah customerID duplikat:",
    df["customerID"].duplicated().sum()
)

# 4. Periksa nilai unik target
print("\nNilai unik Churn:")
print(df["Churn"].value_counts(dropna=False))

# 5. Periksa nilai unik kolom kategorikal
categorical_cols = df.select_dtypes(
    include=["object", "str"]
).columns

print("\nJumlah kategori setiap kolom:")
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} kategori")

# 6. Tampilkan baris dengan TotalCharges kosong
print("\nBaris TotalCharges kosong:")
print(
    df.loc[
        df["TotalCharges"].isna(),
        ["customerID", "tenure", "TotalCharges", "Churn"]
    ]
)
