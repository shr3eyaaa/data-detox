
import shutil
import numpy as np
import pandas as pd

RAW = "customer_data.csv"
BACKUP = "customer_data_raw_backup.csv"
CLEAN = "cleaned_customer_data.csv"
# 0. Backup the raw file BEFORE touching anything
shutil.copy(RAW, BACKUP)
# 1. Load + explore
df = pd.read_csv(RAW)
print("Shape:", df.shape)
print(df.dtypes, "\n")
print("Missing values:\n", df.isna().sum(), "\n")
print("Duplicate rows:", df.duplicated().sum())
print("Age range:", df["Age"].min(), "to", df["Age"].max())
print("TotalPurchase min:", df["TotalPurchase"].min(), "\n")
for col in ["Gender", "Country", "PreferredDevice"]:
    print(df[col].value_counts(dropna=False), "\n")

rows_before = len(df)
# 2. Remove duplicate rows (done first so they don't skew medians later)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {rows_before - len(df)} exact duplicate rows")
no_id = df["CustomerID"].isna().sum()
df = df.dropna(subset=["CustomerID"]).reset_index(drop=True)
print(f"Dropped {no_id} rows with no CustomerID")
# 3. Fix inconsistent text / categorical values
for col in ["Gender", "Country", "PreferredDevice"]:
    df[col] = df[col].str.strip().str.lower()

df["Gender"] = df["Gender"].replace({"femlae": "female", "mle": "male"})
df["Country"] = df["Country"].replace({"indai": "india", "canda": "canada"})
df["PreferredDevice"] = df["PreferredDevice"].replace(
    {"dasktop": "desktop", "moblie": "mobile"}
)
df["Country"] = df["Country"].apply(
    lambda x: x.upper() if x in ("usa", "uk") else (x.title() if pd.notna(x) else x)
)
df["Gender"] = df["Gender"].str.title()
df["PreferredDevice"] = df["PreferredDevice"].str.title()

df["Name"] = df["Name"].str.strip()
df.loc[df["Name"].str.fullmatch(r"\d+", na=False), "Name"] = np.nan

valid_email = df["Email"].str.fullmatch(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", na=False)
df.loc[~valid_email, "Email"] = np.nan
df["Email"] = df["Email"].str.lower()
# 4. Invalid numeric values -> NaN (then imputed below)
df.loc[(df["Age"] < 0) | (df["Age"] > 100), "Age"] = np.nan   
df.loc[df["TotalPurchase"] < 0, "TotalPurchase"] = np.nan      
# 5. Correct data types
df["SignupDate"] = pd.to_datetime(df["SignupDate"], errors="coerce")
df["LastLogin"] = pd.to_datetime(df["LastLogin"], errors="coerce")
# 6. Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median()).round().astype(int)
df["TotalPurchase"] = df["TotalPurchase"].fillna(df["TotalPurchase"].median()).round(2)

# Categorical -> "Unknown" (the data already uses this label for Gender)
for col in ["Gender", "Country", "PreferredDevice"]:
    df[col] = df[col].fillna("Unknown")
df["LoginBeforeSignup"] = df["LastLogin"] < df["SignupDate"]
print("Rows where LastLogin < SignupDate:", df["LoginBeforeSignup"].sum())
# 7. Verify + save
print("\nFinal shape:", df.shape)
print(df.dtypes, "\n")
print("Remaining missing values:\n", df.isna().sum())
print("Remaining duplicates:", df.duplicated().sum())

df.to_csv(CLEAN, index=False)
print(f"\nSaved cleaned data to {CLEAN}")
