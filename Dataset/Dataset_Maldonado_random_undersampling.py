import pandas as pd
from pathlib import Path

input_path = Path("maldonado.csv")
output_path = Path("maldonado_undersampled.csv")

df = pd.read_csv(input_path)

# label corretta per Maldonado
label_col = "satd"

# pulizia minima (consigliata)
df.columns = df.columns.str.strip()
df[label_col] = df[label_col].astype(int)

df_satd = df[df[label_col] == 1]
df_non_satd = df[df[label_col] == 0]

print("Original Distribution:")
print(df[label_col].value_counts(), "\n")

# === Undersampling ===
# (nota: funziona solo se i non-SATD sono >= SATD; in genere sì)
df_non_satd_down = df_non_satd.sample(
    n=len(df_satd),
    random_state=42
)

df_balanced = (
    pd.concat([df_satd, df_non_satd_down])
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

print("Distribution after undersampling:")
print(df_balanced[label_col].value_counts(), "\n")

df_balanced.to_csv(output_path, index=False)
print(f"Balanced dataset saved in: {output_path}")
