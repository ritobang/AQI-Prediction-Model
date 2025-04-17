import pandas as pd

file_path = "city_day.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df_cleaned = df.dropna(subset=['AQI'])
numeric_cols = df_cleaned.select_dtypes(include=['float64']).columns
df_cleaned.loc[:, numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
df_cleaned = df_cleaned.drop_duplicates()
cleaned_file_path = "city_day_cleaned.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved as: {cleaned_file_path}")