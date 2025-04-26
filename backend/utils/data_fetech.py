from yahooquery import Ticker
import pandas as pd

# Get NVIDIA data
nvda = Ticker("NVDA")

# Fetch quarterly key statistics
stats = nvda.valuation_measures

# Convert to DataFrame
df = pd.DataFrame(stats).T

# Reset index to clean structure
df = df.reset_index()

# Drop 'periodType' row if it exists
if "periodType" in df["index"].values:
    df = df[df["index"] != "periodType"]
    print("Dropped 'periodType' row.")

# Rename index column for clarity
df = df.rename(columns={"index": "Metric"})  

# Pivot: Convert Metrics into Columns (Dates in Rows)
df_pivoted = df.set_index("Metric").T  # Transpose the table

# Reset index to get the correct format for Snowflake
df_pivoted.reset_index(inplace=True)

# Rename the first column to 'Date'
df_pivoted = df_pivoted.rename(columns={"index": "Date"})

# Drop the 'symbol' column if it exists
if "symbol" in df_pivoted.columns:
    df_pivoted = df_pivoted.drop(columns=["symbol"])
    print("Removed 'symbol' column.")

# Save as CSV (for Snowflake ingestion)
csv_filename = "nvidia_pivoted_cleaned_data.csv"
df_pivoted.to_csv(csv_filename, index=False)  # No extra index column

print(f"✅ Data saved successfully to: {csv_filename} (without 'symbol' column).")


import pandas as pd

# Load CSV
csv_filename = "nvidia_pivoted_cleaned_data.csv"
df = pd.read_csv(csv_filename)

# Print column names
print("Column names in CSV:", df.columns.tolist())
