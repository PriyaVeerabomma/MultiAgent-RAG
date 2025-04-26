import os
from pathlib import Path
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
import glob

# ---------- LOAD .env FILE SAFELY ---------- #
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    raise FileNotFoundError(f".env file not found at {env_path}")

# ---------- LOAD CREDENTIALS ---------- #
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_REGION = os.getenv("SNOWFLAKE_REGION")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_STAGE = os.getenv("SNOWFLAKE_STAGE")

# ---------- FIND THE LATEST NVDA CSV FILE ---------- #
def find_latest_nvda_csv():
    # Find all NVDA CSV files in the current directory
    nvda_files = glob.glob("NVDA_5yr_history_*.csv")
    
    if not nvda_files:
        raise FileNotFoundError("No NVDA CSV files found in the current directory")
    
    # Sort by date (assuming filename format NVDA_5yr_history_YYYYMMDD.csv)
    latest_file = sorted(nvda_files, reverse=True)[0]
    print(f"📊 Found latest NVDA data file: {latest_file}")
    return latest_file

# ---------- DEBUG PRINT ---------- #
def print_env_check():
    print("✅ ENV CHECK:")
    print("USER:", SNOWFLAKE_USER)
    print("ACCOUNT (full):", SNOWFLAKE_ACCOUNT)
    print("ROLE:", SNOWFLAKE_ROLE)
    print("WAREHOUSE:", SNOWFLAKE_WAREHOUSE)
    print("DATABASE:", SNOWFLAKE_DATABASE)
    print("SCHEMA:", SNOWFLAKE_SCHEMA)
    print("STAGE:", SNOWFLAKE_STAGE)

# ---------- UPLOAD TO SNOWFLAKE STAGE ---------- #
def upload_csv_to_stage(file_path: str):
    print("🔗 Connecting to Snowflake...")
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        role=SNOWFLAKE_ROLE,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    cursor = conn.cursor()

    try:
        print("📤 Uploading CSV to Snowflake stage...")
        cursor.execute(f"PUT file://{file_path} @{SNOWFLAKE_STAGE} AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
        print("✅ Upload complete!")
    except Exception as e:
        print(f"❌ Failed to upload CSV: {e}")
    finally:
        cursor.close()
        conn.close()

# ---------- EXECUTE SETUP QUERIES IN SNOWFLAKE ---------- #
def run_sql_setup(csv_filename):
    base_filename = os.path.basename(csv_filename)
    
    sql_commands = [
        f"LIST @{SNOWFLAKE_STAGE}",
        """
        CREATE OR REPLACE TABLE NVDA_STOCK_DATA (
            DATE DATE,
            CLOSE FLOAT,
            HIGH FLOAT,
            LOW FLOAT,
            OPEN FLOAT,
            VOLUME FLOAT,
            MA_50 FLOAT,
            MA_200 FLOAT,
            DAILY_RETURN FLOAT,
            MONTHLY_RETURN FLOAT,
            YEARLY_RETURN FLOAT,
            VOLATILITY_21D FLOAT
        )
        """,
        f"""
        COPY INTO NVDA_STOCK_DATA
        FROM @{SNOWFLAKE_STAGE}/{base_filename}
        FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1)
        """
    ]

    print("⚙️ Executing SQL setup in Snowflake...")
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        role=SNOWFLAKE_ROLE,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    cursor = conn.cursor()

    try:
        for stmt in sql_commands:
            stmt_clean = stmt.strip()
            if stmt_clean:
                print(f"🔄 Executing: {stmt_clean.splitlines()[0].strip()}...")
                cursor.execute(stmt_clean)
                
                # If this is a query that returns results (like LIST), print them
                if cursor.description:
                    results = cursor.fetchall()
                    if results:
                        print("   Results:")
                        for row in results[:5]:  # Show first 5 results
                            print(f"   - {row}")
                        if len(results) > 5:
                            print(f"   ...and {len(results) - 5} more")
                
                print(f"✅ Executed successfully")
    except Exception as e:
        print(f"❌ Error executing setup: {e}")
    finally:
        cursor.close()
        conn.close()

# ---------- VERIFY DATA LOADED ---------- #
def verify_data_loaded():
    print("🔍 Verifying data loaded correctly...")
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        role=SNOWFLAKE_ROLE,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    cursor = conn.cursor()

    try:
        # Count rows
        cursor.execute("SELECT COUNT(*) FROM NVDA_STOCK_DATA")
        count = cursor.fetchone()[0]
        print(f"✅ Found {count} rows in NVDA_STOCK_DATA table")
        
        # Sample data
        cursor.execute("SELECT * FROM NVDA_STOCK_DATA ORDER BY DATE DESC LIMIT 5")
        rows = cursor.fetchall()
        print("📊 Recent data samples:")
        for row in rows:
            print(f"   {row[0]} | Close: ${row[1]:.2f} | Volume: {row[5]:.0f}")
            
    except Exception as e:
        print(f"❌ Error verifying data: {e}")
    finally:
        cursor.close()
        conn.close()

# ---------- MAIN EXECUTION ---------- #
if __name__ == "__main__":
    print_env_check()
    csv_file = find_latest_nvda_csv()
    upload_csv_to_stage(csv_file)
    run_sql_setup(csv_file)
    verify_data_loaded()
    print("🎯 All steps completed. NVIDIA stock data is live in Snowflake!")