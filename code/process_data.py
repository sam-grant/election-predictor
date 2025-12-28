""" Preprocess data for analysis """
from pathlib import Path
import pandas as pd
import sqlite3

def _resolve_path(relative_path):
    """ Helper to resolve relative paths """
    base_path = Path(__file__).parent
    return base_path / relative_path    

def process_election_data(file_path):
    """ Election winners """
    # Resolve the file path
    file_path = _resolve_path(file_path)
    # Election winners by year
    df_ele = pd.read_csv(file_path)
    # Filter before 1960 
    df_ele = df_ele[df_ele["Year"] >= 1960]
    # Add Label
    df_ele["Label"] = df_ele["Lean"].map({"Left": 0, "Right": 1})
    print("Processed election data\n")
    return df_ele

def process_wb_data(file_path):
    """
    World Bank indicators
    https://databank.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG/1ff4a498/Popular-Indicators
    """
    # Resolve the file path
    file_path = _resolve_path(file_path)
    # World Bank indicators
    df_wb = ( # DataFrames are builder objects, can chain methods
        pd.read_csv(file_path)
        .replace("..", pd.NA)  # Replace missing values
        .drop(["Series Code", "Country Name", "Country Code"], axis=1)  # Drop unneeded columns
        .rename(columns={"Series Name": "Year"})  # Rename for clarity
    )
    # Clean year formatting
    df_wb.columns = df_wb.columns.str.replace(r"\s\[YR\d{4}\]", "", regex=True) 
    # Transpose 
    df_wb = df_wb.transpose().reset_index()
    # Set first row as header
    df_wb.columns = df_wb.iloc[0]
    df_wb = df_wb.drop(0).reset_index() 
    # For the sake of this exercise, treat 2023 data as 2024 data
    df_wb.loc[df_wb["Year"] == "2023", "Year"] = "2024"
    # Convert Year to int 
    df_wb["Year"] = df_wb["Year"].astype(int)
    
    # Rescale Foreign direct investment from dollars to trillions 
    if "Foreign direct investment, net (BoP, current US$)" in df_wb.columns:
        df_wb["Foreign direct investment, net (BoP, current US$)"] = (
            pd.to_numeric(df_wb["Foreign direct investment, net (BoP, current US$)"], errors="coerce") / 1e12
        )
    
    print("Processed World Bank data\n")
    return df_wb

def process_fred_data(file_path):
    """
    FRED unemployment data
    https://fred.stlouisfed.org/series/UNRATE
    """
    # Resolve the file path
    file_path = _resolve_path(file_path)
    # Load FRED data 
    df_fred =(
        pd.read_csv(file_path)
        .rename(columns={"UNRATE": "Unemployment"})
    ) 
    # Extract year
    df_fred["observation_date"] = pd.to_datetime(
        df_fred["observation_date"],
        errors="coerce"
    )
    df_fred["Year"] = df_fred["observation_date"].dt.year
    # Filter to years >= 1960
    df_fred = df_fred[df_fred["Year"] >= 1960]
    # Get yearly average
    df_fred = (
        df_fred.groupby("Year")["Unemployment"]
        .mean()
        .reset_index()
    )
    print("Processed FRED unemployment data\n")
    return df_fred
    
def merge_on_year(dfs):
    """ Merge multiple DataFrames on Year """
    from functools import reduce
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on="Year", how="inner"),
        dfs
    )
    print("Merged data on year\n")
    return df_merged

def clean_and_interpolate(df, threshold=0.5): 
    """ Clean and interpolate missing values """
    # Create explicit copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Drop columns missing >50%
    df = df.dropna(axis=1, thresh=len(df)*threshold) 
          # f"{df.columns.tolist()}")

    # Ensure Year is int for interpolation
    df["Year"] = df["Year"].astype(int)

    # Convert numeric columns that are stored as objects
    # Skip truly categorical columns
    categorical_cols = {"Winner", "Party", "Lean", "Era"}
    for col in df.columns:
        if col not in categorical_cols and col != "Year" and df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Interpolate missing numeric value
    df = df.interpolate(method="linear", limit_direction="both")
    
    # Drop only numeric columns that still have missing values
    # Keep non-numeric columns (like Winner, Party) even if they have NaN
    numeric_cols = df.select_dtypes(include=["number"]).columns
    cols_to_drop = [col for col in numeric_cols if df[col].isnull().any()]
    df = df.drop(columns=cols_to_drop)

    # Check if any missing values remain
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} missing values remain after interpolation.")

    print(f"Cleaned and interpolated data with >{threshold*100}% missing values dropped\n"
          f"{len(df.columns)} columns remain\n")

    return df

def write(df, csv_output_path="../data/proc/data.csv",
           sql_output_path="../data/proc/data.sqlite3"):
    """ Write processed data to CSV and SQLite database """
    # Write to CSV
    csv_output_path = _resolve_path(csv_output_path)
    df.to_csv(csv_output_path, index=False)

    # Also make an SQLite database for fun
    sql_output_path = _resolve_path(sql_output_path)
    conn = sqlite3.connect(sql_output_path)
    df.to_sql("data", conn, if_exists="replace", index=False)
    conn.close()
    
    print(f"Processed data written to:")
    print(f"CSV output path: {csv_output_path}")
    print(f"SQLite output path: {sql_output_path}")

def main():
    # Run through pipeline
    df_ele = process_election_data("../data/raw/election_data.csv")
    df_wb = process_wb_data("../data/raw/world_bank_indicators_1960-2023.csv")
    df_fred = process_fred_data("../data/raw/FRED_unemployment_rate_1948-2024.csv")
    df_merged = merge_on_year([df_ele, df_wb, df_fred])
    df_cleaned = clean_and_interpolate(df_merged)
    
    print("Final processed data:")
    print(df_cleaned.head(), "\n")

    write(df_cleaned)

if __name__ == '__main__':
    main()