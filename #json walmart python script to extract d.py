#json walmart V2 Merged python script to extract dataframe metadata and save as JSON
#Using the merged_walmart_state_data.csv file, create a python script that extracts the dataframe metadata (column names, data types, number of rows) and saves it as a JSON file named walmart_metadata.json.
import pandas as pd
import json

def int_to_categorical(df, max_unique_values = 10):
    """ Parmeter:
    - df :  Input pd.DataFrame
    - max_unique_values: int, max number of unique values
    """

    df_copy = df.copy()
    n_rows = len(df_copy)

    for col in df_copy.select_dtypes(
        include = ['int', 'int64', 'int32']
    ).columns :
        n_unique = df_copy[col].nunique()

        if n_unique <= max_unique_values:
            df_copy[col] = df_copy[col].astype('category')

    return df_copy



# Step 1: Load your CSV
walmart_df = pd.read_csv("Walmart.csv")

walmart_df['Date'] = pd.to_datetime(walmart_df['Date'], dayfirst=True)


# Change relevant integer columns
walmart_df_new = int_to_categorical(walmart_df)



# Step 2: Extract dtypes into a dictionary
dtypes = walmart_df_new.dtypes.apply(lambda dt: dt.name).to_dict()

# Optional: convert pandas types to more universal/JSON-friendly strings
# For example, 'int64' ➜ 'int64', 'float64' ➜ 'float64', etc.
# You can skip this if you're happy with how they are

# Step 3: Create metadata dictionary
metadata = {
    "dtypes": dtypes,
    "columns": list(walmart_df_new.columns),
    "n_rows": len(walmart_df_new)
}

# Step 4: Save to JSON
with open("walmart_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ Metadata saved as JSON!")
