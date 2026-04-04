from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import re
import io
from imblearn.over_sampling import RandomOverSampler

# Initialize the API app
app = FastAPI(
    title="Data Cleaning API",
    description="Upload an unclean CSV and get a fully cleaned CSV back.",
    version="1.0.0"
)

# DEFINE THE CLEANING FUNCTIONS

def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace(r'[^\w_]', '', regex=True)
    )
    return df

def remove_empty_garbage(df):
    return df.dropna(axis=0, how='all').dropna(axis=1, how='all')

def fill_missing_smartly(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', '', 'NaN'], 'Unknown')
            df[col] = df[col].fillna('Unknown')
    return df

def check_imbalance(df):
    # Note: Prints will show up in your server's console logs
    print("Scanning for Imbalanced Data...")
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if 1 < df[col].nunique() <= 10:
            val_counts = df[col].value_counts(normalize=True)
            if not val_counts.empty:
                top_class_pct = val_counts.iloc[0]
                if top_class_pct > 0.75:
                    print(f"    WARNING: '{col}' is heavily imbalanced! Top category is {top_class_pct:.1%} of the data.")
    return df 

def balance_target(df, target_column=None):
    if not target_column:
        return df

    target_column = target_column.lower().replace(" ", "_")

    if target_column in df.columns:
        print(f" Balancing dataset based on target: '{target_column}'...")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        print("   Dataset successfully balanced!")
        return pd.concat([X_resampled, y_resampled], axis=1)
    else:
        print(f"Wrong! Target column '{target_column}' not found. Skipping balancing step.")
        return df

# DEFINE THE API ENDPOINT

@app.post("/clean-dataset/")
async def clean_dataset(
    file: UploadFile = File(..., description="The unclean CSV file to process"),
    target_column: str = Form(None, description="Optional: Column name to balance")
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        # Read the uploaded file directly into memory (no saving to disk)
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), encoding='latin1')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    print("Running Pure Pandas Cleaning Pipeline...")

    # Execute the pipeline
    try:
        cleaned_df = (
            df
            .pipe(clean_column_names)
            .pipe(remove_empty_garbage)
            .drop_duplicates()
            .pipe(fill_missing_smartly)
            .pipe(check_imbalance)
            .pipe(balance_target, target_column=target_column)
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error during cleaning pipeline: {str(e)}")

    # Convert the cleaned dataframe back into a CSV format in memory
    stream = io.StringIO()
    cleaned_df.to_csv(stream, index=False)
    
    # Create an iterator from the stream string
    response_iter = iter([stream.getvalue()])

    # Return the file as a downloadable streaming response
    return StreamingResponse(
        response_iter,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=cleaned_{file.filename}"}
    )