from groq import Groq
from app.core.config import settings
import pandas as pd
import os

client = Groq(api_key=settings.GROQ_API_KEY)

UPLOAD_DIR = "uploads"

def get_dataset_context(filename: str) -> str:
    filepath = os.path.join(UPLOAD_DIR, filename)
    df = pd.read_csv(filepath)
    
    context_parts = []
    context_parts.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")
    context_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
    
    # Null analysis across entire dataset
    context_parts.append("\nNull value analysis (entire dataset):")
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round(null_count / len(df) * 100, 1)
        context_parts.append(f"  - {col}: {null_count} nulls ({null_pct}%)")
    
    # Numeric column statistics across entire dataset
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        context_parts.append("\nNumeric column statistics (entire dataset):")
        for col in numeric_cols[:10]:
            context_parts.append(
                f"  - {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
                f"mean={df[col].mean():.2f}, median={df[col].median():.2f}, "
                f"std={df[col].std():.2f}"
            )
    
    # Categorical column analysis
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if text_cols:
        context_parts.append("\nCategorical column analysis (entire dataset):")
        for col in text_cols[:5]:
            unique_count = df[col].nunique()
            context_parts.append(f"  - {col}: {unique_count} unique values")
            if unique_count <= 20:
                value_counts = df[col].value_counts().head(10)
                for val, cnt in value_counts.items():
                    context_parts.append(f"      '{val}': {cnt} rows ({round(cnt/len(df)*100,1)}%)")
    
    # Sample rows
    context_parts.append("\nFirst 3 rows (sample):")
    context_parts.append(df.head(3).to_string())
    
    return "\n".join(context_parts)


def chat_with_data(filename: str, user_question: str, chat_history: list) -> str:
    dataset_context = get_dataset_context(filename)
    
    system_prompt = f"""You are a data analyst assistant. The user has uploaded a dataset.
Here is the dataset summary:

{dataset_context}

Answer the user's questions about this data. Be specific and use actual numbers.
Keep answers concise and useful."""

    # Build messages list for Groq
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history
    for msg in chat_history:
        role = msg.get("role", "user")
        if role == "model":
            role = "assistant"
        text = msg.get("parts", [{}])[0].get("text", "")
        messages.append({"role": role, "content": text})
    
    # Add current question
    messages.append({"role": "user", "content": user_question})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1000
    )
    
    return response.choices[0].message.content
