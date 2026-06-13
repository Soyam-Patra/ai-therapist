import pandas as pd
import json
from bs4 import BeautifulSoup
import html
import re

# Load dataset
df = pd.read_csv("data/counselchat-data.csv")

print("Original rows:", len(df))

# Keep only useful columns
df = df[["questionText", "answerText", "topics"]]

# Remove NaN rows
df = df.dropna()

# HTML cleaning
def clean_text(text):

    # Convert html entities
    text = html.unescape(str(text))

    # Remove html tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["questionText"] = df["questionText"].apply(clean_text)
df["answerText"] = df["answerText"].apply(clean_text)

# Remove very short examples
df = df[
    (df["questionText"].str.len() > 20)
    &
    (df["answerText"].str.len() > 50)
]

print("Rows after cleaning:", len(df))

# Create instruction format
formatted_data = []

for _, row in df.iterrows():

    text = f"""### Instruction:{row['questionText']}

### Response:
{row['answerText']}"""

    formatted_data.append({"text": text})

# Save JSON
with open("therapist_sft.json", "w", encoding="utf-8") as f:
    json.dump(
        formatted_data,
        f,
        ensure_ascii=False,
        indent=2
    )

print("Saved:", len(formatted_data), "examples")