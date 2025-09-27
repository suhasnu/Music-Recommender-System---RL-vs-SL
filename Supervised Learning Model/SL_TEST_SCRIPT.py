"""""
---------------------------------------------------------------------------------------------------------------------
## How to Run the SL Model:
### Step 1 :
  Download the folder named Sl from the drive link shared, then unzip the folder.

### Step 2 :
  Open the file named SL_TEST_SCRIPT.py .

### Step 3 :
  Run the script, after running check the output window.
  There will be to option to enter : quit or next.
  If you enter next it will show you another output.
  if you enter quit it will exit and end the script.

--------------------------------------------------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import joblib
import json
import os

# Define the model class
class BertGenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertGenreClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-large-uncased")  # Match training config
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# Force CPU usage (even if CUDA is available)
device = torch.device("cpu")

# Load label encoder
label_encoder = joblib.load("label_encoder.pkl")
num_classes = len(label_encoder.classes_)

# Load model
model = BertGenreClassifier(num_classes)
model.load_state_dict(torch.load("SL_MODEL.pt", map_location=device))
model.to(device)
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

# Predict from one sample
text = "An emotional rock ballad with heavy guitar and slow tempo"

# Tokenize
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Predict genre
with torch.no_grad():
    outputs = model(input_ids, attention_mask)
    predicted_class = torch.argmax(outputs, dim=1).item()

predicted_genre = label_encoder.inverse_transform([predicted_class])[0]
print("🎵 Predicted Genre:", predicted_genre)

# Load genre-to-songs mapping
genre_to_songs = {}
if os.path.exists("genre_to_songs.json"):
    with open("genre_to_songs.json", "r", encoding="utf-8") as f:
        genre_to_songs = json.load(f)
else:
    print("⚠️ genre_to_songs.json not found.")

# Load descriptions
descriptions = []
if os.path.exists("descriptions.txt"):
    with open("descriptions.txt", "r", encoding="utf-8") as f:
        descriptions = [line.strip() for line in f if line.strip()]
else:
    print("❌ descriptions.txt not found.")

# Function to predict genre and recommend songs
def process_description(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predicted_class = torch.argmax(outputs, dim=1).item()

    predicted_genre = label_encoder.inverse_transform([predicted_class])[0]
    print("\n🎵 Description:", text)
    print("🎵 Predicted Genre:", predicted_genre)

    recommended_songs = genre_to_songs.get(predicted_genre.lower(), [])
    if recommended_songs:
        print("🎧 Recommended Songs:")
        for song in recommended_songs[:5]:
            print("•", song)
    else:
        print("❌ No songs found for this genre.")

# Interactive loop
index = 0
while index < len(descriptions):
    process_description(descriptions[index])
    index += 1

    if index < len(descriptions):
        user_input = input("\nType 'next' to see the next recommendation, or anything else to quit: ").strip().lower()
        if user_input != "next":
            break
    else:
        print("\n✅ No more descriptions to process.")
