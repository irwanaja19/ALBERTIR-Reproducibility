from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

# Load model and tokenizer from Hugging Face Hub
print("🔽 Loading model from Hugging Face Hub...")
model_id = "irwan19/albertir-quran-hadith"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)
print("✅ Model loaded successfully!")

# Prepare example sentence with [MASK] token
text = "Umat Islam wajib melaksanakan ibadah [MASK] di bulan Ramadhan."

# Tokenize and predict
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# Get prediction for [MASK] token
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_token_id = predictions[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"\n📝 Original sentence: {text}")
print(f"✨ Prediction for [MASK]: '{predicted_token}'")

# Display top 5 predictions
print("\n🔍 Top 5 predicted words:")
top_k = torch.topk(predictions[0, mask_token_index], k=5)
for i, (score, token_id) in enumerate(zip(top_k.values[0], top_k.indices[0])):
    print(f"   {i+1}. {tokenizer.decode(token_id)} (score: {score.item():.4f})")
