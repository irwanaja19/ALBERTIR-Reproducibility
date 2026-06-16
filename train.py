import torch
import os
os.environ["WANDB_DISABLED"] = "true"
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

print("📖 Loading cleaned Quran and Hadith corpus...")
# Ganti dengan path file korpus Anda
with open('combined_corpus.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"📊 Total documents: {len(texts)}")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=128,
        return_special_tokens_mask=True,
    )

dataset = Dataset.from_dict({"text": texts})
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

print("🧠 Loading BERT model...")
model = AutoModelForMaskedLM.from_pretrained("indobenchmark/indobert-base-p1")
print(f"📐 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

training_args = TrainingArguments(
    output_dir="./model_output",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    logging_steps=100,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    prediction_loss_only=True,
    remove_unused_columns=False,
)

train_size = int(0.9 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

print("🎯 Starting continued pre-training...")
trainer.train()

trainer.save_model("./model_output/final_model")
tokenizer.save_pretrained("./model_output/final_model")
print("✅ Training completed! Model saved.")
