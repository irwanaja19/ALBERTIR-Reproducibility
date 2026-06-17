# ALBERTIR-Reproducibility

Code for the continued pre-training of ALBERTIR on Quran and Hadith corpora.

---

## 🤗 Model di Hugging Face

The ALBERTIR model is available at:  
[https://huggingface.co/irwan19/albertir-quran-hadith](https://huggingface.co/irwan19/albertir-quran-hadith)

---

## 📂 Repository Structure

```
ALBERTIR-Reproducibility/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── train.py                     # Continued pre-training script
├── test_model.py                # Simple inference test
├── combined_corpus.txt          # Pretraining dataset (Quran + Hadith)
└── preprocess_khutbah.py        # PDF preprocessing (OCR + normalization)
```

---

## ⚙️ Installation

Install all required libraries:

```bash
pip install -r requirements.txt
```

**Note:** For preprocessing, you also need to install **Tesseract OCR**:
- **Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki
- **Linux:** `sudo apt-get install tesseract-ocr tesseract-ocr-ind`
- **Mac:** `brew install tesseract`

---

## 📄 Preprocessing (PDF → Text)

To process Friday Sermon (Khutbah) PDF files into clean text:

### For a single PDF file:

```python
from preprocess_khutbah import preprocess_khutbah

result = preprocess_khutbah('path/to/khutbah.pdf')
print(result['sentences'][:5])  # Print first 5 sentences
```

### For batch processing all PDFs in a folder:

```python
from preprocess_khutbah import batch_preprocess_khutbah

batch_preprocess_khutbah(
    input_folder='./khutbah_pdfs/',
    output_file='./preprocessed_sentences.txt'
)
```

### Preprocessing Steps:

1. **OCR Extraction:** Tesseract v5.3.0 with Indonesian language model (`-l ind`)
2. **Text Normalization:** 
   - Removal of non-printable characters
   - Diacritic removal (harakat from Arabic)
   - Standardization of Arabic-derived terms (e.g., 'Allah' vs 'Alloh' → 'Allah')
   - Common OCR error correction
3. **Sentence Segmentation:** NLTK Indonesian sentence tokenizer

---

## 🧪 Testing the Model (Quick Check)

**To verify the model works correctly, run:**

```bash
python test_model.py
```

**Or copy and run this code directly in Google Colab (no GPU needed):**

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

print("🔽 Loading model from Hugging Face Hub...")
model_id = "irwan19/albertir-quran-hadith"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)
print("✅ Model loaded successfully!")

text = "Umat Islam wajib melaksanakan ibadah [MASK] di bulan Ramadhan."

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_token_id = predictions[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"\n📝 Original sentence: {text}")
print(f"✨ Prediction for [MASK]: '{predicted_token}'")

print("\n🔍 Top 5 predicted words:")
top_k = torch.topk(predictions[0, mask_token_index], k=5)
for i, (score, token_id) in enumerate(zip(top_k.values[0], top_k.indices[0])):
    print(f"   {i+1}. {tokenizer.decode(token_id)} (score: {score.item():.4f})")
```

**Expected output:** The model will predict `puasa` (fasting) or similar Islamic-related words.

---

## 🚀 Training (Reproducing the Model from Scratch)

**If you want to reproduce the training process:**

```bash
python train.py
```

**Note:** The `combined_corpus.txt` file must be in the same directory. This file contains 18,606 verses/hadith from Quran, Sahih Bukhari, and Sahih Muslim in Indonesian translation.

---

## 📚 Dataset Information

The pretraining dataset consists of:
- Quran (Indonesian translation): 6,236 verses
- Sahih Bukhari (Indonesian translation): 7,008 hadith  
- Sahih Muslim (Indonesian translation): 5,362 hadith
- **Total: 18,606 documents**

---

## 📝 Pretraining Details

| Parameter | Value |
|-----------|-------|
| Base Model | `indobenchmark/indobert-base-p1` |
| Total Corpus | 18,606 documents |
| Train Samples | 16,747 |
| Eval Samples | 1,861 |
| Epochs | 10 |
| Batch Size | 8 |
| Learning Rate | 5e-5 |
| Optimizer | AdamW |
| Total Steps | 10,470 |
| Final Training Loss | ~1.2473 |
| Final Validation Loss | ~1.1415 |

---

## 📖 Citation

If you use this code or model, please cite our paper:

```
I. Darmawan, H. Elmunsyah, and D. D. Prasetya, "ALBERTIR: A BERT-Based Pretraining for Indonesian Religious Texts Using Qur'an and Hadith Translations", Eng. Technol. Appl. Sci. Res., vol. 15, no. 5, pp. 28307–28312, Oct. 2025. doi: 10.48084/etasr.12977
```

---

## 📄 License

MIT License
