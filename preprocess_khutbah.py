"""
Preprocessing Friday Sermon (Khutbah) PDFs
- OCR: Tesseract v5.3.0 with Indonesian language model
- Normalization: diacritic removal, standardization of Arabic-derived terms
- Sentence segmentation: NLTK Indonesian sentence tokenizer

Author: Irwan Darmawan
"""

import os
import re
import pytesseract
from pdf2image import convert_from_path
import nltk

# Download NLTK data (only once)
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize


# ============================================
# 1. OCR EXTRACTION
# ============================================

def extract_text_with_ocr(pdf_path, dpi=300, lang='ind'):
    """
    Extract text from PDF using Tesseract OCR
    
    Args:
        pdf_path (str): Path to PDF file
        dpi (int): Resolution for image conversion (default: 300)
        lang (str): OCR language model (default: 'ind' for Indonesian)
    
    Returns:
        str: Extracted text from all pages
    """
    print(f"  OCR Processing: {os.path.basename(pdf_path)}")
    
    # Convert PDF pages to images
    images = convert_from_path(pdf_path, dpi=dpi)
    
    full_text = ""
    for i, image in enumerate(images):
        # OCR with Tesseract using Indonesian language model
        text = pytesseract.image_to_string(image, lang=lang)
        full_text += text + "\n"
    
    return full_text


# ============================================
# 2. TEXT NORMALIZATION
# ============================================

def normalize_khutbah_text(text):
    """
    Normalize extracted text from OCR
    
    Steps:
    1. Remove non-printable characters
    2. Remove excessive whitespace
    3. Remove diacritics (harakat from Arabic)
    4. Standardize Arabic-derived terms
    5. Fix common OCR errors
    
    Args:
        text (str): Raw text from OCR
    
    Returns:
        str: Cleaned and normalized text
    """
    
    # 1. Remove non-printable characters
    text = re.sub(r'[^\x00-\x7F\x80-\xFF]', ' ', text)
    
    # 2. Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Remove diacritics (harakat from Arabic)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # 4. Standardize Arabic-derived terms
    replacements = {
        'Alloh': 'Allah',
        'Allah': 'Allah',
        'Quran': 'Al-Quran',
        'Alquran': 'Al-Quran',
        'Al Qur\'an': 'Al-Quran',
        'Rasul': 'Rasulullah',
        'Nabi': 'Nabi Muhammad',
        'Hadis': 'Hadith',
        'Hadits': 'Hadith'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 5. Fix common OCR errors
    ocr_fixes = {
        'ان': 'أن',
        'لم': 'لَم',
        'ال': 'ال',
        'ة': 'ة'
    }
    for old, new in ocr_fixes.items():
        text = text.replace(old, new)
    
    # Remove multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ============================================
# 3. SENTENCE SEGMENTATION
# ============================================

def segment_sentences(text):
    """
    Segment text into sentences using NLTK
    
    Args:
        text (str): Normalized text
    
    Returns:
        list: List of sentences
    """
    return sent_tokenize(text, language='indonesian')


# ============================================
# 4. COMPLETE PREPROCESSING PIPELINE
# ============================================

def preprocess_khutbah(pdf_path, verbose=True):
    """
    Complete preprocessing pipeline:
    PDF → OCR → Normalization → Sentence Segmentation
    
    Args:
        pdf_path (str): Path to PDF file
        verbose (bool): Print progress messages
    
    Returns:
        dict: {
            'raw_text': str,
            'normalized_text': str,
            'sentences': list,
            'filename': str,
            'num_sentences': int
        }
    """
    filename = os.path.basename(pdf_path)
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Processing: {filename}")
        print(f"{'='*50}")
    
    # Step 1: OCR Extraction
    if verbose:
        print("  [1/3] Extracting text with OCR...")
    raw_text = extract_text_with_ocr(pdf_path)
    
    if not raw_text or len(raw_text.strip()) < 10:
        print(f"  ⚠️ Warning: No text extracted from {filename}")
        return None
    
    # Step 2: Normalization
    if verbose:
        print(f"  [2/3] Normalizing text... (raw: {len(raw_text)} chars)")
    normalized_text = normalize_khutbah_text(raw_text)
    
    # Step 3: Sentence Segmentation
    if verbose:
        print(f"  [3/3] Segmenting sentences... (normalized: {len(normalized_text)} chars)")
    sentences = segment_sentences(normalized_text)
    
    if verbose:
        print(f"  ✅ Done! Extracted {len(sentences)} sentences")
    
    return {
        'filename': filename,
        'raw_text': raw_text,
        'normalized_text': normalized_text,
        'sentences': sentences,
        'num_sentences': len(sentences)
    }


# ============================================
# 5. BATCH PROCESSING
# ============================================

def batch_preprocess_khutbah(input_folder, output_file, verbose=True):
    """
    Process all PDF files in a folder
    
    Args:
        input_folder (str): Folder containing PDF files
        output_file (str): Output file path (.txt)
        verbose (bool): Print progress
    
    Returns:
        list: List of all extracted sentences
    """
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if verbose:
        print(f"\n📂 Found {len(pdf_files)} PDF files in: {input_folder}")
    
    all_sentences = []
    failed_files = []
    
    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(input_folder, filename)
        
        if verbose and i % 10 == 0:
            print(f"  Processing file {i+1}/{len(pdf_files)}...")
        
        result = preprocess_khutbah(pdf_path, verbose=False)
        
        if result:
            all_sentences.extend(result['sentences'])
        else:
            failed_files.append(filename)
    
    # Save all sentences to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in all_sentences:
            f.write(sentence + '\n')
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"✅ BATCH PROCESSING COMPLETE!")
        print(f"   Total PDFs: {len(pdf_files)}")
        print(f"   Processed: {len(pdf_files) - len(failed_files)}")
        print(f"   Failed: {len(failed_files)}")
        print(f"   Total sentences: {len(all_sentences)}")
        print(f"   Output saved to: {output_file}")
        
        if failed_files:
            print(f"   Failed files: {failed_files}")
    
    return all_sentences


# ============================================
# 6. USAGE EXAMPLES
# ============================================

if __name__ == "__main__":
    
    # Example 1: Process a single PDF
    # result = preprocess_khutbah('sample_khutbah.pdf')
    # print(result['sentences'][:5])  # Print first 5 sentences
    
    # Example 2: Batch process all PDFs
    # batch_preprocess_khutbah(
    #     input_folder='./khutbah_pdfs/',
    #     output_file='./preprocessed_sentences.txt'
    # )
    
    print("✅ Preprocessing script loaded!")
    print("\nTo use:")
    print("  - For single file: preprocess_khutbah('path/to/file.pdf')")
    print("  - For batch: batch_preprocess_khutbah('pdf_folder/', 'output.txt')")