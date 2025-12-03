"""
Utility functions for Book Summarization Streamlit App
Handles model loading, text processing, and evaluation
"""

import torch
import streamlit as st
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    pipeline
)
from peft import PeftModel, PeftConfig
import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import logging
import re
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_model_and_tokenizer(model_name: str, model_config: Dict, device: str = 'cpu'):
    """
    Load model and tokenizer with caching
    
    Args:
        model_name: Name of the model
        model_config: Configuration dictionary from config.py
        device: Device to load model on ('cpu', 'cuda', 'mps')
    
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        logger.info(f"Loading model: {model_name}")
        
        model_path = model_config['path']
        base_model_name = model_config['base_model']
        model_type = model_config['type']
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load model based on type
        if model_type == 'lora':
            # Load base model first
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32
            )
            
            # Load LoRA adapter
            if Path(model_path).exists():
                model = PeftModel.from_pretrained(base_model, model_path)
            else:
                logger.warning(f"LoRA adapter not found at {model_path}, using base model")
                model = base_model
        else:
            # Load fine-tuned model directly
            if Path(model_path).exists():
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32
                )
            else:
                logger.warning(f"Model not found at {model_path}, using base model")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32
                )
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        logger.info(f"Successfully loaded {model_name} on {device}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None, None


def preprocess_text(text: str) -> str:
    """
    Clean and prepare text for summarization
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
    
    # Trim
    text = text.strip()
    
    return text


def truncate_text(text: str, tokenizer, max_length: int = 1024) -> str:
    """
    Truncate text to maximum token length
    
    Args:
        text: Input text
        tokenizer: Tokenizer instance
        max_length: Maximum number of tokens
    
    Returns:
        Truncated text
    """
    tokens = tokenizer.encode(text, truncation=False)
    
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        
    return text


def generate_summary(
    text: str, 
    model, 
    tokenizer, 
    gen_params: Dict,
    device: str = 'cpu'
) -> str:
    """
    Generate summary for given text
    
    Args:
        text: Input text to summarize
        model: Model instance
        tokenizer: Tokenizer instance
        gen_params: Generation parameters
        device: Device to run on
    
    Returns:
        Generated summary
    """
    try:
        # Preprocess text
        text = preprocess_text(text)
        
        # Truncate if needed
        text = truncate_text(text, tokenizer, max_length=1024)
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=gen_params.get('max_length', 500),
                min_length=gen_params.get('min_length', 100),
                num_beams=gen_params.get('num_beams', 4),
                length_penalty=gen_params.get('length_penalty', 2.0),
                no_repeat_ngram_size=gen_params.get('no_repeat_ngram_size', 3),
                early_stopping=gen_params.get('early_stopping', True),
                temperature=gen_params.get('temperature', 1.0),
                do_sample=gen_params.get('do_sample', False)
            )
        
        # Decode
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise e


@st.cache_resource
def load_metrics():
    """Load evaluation metrics with caching"""
    try:
        rouge = evaluate.load('rouge')
        bertscore = evaluate.load('bertscore')
        return rouge, bertscore
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        return None, None


@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer for semantic similarity"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        logger.error(f"Error loading sentence transformer: {str(e)}")
        return None


def calculate_rouge_scores(reference: str, generated: str) -> Dict:
    """
    Calculate ROUGE scores
    
    Args:
        reference: Reference summary
        generated: Generated summary
    
    Returns:
        Dictionary of ROUGE scores
    """
    try:
        rouge, _ = load_metrics()
        
        if rouge is None:
            return {}
        
        results = rouge.compute(
            predictions=[generated],
            references=[reference],
            use_stemmer=True
        )
        
        return {
            'rouge1': results['rouge1'],
            'rouge2': results['rouge2'],
            'rougeL': results['rougeL']
        }
        
    except Exception as e:
        logger.error(f"Error calculating ROUGE: {str(e)}")
        return {}


def calculate_bertscore(reference: str, generated: str) -> Dict:
    """
    Calculate BERTScore
    
    Args:
        reference: Reference summary
        generated: Generated summary
    
    Returns:
        Dictionary of BERTScore metrics
    """
    try:
        _, bertscore = load_metrics()
        
        if bertscore is None:
            return {}
        
        results = bertscore.compute(
            predictions=[generated],
            references=[reference],
            lang='en',
            model_type='distilbert-base-uncased'
        )
        
        return {
            'precision': float(results['precision'][0]),
            'recall': float(results['recall'][0]),
            'f1': float(results['f1'][0])
        }
        
    except Exception as e:
        logger.error(f"Error calculating BERTScore: {str(e)}")
        return {}


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity using sentence transformers
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score (0-1)
    """
    try:
        model = load_sentence_transformer()
        
        if model is None:
            return 0.0
        
        # Encode sentences
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        
        return float(similarity[0][0])
        
    except Exception as e:
        logger.error(f"Error calculating semantic similarity: {str(e)}")
        return 0.0


def calculate_all_metrics(reference: str, generated: str) -> Dict:
    """
    Calculate all evaluation metrics
    
    Args:
        reference: Reference summary
        generated: Generated summary
    
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # ROUGE scores
    rouge_scores = calculate_rouge_scores(reference, generated)
    metrics.update(rouge_scores)
    
    # BERTScore
    bertscore_scores = calculate_bertscore(reference, generated)
    metrics.update({f'bertscore_{k}': v for k, v in bertscore_scores.items()})
    
    # Semantic similarity
    similarity = calculate_semantic_similarity(reference, generated)
    metrics['semantic_similarity'] = similarity
    
    # Length metrics
    metrics['generated_length'] = len(generated.split())
    metrics['reference_length'] = len(reference.split())
    metrics['compression_ratio'] = len(generated) / max(len(reference), 1)
    
    return metrics


def format_metrics_display(metrics: Dict) -> str:
    """
    Format metrics for display
    
    Args:
        metrics: Dictionary of metrics
    
    Returns:
        Formatted string
    """
    output = "### 📊 Evaluation Metrics\n\n"
    
    # ROUGE scores
    if 'rouge1' in metrics:
        output += "**ROUGE Scores:**\n"
        output += f"- ROUGE-1: {metrics['rouge1']:.4f}\n"
        output += f"- ROUGE-2: {metrics['rouge2']:.4f}\n"
        output += f"- ROUGE-L: {metrics['rougeL']:.4f}\n\n"
    
    # BERTScore
    if 'bertscore_f1' in metrics:
        output += "**BERTScore:**\n"
        output += f"- Precision: {metrics['bertscore_precision']:.4f}\n"
        output += f"- Recall: {metrics['bertscore_recall']:.4f}\n"
        output += f"- F1: {metrics['bertscore_f1']:.4f}\n\n"
    
    # Semantic similarity
    if 'semantic_similarity' in metrics:
        output += f"**Semantic Similarity:** {metrics['semantic_similarity']:.4f}\n\n"
    
    # Length metrics
    if 'generated_length' in metrics:
        output += "**Length Statistics:**\n"
        output += f"- Generated: {metrics['generated_length']} words\n"
        output += f"- Reference: {metrics['reference_length']} words\n"
        output += f"- Compression: {metrics['compression_ratio']:.2%}\n"
    
    return output


def extract_text_from_file(uploaded_file) -> Optional[str]:
    """
    Extract text from uploaded file
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Extracted text or None
    """
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.txt':
            # Read text file
            text = uploaded_file.read().decode('utf-8')
            return text
            
        elif file_extension == '.pdf':
            # For PDF, would need PyPDF2 or similar
            st.warning("PDF support requires additional libraries. Please use .txt files.")
            return None
            
        elif file_extension == '.docx':
            # For DOCX, would need python-docx
            st.warning("DOCX support requires additional libraries. Please use .txt files.")
            return None
            
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}")
        st.error(f"Failed to extract text: {str(e)}")
        return None


def get_text_statistics(text: str) -> Dict:
    """
    Get statistics about the text
    
    Args:
        text: Input text
    
    Returns:
        Dictionary of statistics
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        'characters': len(text),
        'words': len(words),
        'sentences': len([s for s in sentences if s.strip()]),
        'avg_word_length': sum(len(w) for w in words) / max(len(words), 1)
    }
