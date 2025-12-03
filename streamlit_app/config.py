"""
Configuration file for Automated Book Summarization using Generative AI
Contains model paths, generation parameters, and application settings

This configuration supports multiple transfer learning approaches:
- BART with full fine-tuning
- BART with LoRA (parameter-efficient)
- Longformer for long-text sequences
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

# Available models (Transfer Learning Architectures)
MODELS = {
    "BART (Week 5)": {
        "path": str(MODELS_DIR / "week5_bart"),
        "base_model": "facebook/bart-large-cnn",
        "type": "bart",
        "description": "BART-large fine-tuned with LoRA for generative summarization",
        "metrics": {
            "rouge1": 0.2114,
            "rouge2": 0.0279,
            "rougeL": 0.1280,
            "bertscore_f1": 0.8195
        }
    },
    "BART Improved (Week 4)": {
        "path": str(MODELS_DIR / "bart_improved" / "checkpoint-1536"),
        "base_model": "facebook/bart-large-cnn",
        "type": "bart",
        "description": "Improved BART checkpoint with optimized hyperparameters",
        "metrics": {
            "rouge1": 0.205,
            "rouge2": 0.025,
            "rougeL": 0.125
        }
    },
    "LoRA Adapter (Week 4)": {
        "path": str(MODELS_DIR / "week4_lora"),
        "base_model": "facebook/bart-large-cnn",
        "type": "lora",
        "description": "LoRA adapter for efficient fine-tuning",
        "metrics": {
            "rouge1": 0.198,
            "rouge2": 0.022,
            "rougeL": 0.120
        }
    }
}

# Generation parameters with descriptions
GEN_PARAMS = {
    "max_length": {
        "default": 500,
        "min": 100,
        "max": 1000,
        "step": 50,
        "description": "Maximum length of generated summary"
    },
    "min_length": {
        "default": 100,
        "min": 50,
        "max": 500,
        "step": 50,
        "description": "Minimum length of generated summary"
    },
    "num_beams": {
        "default": 4,
        "min": 1,
        "max": 8,
        "step": 1,
        "description": "Number of beams for beam search (higher = better quality but slower)"
    },
    "length_penalty": {
        "default": 2.0,
        "min": 0.5,
        "max": 5.0,
        "step": 0.5,
        "description": "Penalty for length (higher = prefer longer summaries)"
    },
    "temperature": {
        "default": 1.0,
        "min": 0.1,
        "max": 2.0,
        "step": 0.1,
        "description": "Sampling temperature (lower = more focused, higher = more diverse)"
    },
    "no_repeat_ngram_size": {
        "default": 3,
        "min": 0,
        "max": 5,
        "step": 1,
        "description": "Prevent repetition of n-grams"
    }
}

# Default generation parameters
DEFAULT_GEN_PARAMS = {
    "max_length": 500,
    "min_length": 100,
    "num_beams": 4,
    "length_penalty": 2.0,
    "temperature": 1.0,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "do_sample": False
}

# Sample texts for testing
SAMPLE_TEXTS = {
    "Short Fiction": """
    In a small village nestled among rolling hills, there lived a young girl named Elena who had an 
    extraordinary gift. She could understand the language of birds. Every morning, she would sit by 
    her window and listen to the sparrows, robins, and crows tell her stories of the world beyond 
    the hills. One day, a majestic eagle landed on her windowsill with news that would change 
    everything. The king of a distant land was searching for someone who could communicate with 
    nature itself. Elena's journey to the palace was filled with adventures, and she discovered 
    that her gift was just the beginning of a much greater destiny.
    """,
    
    "Classic Literature": """
    It is a truth universally acknowledged, that a single man in possession of a good fortune must 
    be in want of a wife. However little known the feelings or views of such a man may be on his 
    first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding 
    families, that he is considered the rightful property of some one or other of their daughters. 
    The arrival of Mr. Bingley at Netherfield Park caused a great stir in the county, and Mrs. 
    Bennet was particularly excited at the prospect of marrying one of her five daughters to this 
    wealthy bachelor. Her husband, Mr. Bennet, was more reserved in his enthusiasm, preferring to 
    observe the situation with his characteristic dry wit.
    """,
    
    "Science Fiction": """
    The year was 2247, and humanity had finally achieved what was once thought impossible: faster-than-light 
    travel. Captain Sarah Chen stood on the bridge of the starship Endeavor, watching as the stars 
    streaked past the viewscreen. They were on a mission to explore a newly discovered planet in the 
    Andromeda galaxy, one that showed signs of intelligent life. As they approached the planet, strange 
    energy readings began to appear on the sensors. The crew was about to make first contact with an 
    alien civilization, but they had no idea that this encounter would reveal a shocking truth about 
    humanity's own origins and challenge everything they thought they knew about the universe.
    """
}

# UI Configuration
APP_TITLE = "📚 Book Summarization Application"
APP_ICON = "📚"
SIDEBAR_TITLE = "⚙️ Settings"

# Color scheme
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#ff7f0e",
    "error": "#d62728"
}

# Metric configurations
METRICS_CONFIG = {
    "rouge": {
        "display_name": "ROUGE",
        "types": ["rouge1", "rouge2", "rougeL"],
        "descriptions": {
            "rouge1": "Overlap of unigrams",
            "rouge2": "Overlap of bigrams",
            "rougeL": "Longest common subsequence"
        }
    },
    "bertscore": {
        "display_name": "BERTScore",
        "types": ["precision", "recall", "f1"],
        "model": "distilbert-base-uncased"
    }
}

# Device configuration
def get_device():
    """Determine the best available device"""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

DEVICE = get_device()

# Cache configuration
CACHE_TTL = 3600  # 1 hour in seconds
MAX_CACHE_SIZE = 100  # Maximum number of cached results

# File upload configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = ['.txt', '.pdf', '.docx']

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
