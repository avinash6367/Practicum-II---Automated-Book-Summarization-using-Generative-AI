# 📚 Book Summarization Streamlit Application

An interactive web application demonstrating **automated book summarization** using **Generative AI and Transfer Learning** techniques. This application showcases fine-tuned transformer models (BART, Longformer) with LoRA adapters that generate **coherent, contextually accurate summaries** of long-form books.

## 🎯 Project Overview

This application addresses key challenges in book summarization:
- **Traditional extractive methods** fail to capture narrative flow
- **Abstractive methods** often lack factual accuracy  
- **Long-form texts** require specialized model architectures

The solution employs **transfer learning** by fine-tuning pre-trained language models on open-source book summarization datasets, enabling effective handling of long text sequences and generation of high-quality abstractive summaries.

## 🌟 Features

- **Multiple Model Support**: Compare different transfer learning architectures
  - BART (Week 5) - Best performance with full fine-tuning
  - BART Improved (Week 4) - Optimized checkpoint
  - LoRA Adapter (Week 4) - Parameter-efficient fine-tuning

- **Flexible Input Methods**:
  - Direct text input for book chapters or passages
  - File upload (.txt) for longer documents
  - Pre-loaded sample texts (Fiction, Classic Literature, Sci-Fi)

- **Generative AI Configuration**:
  - Adjustable summary length (100-1000 tokens)
  - Beam search for quality optimization (1-8 beams)
  - Length penalty controls for summary characteristics
  - N-gram repetition prevention for coherence

- **Model Comparison**:
  - Side-by-side comparison of transfer learning approaches
  - Performance metrics visualization across architectures
  - Speed vs. quality trade-off analysis
  - Effectiveness demonstration for different model types

- **Comprehensive Evaluation Metrics**:
  - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for n-gram overlap
  - BERTScore (Precision, Recall, F1) for semantic accuracy
  - Semantic similarity using sentence transformers
  - Compression ratio and text statistics
  - Factual accuracy assessment

- **Interactive Visualizations**:
  - Metric comparison charts
  - Model performance radar plots
  - Real-time statistics

## 📋 Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- GPU/MPS support optional but recommended for faster inference

## 🚀 Installation

### 1. Clone the Repository

```bash
cd /path/to/book_summ
```

### 2. Install Dependencies

```bash
pip install -r streamlit_app/requirements.txt
```

### 3. Verify Model Files

Ensure your trained models are in the `results/` directory:

```
results/
├── week5_bart/           # BART fine-tuned model
├── bart_improved/        # Improved BART checkpoint
│   └── checkpoint-1536/
└── week4_lora/          # LoRA adapter
```

## 🎮 Usage

### Running Locally

```bash
# From project root
streamlit run streamlit_app/app.py
```

The app will open in your browser at `http://localhost:8501`

### Running on Specific Port

```bash
streamlit run streamlit_app/app.py --server.port 8080
```

### Running on Network

```bash
streamlit run streamlit_app/app.py --server.address 0.0.0.0
```

Access from other devices: `http://<your-ip>:8501`

## 📖 How to Use

### 1. Single Model Summarization

1. **Select Model**: Choose a model from the sidebar
2. **Configure Parameters**: Adjust generation settings (optional)
3. **Input Text**: 
   - Type/paste text directly
   - Upload a .txt file
   - Select a sample text
4. **Generate**: Click "Generate Summary"
5. **View Results**: See generated summary and statistics
6. **Evaluate** (Optional): Provide reference summary for metrics

### 2. Compare Models

1. Navigate to the "Compare Models" tab
2. Select 2 or more models
3. Enter text to summarize
4. Click "Compare Models"
5. View side-by-side results and comparison metrics

### 3. View Metrics

1. Navigate to the "Metrics" tab
2. View model performance comparisons
3. Explore interactive visualizations
4. Compare ROUGE, BERTScore, and other metrics

## ⚙️ Configuration

### Model Configuration

Edit `streamlit_app/config.py` to:
- Add/remove models
- Update model paths
- Modify default parameters
- Change sample texts

### Generation Parameters

Available in the sidebar:
- **Max Length**: 100-1000 tokens
- **Min Length**: 50-500 tokens
- **Number of Beams**: 1-8
- **Length Penalty**: 0.5-5.0
- **No Repeat N-gram Size**: 0-5

## 🏗️ Project Structure

```
streamlit_app/
├── app.py              # Main Streamlit application
├── utils.py            # Utility functions (model loading, metrics)
├── config.py           # Configuration and constants
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🧪 Testing

### Test the Application

1. **Model Loading**: Verify all models load correctly
2. **Text Input**: Test with various input lengths
3. **Generation**: Try different parameter combinations
4. **Comparison**: Compare multiple models
5. **Metrics**: Verify metric calculations

### Sample Test Cases

```python
# Short text (< 512 tokens)
"The quick brown fox jumps over the lazy dog. This is a test."

# Medium text (512-1024 tokens)
# Use sample texts from the app

# Long text (> 1024 tokens)
# Upload a book chapter
```

## 🐛 Troubleshooting

### Common Issues

**1. Model not found**
```
Error: Model path does not exist
```
**Solution**: Check `results/` directory and update paths in `config.py`

**2. Out of memory**
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Reduce batch size
- Use CPU instead
- Close other applications

**3. Slow generation**
```
Taking too long...
```
**Solution**:
- Reduce `num_beams`
- Decrease `max_length`
- Use GPU/MPS if available

**4. Import errors**
```
ModuleNotFoundError: No module named 'X'
```
**Solution**: 
```bash
pip install -r streamlit_app/requirements.txt
```
