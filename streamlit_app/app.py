"""
Automated Book Summarization using Generative AI and Transfer Learning
Streamlit Application - Week 7 Demonstration Tool

This application showcases fine-tuned transformer models (BART, Longformer)
that generate coherent, contextually accurate summaries of long-form books
using transfer learning and generative AI techniques.
"""

import streamlit as st
import torch
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
import json

# Import local modules
from config import (
    MODELS, 
    DEFAULT_GEN_PARAMS, 
    GEN_PARAMS, 
    SAMPLE_TEXTS,
    APP_TITLE,
    APP_ICON,
    SIDEBAR_TITLE,
    DEVICE
)
from utils import (
    load_model_and_tokenizer,
    generate_summary,
    calculate_all_metrics,
    format_metrics_display,
    extract_text_from_file,
    get_text_statistics,
    preprocess_text
)

# Page configuration
st.set_page_config(
    page_title="Book Summarization App",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'loaded_models' not in st.session_state:
        st.session_state.loaded_models = {}


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.title(SIDEBAR_TITLE)
        
        # Model selection
        st.subheader("🤖 Model Selection")
        selected_model = st.selectbox(
            "Choose a model",
            list(MODELS.keys()),
            help="Select the summarization model to use"
        )
        
        # Display model info
        model_info = MODELS[selected_model]
        with st.expander("ℹ️ Model Information"):
            st.write(f"**Description:** {model_info['description']}")
            st.write(f"**Base Model:** {model_info['base_model']}")
            st.write(f"**Type:** {model_info['type'].upper()}")
            
            if 'metrics' in model_info:
                st.write("**Performance:**")
                for metric, value in model_info['metrics'].items():
                    st.write(f"- {metric}: {value:.4f}")
        
        st.divider()
        
        # Generation parameters
        st.subheader("⚙️ Generation Parameters")
        
        gen_params = {}
        
        gen_params['max_length'] = st.slider(
            "Max Length",
            min_value=GEN_PARAMS['max_length']['min'],
            max_value=GEN_PARAMS['max_length']['max'],
            value=GEN_PARAMS['max_length']['default'],
            step=GEN_PARAMS['max_length']['step'],
            help=GEN_PARAMS['max_length']['description']
        )
        
        gen_params['min_length'] = st.slider(
            "Min Length",
            min_value=GEN_PARAMS['min_length']['min'],
            max_value=GEN_PARAMS['min_length']['max'],
            value=GEN_PARAMS['min_length']['default'],
            step=GEN_PARAMS['min_length']['step'],
            help=GEN_PARAMS['min_length']['description']
        )
        
        gen_params['num_beams'] = st.slider(
            "Number of Beams",
            min_value=GEN_PARAMS['num_beams']['min'],
            max_value=GEN_PARAMS['num_beams']['max'],
            value=GEN_PARAMS['num_beams']['default'],
            step=GEN_PARAMS['num_beams']['step'],
            help=GEN_PARAMS['num_beams']['description']
        )
        
        gen_params['length_penalty'] = st.slider(
            "Length Penalty",
            min_value=GEN_PARAMS['length_penalty']['min'],
            max_value=GEN_PARAMS['length_penalty']['max'],
            value=GEN_PARAMS['length_penalty']['default'],
            step=GEN_PARAMS['length_penalty']['step'],
            help=GEN_PARAMS['length_penalty']['description']
        )
        
        gen_params['no_repeat_ngram_size'] = st.slider(
            "No Repeat N-gram Size",
            min_value=GEN_PARAMS['no_repeat_ngram_size']['min'],
            max_value=GEN_PARAMS['no_repeat_ngram_size']['max'],
            value=GEN_PARAMS['no_repeat_ngram_size']['default'],
            step=GEN_PARAMS['no_repeat_ngram_size']['step'],
            help=GEN_PARAMS['no_repeat_ngram_size']['description']
        )
        
        gen_params['early_stopping'] = st.checkbox(
            "Early Stopping",
            value=True,
            help="Stop generation when all beams are finished"
        )
        
        st.divider()
        
        # System information
        st.subheader("💻 System Info")
        st.write(f"**Device:** {DEVICE}")
        st.write(f"**PyTorch:** {torch.__version__}")
        
        if torch.cuda.is_available():
            st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            st.write("**Accelerator:** Apple Silicon (MPS)")
        
        return selected_model, gen_params


def render_summarize_tab():
    """Render the summarize tab"""
    st.header("📝 Generate Summary")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload", "Sample Text"],
        horizontal=True
    )
    
    input_text = ""
    
    if input_method == "Text Input":
        input_text = st.text_area(
            "Enter text to summarize:",
            height=300,
            placeholder="Paste your book text here...",
            value=st.session_state.current_text
        )
        
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload a text file:",
            type=['txt'],
            help="Currently supports .txt files only"
        )
        
        if uploaded_file is not None:
            input_text = extract_text_from_file(uploaded_file)
            if input_text:
                st.success(f"✅ File loaded: {uploaded_file.name}")
                with st.expander("Preview uploaded text"):
                    st.text(input_text[:500] + "..." if len(input_text) > 500 else input_text)
        
    elif input_method == "Sample Text":
        sample_choice = st.selectbox(
            "Select a sample:",
            list(SAMPLE_TEXTS.keys())
        )
        input_text = SAMPLE_TEXTS[sample_choice]
        st.info(f"📚 Using sample: {sample_choice}")
        with st.expander("View sample text"):
            st.write(input_text)
    
    # Text statistics
    if input_text:
        stats = get_text_statistics(input_text)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", f"{stats['characters']:,}")
        with col2:
            st.metric("Words", f"{stats['words']:,}")
        with col3:
            st.metric("Sentences", f"{stats['sentences']:,}")
        with col4:
            st.metric("Avg Word Length", f"{stats['avg_word_length']:.1f}")
    
    return input_text


def render_compare_tab():
    """Render the compare models tab"""
    st.header("🔄 Compare Models")
    
    st.write("Compare summaries from multiple models side-by-side")
    
    # Model selection for comparison
    selected_models = st.multiselect(
        "Select models to compare:",
        list(MODELS.keys()),
        default=list(MODELS.keys())[:2] if len(MODELS) >= 2 else list(MODELS.keys())
    )
    
    if len(selected_models) < 2:
        st.warning("⚠️ Please select at least 2 models to compare")
        return None, None
    
    # Input text
    input_text = st.text_area(
        "Enter text to summarize:",
        height=200,
        placeholder="Enter text for comparison..."
    )
    
    return selected_models, input_text


def render_metrics_tab():
    """Render the metrics visualization tab"""
    st.header("📊 Model Metrics")
    
    st.write("View and compare model performance metrics")
    
    # Load model metadata
    try:
        metadata_path = Path("results/model_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Display metrics
            st.subheader("Performance Metrics")
            
            metrics_data = []
            for model_name, model_config in MODELS.items():
                if 'metrics' in model_config:
                    row = {'Model': model_name}
                    row.update(model_config['metrics'])
                    metrics_data.append(row)
            
            if metrics_data:
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, use_container_width=True)
                
                # Visualizations
                st.subheader("Metric Comparisons")
                
                # ROUGE scores bar chart
                rouge_cols = [col for col in df_metrics.columns if 'rouge' in col.lower()]
                if rouge_cols:
                    fig = go.Figure()
                    for col in rouge_cols:
                        fig.add_trace(go.Bar(
                            name=col.upper(),
                            x=df_metrics['Model'],
                            y=df_metrics[col],
                            text=df_metrics[col].round(4),
                            textposition='auto',
                        ))
                    
                    fig.update_layout(
                        title="ROUGE Scores Comparison",
                        xaxis_title="Model",
                        yaxis_title="Score",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Radar chart for all metrics
                if len(metrics_data) > 0:
                    fig = go.Figure()
                    
                    for model_data in metrics_data:
                        model_name = model_data['Model']
                        categories = [k for k in model_data.keys() if k != 'Model']
                        values = [model_data[k] for k in categories]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name=model_name
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        title="Model Performance Radar Chart",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No metrics data available for comparison")
        else:
            st.info("Model metadata file not found")
            
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")


def main():
    """Main application function"""
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown(f'<h1 class="main-header">{APP_ICON} {APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Automated book summarization using Generative AI and Transfer Learning</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="font-size: 1rem; margin-top: -1rem;">Generate coherent, contextually accurate summaries with fine-tuned transformer models</p>', unsafe_allow_html=True)
    
    # Sidebar
    selected_model, gen_params = render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📝 Summarize", "🔄 Compare Models", "📊 Metrics"])
    
    # Tab 1: Single model summarization
    with tab1:
        input_text = render_summarize_tab()
        
        # Generate summary button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            generate_btn = st.button(
                "🚀 Generate Summary",
                type="primary",
                use_container_width=True,
                disabled=not input_text
            )
        
        if generate_btn and input_text:
            with st.spinner(f"Loading {selected_model}..."):
                # Load model
                model_config = MODELS[selected_model]
                model, tokenizer = load_model_and_tokenizer(
                    selected_model, 
                    model_config, 
                    DEVICE
                )
                
                if model is None or tokenizer is None:
                    st.error("Failed to load model. Please check the model path.")
                    return
            
            with st.spinner("Generating summary..."):
                start_time = time.time()
                
                # Generate summary
                try:
                    summary = generate_summary(
                        input_text,
                        model,
                        tokenizer,
                        gen_params,
                        DEVICE
                    )
                    
                    generation_time = time.time() - start_time
                    
                    # Store in session state
                    st.session_state.summaries[selected_model] = summary
                    st.session_state.current_text = input_text
                    
                    # Display results
                    st.success(f"✅ Summary generated in {generation_time:.2f} seconds")
                    
                    # Summary output
                    st.subheader("📄 Generated Summary")
                    st.write(summary)
                    
                    # Summary statistics
                    summary_stats = get_text_statistics(summary)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Summary Words", summary_stats['words'])
                    with col2:
                        st.metric("Compression Ratio", 
                                f"{(summary_stats['words'] / get_text_statistics(input_text)['words']):.2%}")
                    with col3:
                        st.metric("Generation Time", f"{generation_time:.2f}s")
                    
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
    
    # Tab 2: Model comparison
    with tab2:
        selected_models, compare_text = render_compare_tab()
        
        if selected_models and compare_text:
            if st.button("🔄 Compare Models", type="primary", use_container_width=True):
                comparison_results = {}
                
                for model_name in selected_models:
                    with st.spinner(f"Processing with {model_name}..."):
                        model_config = MODELS[model_name]
                        model, tokenizer = load_model_and_tokenizer(
                            model_name,
                            model_config,
                            DEVICE
                        )
                        
                        if model and tokenizer:
                            start_time = time.time()
                            summary = generate_summary(
                                compare_text,
                                model,
                                tokenizer,
                                gen_params,
                                DEVICE
                            )
                            generation_time = time.time() - start_time
                            
                            comparison_results[model_name] = {
                                'summary': summary,
                                'time': generation_time,
                                'words': len(summary.split())
                            }
                
                # Display comparison
                if comparison_results:
                    st.success("✅ Comparison complete!")
                    
                    # Side-by-side summaries
                    cols = st.columns(len(selected_models))
                    for idx, (model_name, results) in enumerate(comparison_results.items()):
                        with cols[idx]:
                            st.subheader(model_name)
                            st.write(results['summary'])
                            st.caption(f"⏱️ {results['time']:.2f}s | 📝 {results['words']} words")
                    
                    # Comparison table
                    st.subheader("📊 Comparison Summary")
                    comparison_df = pd.DataFrame([
                        {
                            'Model': name,
                            'Words': data['words'],
                            'Time (s)': f"{data['time']:.2f}",
                            'Words/sec': f"{data['words']/data['time']:.1f}"
                        }
                        for name, data in comparison_results.items()
                    ])
                    st.dataframe(comparison_df, use_container_width=True)
    
    # Tab 3: Metrics visualization
    with tab3:
        render_metrics_tab()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>📚 Automated Book Summarization using Generative AI and Transfer Learning</p>
        <p>Week 7 Deliverable | Built with Streamlit, PyTorch, and Hugging Face Transformers</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
