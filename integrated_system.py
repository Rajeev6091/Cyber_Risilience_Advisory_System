#source venv/bin/activate

import os
import time
import pandas as pd
import torch
import gradio as gr
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Import components from your existing files
from app import (
    Config, RAGApplication, MetricsTracker, TemperatureLevel
)
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load environment variables
load_dotenv()

# Project root = directory containing this file. Paths are derived from it so
# the system runs regardless of where the repo is cloned.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

class IntegratedCyberSecuritySystem:
    """
    Integrated system that combines RAG-based and BERT-based classification
    for cybersecurity assessment.
    """
    def __init__(self):
        # Initialize the RAG application
        self.rag_app = RAGApplication()
        
        # Initialize BERT model
        self.bert_model_path = os.getenv('BERT_MODEL_PATH',
                                        os.path.join(BASE_DIR, "llm_finetune_project", "bert-mini-finetuned"))
        self.bert_tokenizer_name = "prajjwal1/bert-mini"

        print("MODEL PATH:", self.bert_model_path)
        print("PATH EXISTS:", os.path.exists(self.bert_model_path))
        print("FILES:", os.listdir(self.bert_model_path) if os.path.exists(self.bert_model_path) else "No folder")
        
        # Load BERT model and tokenizer
        try:
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(self.bert_model_path)
            # self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_tokenizer_name)
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_path)
            # self.bert_trainer = Trainer(model=self.bert_model, tokenizer=self.bert_tokenizer)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_model.to(self.device)
            self.bert_model.eval()

            self.id2label = {0: "bad", 1: "good", 2: "excellent"}
            print("BERT model loaded successfully")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            self.bert_model = None
            self.bert_tokenizer = None
            self.bert_trainer = None
            
        # Set up metrics tracking
        self.metrics_file = os.getenv('HYBRID_METRICS_FILE',
                                     os.path.join(OUTPUT_DIR, "hybrid_metrics.csv"))
        self.metrics_tracker = MetricsTracker(self.metrics_file)
        
        # Query counter for tracking
        self.query_counter = 0
        
        print("Integrated system initialized successfully")
        
    def classify_with_bert(self, text: str) -> Dict[str, Any]:
        """
        Classify text using the fine-tuned BERT model
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with classification results
        """
        if self.bert_model is None or self.bert_tokenizer is None:
            return {"class": "unknown", "confidence": 0, "error": "BERT model not loaded"}
        
        try:
            # Tokenize input text
            inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # Get device from trainer
            # device = self.bert_trainer.model.device
            device = self.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                # outputs = self.bert_trainer.model(**inputs)
                outputs = self.bert_model(**inputs)
                logits = outputs.logits
                predicted_class_id = logits.argmax().item()
                confidences = torch.softmax(logits, dim=1)[0].tolist()
            
            # Prepare result
            result = {
                "class": self.id2label[predicted_class_id],
                "confidence": confidences[predicted_class_id],
                "all_confidences": {
                    self.id2label[i]: conf for i, conf in enumerate(confidences)
                }
            }
            return result
        except Exception as e:
            print(f"BERT classification error: {e}")
            return {"class": "unknown", "confidence": 0, "error": str(e)}
    
    def bert_only_classification(self, query: str) -> str:
        """
        Perform BERT-only classification
        
        Args:
            query: Input text query
            
        Returns:
            BERT classification result as formatted string
        """
        # Track metrics
        start_time = time.time()
        self.query_counter += 1
        query_id = f"Q{self.query_counter:04d}"
        
        # Initialize metrics data
        metrics_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query_id": query_id,
            "query_text": query,
            "method": "bert",
            "bert_classification": "",
            "bert_confidence": 0,
            "rag_majority_profile": "",
            "agreement": False,
            "final_classification": "",
            "total_time_ms": 0
        }
        
        try:
            # Perform BERT classification
            bert_result = self.classify_with_bert(query)
            bert_class = bert_result.get("class", "unknown")
            bert_confidence = bert_result.get("confidence", 0)
            
            metrics_data["bert_classification"] = bert_class
            metrics_data["bert_confidence"] = bert_confidence
            metrics_data["final_classification"] = bert_class
            
            end_time = time.time()
            metrics_data["total_time_ms"] = (end_time - start_time) * 1000
            self.metrics_tracker.log_metrics(metrics_data)
            
            # Format response for BERT-only
            formatted_response = f"🤖 BERT CLASSIFICATION RESULT\n"
            formatted_response += f"=" * 35 + "\n\n"
            formatted_response += f"📊 **Primary Classification:** {bert_class.upper()}\n"
            formatted_response += f"🎯 **Confidence Score:** {bert_confidence:.3f} ({bert_confidence*100:.1f}%)\n\n"
            formatted_response += f"📈 **Detailed Confidence Distribution:**\n"
            formatted_response += f"-" * 35 + "\n"
            
            for cls, conf in bert_result.get("all_confidences", {}).items():
                bar_length = int(conf * 20)  # Scale to 20 characters
                bar = "█" * bar_length + "░" * (20 - bar_length)
                formatted_response += f"• {cls.capitalize():>10}: {conf:.3f} |{bar}| {conf*100:.1f}%\n"
            
            formatted_response += f"\n⚡ **Processing Time:** {metrics_data['total_time_ms']:.2f} ms\n"
            formatted_response += f"🔢 **Query ID:** {query_id}\n"
            
            # # Add interpretation
            # formatted_response += f"\n📝 **Interpretation:**\n"
            # formatted_response += f"-" * 20 + "\n"
            # if bert_confidence > 0.8:
            #     formatted_response += f"✅ High confidence prediction - Very reliable result\n"
            # elif bert_confidence > 0.6:
            #     formatted_response += f"⚠️ Moderate confidence - Generally reliable result\n"
            # else:
            #     formatted_response += f"\n"
            
            return formatted_response
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(f"Error in BERT classification: {trace}")
            
            # Log error metrics
            end_time = time.time()
            metrics_data["total_time_ms"] = (end_time - start_time) * 1000
            metrics_data["error"] = str(e)
            self.metrics_tracker.log_metrics(metrics_data)
            
            return f"❌ Error in BERT classification: {str(e)}"
    
    def hybrid_classification(self, query: str, model_name: str, temp_level: str, 
                             noise_level: float, context_chunks: int, method: str = "hybrid") -> str:
        """
        Perform classification using different methods: BERT, RAG, or Hybrid
        
        Args:
            query: Input text query
            model_name: Name of the LLM to use for RAG
            temp_level: Temperature level (Low, Medium, High)
            noise_level: Additional randomness level (0.0-0.5)
            context_chunks: Number of context chunks to retrieve
            method: Classification method ("bert", "rag", or "hybrid")
            
        Returns:
            Classification result as formatted string
        """
        # If BERT-only method is selected, use the simplified function
        if method.lower() == "bert":
            return self.bert_only_classification(query)
        
        # Track metrics
        start_time = time.time()
        self.query_counter += 1
        query_id = f"Q{self.query_counter:04d}"
        
        # Initialize metrics data
        metrics_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query_id": query_id,
            "query_text": query,
            "method": method,
            "bert_classification": "",
            "bert_confidence": 0,
            "rag_majority_profile": "",
            "agreement": False,
            "final_classification": "",
            "total_time_ms": 0
        }
        
        try:
            # Perform BERT classification for all methods
            bert_result = self.classify_with_bert(query)
            bert_class = bert_result.get("class", "unknown")
            bert_confidence = bert_result.get("confidence", 0)
            
            metrics_data["bert_classification"] = bert_class
            metrics_data["bert_confidence"] = bert_confidence
            
            # For RAG or Hybrid methods, perform RAG classification
            rag_response = self.rag_app.get_response(query, model_name, temp_level, noise_level, context_chunks)
            
            # Extract majority profile from RAG response
            # This is a simplified approach - you might need to improve it
            if "Profile Analysis: Majority of retrieved documents are from" in rag_response:
                profile_part = rag_response.split("Profile Analysis: Majority of retrieved documents are from")[1]
                majority_profile = profile_part.split("profiles")[0].strip()
            else:
                # Try to determine from the response text
                if "excellent" in rag_response.lower():
                    majority_profile = "excellent"
                elif "good" in rag_response.lower():
                    majority_profile = "good"
                else:
                    majority_profile = "bad"
            
            metrics_data["rag_majority_profile"] = majority_profile.lower()
            
            # If using RAG-only method, return early
            if method.lower() == "rag":
                end_time = time.time()
                metrics_data["total_time_ms"] = (end_time - start_time) * 1000
                self.metrics_tracker.log_metrics(metrics_data)
                return rag_response
            
            # For hybrid method, combine both approaches
            # Determine if BERT and RAG agree
            agreement = bert_class.lower() == majority_profile.lower()
            metrics_data["agreement"] = agreement
            
            # Determine final classification with a weighted approach
            if bert_confidence > 0.8:
                # High confidence BERT prediction takes precedence
                final_class = bert_class
                confidence_note = "High BERT confidence, using BERT classification"
            elif agreement:
                # If models agree, use their shared classification
                final_class = bert_class
                confidence_note = "Models agree on classification"
            else:
                # If disagreement, use weighted decision
                if bert_confidence < 0.6:
                    final_class = majority_profile.lower()
                    confidence_note = "Low BERT confidence, using RAG classification"
                else:
                    final_class = bert_class
                    confidence_note = "Moderate BERT confidence, using BERT classification"
            
            metrics_data["final_classification"] = final_class
            
            # Calculate time and log metrics
            end_time = time.time()
            metrics_data["total_time_ms"] = (end_time - start_time) * 1000
            self.metrics_tracker.log_metrics(metrics_data)
            
            # Create enhanced response with combined results
            formatted_response = f"HYBRID CLASSIFICATION RESULT\n"
            formatted_response += f"-----------------------------\n"
            formatted_response += f"BERT Classification: {bert_class.capitalize()} (Confidence: {bert_confidence:.2f})\n"
            formatted_response += f"RAG Classification: {majority_profile.capitalize()}\n"
            formatted_response += f"Models {'agree' if agreement else 'disagree'} on classification.\n"
            formatted_response += f"Final Classification: {final_class.capitalize()} ({confidence_note})\n\n"
            formatted_response += f"Detailed Analysis:\n{rag_response}"
            
            return formatted_response
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(f"Error in hybrid classification: {trace}")
            
            # Log error metrics
            end_time = time.time()
            metrics_data["total_time_ms"] = (end_time - start_time) * 1000
            metrics_data["error"] = str(e)
            self.metrics_tracker.log_metrics(metrics_data)
            
            return f"Error in classification: {str(e)}"
    
    def generate_analytics(self) -> str:
        """
        Generate analytics from the metrics file
        
        Returns:
            String with analytics summary
        """
        try:
            # Read metrics data
            if not os.path.exists(self.metrics_file):
                return "No metrics data available yet."
            
            df = pd.read_csv(self.metrics_file)
            
            # Calculate statistics
            total_queries = len(df)
            method_counts = df['method'].value_counts().to_dict()
            avg_time = df['total_time_ms'].mean()
            agreement_rate = df[df['agreement'] == True].shape[0] / total_queries if total_queries > 0 else 0
            
            # Format the analytics report
            analytics = f"SYSTEM ANALYTICS\n"
            analytics += f"---------------\n"
            analytics += f"Total queries processed: {total_queries}\n"
            analytics += f"Methods used: {method_counts}\n"
            analytics += f"Average processing time: {avg_time:.2f} ms\n"
            analytics += f"Model agreement rate: {agreement_rate:.2%}\n\n"
            
            # Add more detailed analytics if available
            if 'bert_classification' in df.columns and 'rag_majority_profile' in df.columns:
                bert_counts = df['bert_classification'].value_counts().to_dict()
                rag_counts = df['rag_majority_profile'].value_counts().to_dict()
                
                analytics += f"BERT classifications: {bert_counts}\n"
                analytics += f"RAG classifications: {rag_counts}\n"
            
            return analytics
        except Exception as e:
            return f"Error generating analytics: {str(e)}"
    
    def batch_process(self, csv_file_path: str, model_name: str, temp_level: str, 
                     noise_level: float, context_chunks: int, method: str) -> str:
        """
        Process a batch of queries from a CSV file
        
        Args:
            csv_file_path: Path to the CSV file with queries
            model_name: Name of the LLM to use
            temp_level: Temperature level (Low, Medium, High)
            noise_level: Additional randomness level (0.0-0.5)
            context_chunks: Number of context chunks to retrieve
            method: Classification method ("bert", "rag", or "hybrid")
            
        Returns:
            Summary of batch processing results
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            
            if 'query' not in df.columns:
                return "Error: CSV file must contain a 'query' column."
            
            # Initialize results list
            results = []
            
            # Process each query
            for idx, row in df.iterrows():
                query = row['query']
                print(f"Processing query {idx+1}/{len(df)}: {query[:50]}...")
                
                # Process the query
                response = self.hybrid_classification(
                    query, model_name, temp_level, noise_level, context_chunks, method
                )
                
                # Extract classification from response
                if "Classification:" in response:
                    classification = response.split("Classification:")[1].split("\n")[0].strip()
                else:
                    classification = "Unknown"
                
                # Add to results
                results.append({
                    'query': query,
                    'classification': classification,
                    'response': response
                })
            
            # Save results to CSV
            results_df = pd.DataFrame(results)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_file = os.path.join(
                OUTPUT_DIR, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            results_df.to_csv(output_file, index=False)
            
            # Return summary
            summary = f"Batch processing completed\n"
            summary += f"Total queries processed: {len(df)}\n"
            summary += f"Results saved to: {output_file}\n"
            
            # Add classification distribution
            classification_counts = results_df['classification'].value_counts().to_dict()
            summary += f"Classification distribution: {classification_counts}\n"
            
            return summary
            
        except Exception as e:
            return f"Error in batch processing: {str(e)}"


def create_gradio_interface(integrated_system: IntegratedCyberSecuritySystem):
    """
    Create a Gradio interface for the integrated system
    
    Args:
        integrated_system: Instance of IntegratedCyberSecuritySystem
        
    Returns:
        Gradio interface
    """
    # Get model configuration from RAG app
    model_config = Config.get_model_config()
    
    # Define classification methods
    classification_methods = ["hybrid", "bert", "rag"]
    
    def update_controls_visibility(method):
        """Update the visibility of controls based on selected method"""
        if method == "bert":
            # When BERT is selected, disable all RAG-related controls
            return (
                gr.update(visible=False),  # model_dropdown
                gr.update(visible=False),  # temperature_dropdown
                gr.update(visible=False),  # noise_slider
                gr.update(visible=False),  # context_slider
                gr.update(visible=True)    # bert_info (show BERT-only info)
            )
        else:
            # When RAG or Hybrid is selected, show all controls
            return (
                gr.update(visible=True),   # model_dropdown
                gr.update(visible=True),   # temperature_dropdown
                gr.update(visible=True),   # noise_slider
                gr.update(visible=True),   # context_slider
                gr.update(visible=False)   # bert_info (hide BERT-only info)
            )
    
    # ---- Theme & custom styling ----
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )

    custom_css = """
    .gradio-container { max-width: 1180px !important; margin: auto !important; }
    #hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
        border-radius: 18px;
        padding: 30px 34px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(15,32,39,0.35);
        color: #ffffff;
        flex-wrap: nowrap;
        align-items: stretch;
        border: none !important;
    }
    #hero-text { flex: 1 1 auto; }
    #hero-btn-col {
        align-self: flex-end;
        flex: 0 0 auto !important;
        min-width: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    #theme-btn {
        white-space: nowrap;
        color: #ffffff !important;
        background: rgba(255,255,255,0.14) !important;
        border: 1px solid rgba(255,255,255,0.30) !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        backdrop-filter: blur(6px);
        transition: background 0.2s ease;
    }
    #theme-btn:hover { background: rgba(255,255,255,0.26) !important; }
    #hero h1 { margin: 0; font-size: 2.05rem; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
    #hero p  { margin: 8px 0 0; opacity: 0.92; font-size: 1.05rem; }
    #hero .badges { margin-top: 16px; }
    #hero .badge {
        display: inline-block; padding: 5px 14px; margin: 4px 8px 0 0;
        background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.28);
        border-radius: 999px; font-size: 0.82rem; font-weight: 600; backdrop-filter: blur(6px);
    }
    .card {
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 3px 14px rgba(0,0,0,0.07);
        border: 1px solid var(--border-color-primary) !important;
    }
    .primary-btn { font-weight: 700 !important; border-radius: 12px !important; }
    .tabitem { padding-top: 12px !important; }
    footer { display: none !important; }
    #footer-note { text-align: center; opacity: 0.65; font-size: 0.85rem; margin-top: 18px; }
    """

    with gr.Blocks(theme=theme, css=custom_css, title="Cyber Resilience Assessment") as demo:
        with gr.Row(elem_id="hero"):
            gr.HTML(
                """
                <div id="hero-text">
                    <h1>🛡️ Integrated Cybersecurity Assessment System</h1>
                    <p>RAG + BERT hybrid intelligence that profiles your security posture as
                       <b>Bad</b>, <b>Good</b>, or <b>Excellent</b> — with actionable insights.</p>
                    <div class="badges">
                        <span class="badge">🤖 BERT</span>
                        <span class="badge">🧠 RAG</span>
                        <span class="badge">🔄 Hybrid</span>
                        <span class="badge">⚡ Batch Mode</span>
                        <span class="badge">📊 Analytics</span>
                    </div>
                </div>
                """
            )
            with gr.Column(scale=0, min_width=160, elem_id="hero-btn-col"):
                theme_toggle = gr.Button("🌗 Toggle Light / Dark", size="sm", elem_id="theme-btn")

        with gr.Tabs():
            # Tab 1: Single Query Analysis
            with gr.TabItem("🔍 Single Query"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes="card"):
                        gr.Markdown("#### ⚙️ Configure Analysis")
                        query_input = gr.Textbox(
                            label="🔍 Cybersecurity Query",
                            placeholder="Describe a cybersecurity setup or scenario to evaluate...",
                            lines=5
                        )
                        
                        method_dropdown = gr.Dropdown(
                            choices=classification_methods,
                            label="🎯 Classification Method",
                            value="hybrid"
                        )
                        
                        # BERT-only information (initially hidden)
                        with gr.Group(visible=False) as bert_info_group:
                            gr.Markdown("### 🤖 BERT-Only Mode")
                            gr.Markdown("**Fast neural network classification** - Only text input required!")
                        
                        # RAG/Hybrid controls (initially visible)
                        with gr.Group(visible=True) as rag_controls_group:
                            with gr.Row():
                                with gr.Column(scale=1):
                                    model_dropdown = gr.Dropdown(
                                        choices=list(model_config.keys()),
                                        label="🧠 LLM Model",
                                        value="gemini-pro"
                                    )
                                with gr.Column(scale=1):
                                    temperature_dropdown = gr.Dropdown(
                                        choices=Config.get_temperature_levels(),
                                        label="🌡️ Temperature Level",
                                        value=TemperatureLevel.MEDIUM.value[0]
                                    )
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    noise_slider = gr.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        value=0,
                                        step=0.1,
                                        label="🎲 Randomness"
                                    )
                                with gr.Column(scale=1):
                                    context_slider = gr.Slider(
                                        minimum=5,
                                        maximum=50,
                                        value=5,
                                        step=5,
                                        label="📄 Context Chunks"
                                    )
                        
                        analyze_button = gr.Button("🚀 Analyze", variant="primary", size="lg", elem_classes="primary-btn")

                    with gr.Column(scale=1, elem_classes="card"):
                        gr.Markdown("#### 📊 Result")
                        output_text = gr.Textbox(label="Analysis Result", lines=22)
            
            # Tab 2: Batch Processing
            with gr.TabItem("⚡ Batch Processing"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes="card"):
                        gr.Markdown("#### 📁 Upload & Configure")
                        batch_file = gr.File(
                            label="Upload CSV with queries (must have a 'query' column)",
                            file_types=[".csv"]
                        )
                        
                        batch_method_dropdown = gr.Dropdown(
                            choices=classification_methods,
                            label="🎯 Classification Method",
                            value="hybrid"
                        )
                        
                        # Batch RAG/Hybrid controls
                        with gr.Group() as batch_rag_controls_group:
                            with gr.Row():
                                with gr.Column(scale=1):
                                    batch_model_dropdown = gr.Dropdown(
                                        choices=list(model_config.keys()),
                                        label="🧠 LLM Model",
                                        value="gemini-pro"
                                    )
                                with gr.Column(scale=1):
                                    batch_temp_dropdown = gr.Dropdown(
                                        choices=Config.get_temperature_levels(),
                                        label="🌡️ Temperature Level",
                                        value=TemperatureLevel.MEDIUM.value[0]
                                    )
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    batch_noise_slider = gr.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        value=0,
                                        step=0.1,
                                        label="🎲 Randomness"
                                    )
                                with gr.Column(scale=1):
                                    batch_context_slider = gr.Slider(
                                        minimum=5,
                                        maximum=50,
                                        value=5,
                                        step=5,
                                        label="📄 Context Chunks"
                                    )
                        
                        batch_button = gr.Button("⚡ Process Batch", variant="primary", size="lg", elem_classes="primary-btn")

                    with gr.Column(scale=1, elem_classes="card"):
                        gr.Markdown("#### 📈 Batch Result")
                        batch_output = gr.Textbox(label="Batch Processing Result", lines=22)
            
            # Tab 3: Analytics
            with gr.TabItem("📊 Analytics"):
                with gr.Row():
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("#### 📊 System Performance")
                        gr.Markdown("Generate live metrics across all processed queries — "
                                    "classification distribution, model agreement, timing and confidence.")
                        analytics_button = gr.Button("📊 Generate Analytics", variant="primary", size="lg", elem_classes="primary-btn")
                        analytics_output = gr.Textbox(label="Analytics Result", lines=25)
                        
            # Tab 4: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## 🛡️ Integrated Cybersecurity Assessment System
                
                This system combines two powerful approaches for cybersecurity assessment:
                
                ### 🤖 BERT Classification
                - **Fast & Efficient**: Fine-tuned BERT model specifically trained for cybersecurity assessment
                - **Direct Classification**: Classifies setups as Bad, Good, or Excellent
                - **High Confidence Scoring**: Provides detailed confidence metrics
                - **Lightweight**: Requires only text input, no additional parameters
                
                ### 🧠 RAG (Retrieval-Augmented Generation)
                - **Context-Aware**: Uses large language models (GPT, Gemini) with vector retrieval
                - **Rich Analysis**: Provides detailed explanations and recommendations
                - **Customizable**: Multiple temperature and randomness settings
                - **Comprehensive**: Analyzes against database of cybersecurity examples
                
                ### 🔄 Hybrid Approach
                - **Best of Both**: Combines BERT speed with RAG depth
                - **Intelligent Weighting**: Uses confidence scores to determine final classification
                - **Agreement Analysis**: Shows when models agree or disagree
                - **Enhanced Reliability**: More accurate and trustworthy results
                
                ### ✨ Features:
                
                - **Single Query Analysis** with multiple classification methods
                - **Batch Processing** for multiple queries at once
                - **Real-time Analytics** to track system performance
                - **Multiple LLM Support** (GPT-4, Gemini, Claude, DeepSeek, Mistral)
                - **Adaptive Interface** that adjusts based on selected method

                ### 👨‍💻 Authors:
                Abhishek Gond, Rajeev Singh
                """)

        gr.HTML(
            '<div id="footer-note">🛡️ Cyber Resilience Assessment &nbsp;•&nbsp; '
            'Powered by RAG + BERT &nbsp;•&nbsp; Built by Abhishek Gond & Rajeev Singh</div>'
        )
        
        # Light / Dark theme toggle (client-side, no reload — keeps all inputs/state)
        theme_toggle.click(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { document.body.classList.toggle('dark'); }"
        )

        # Update controls visibility when method changes
        method_dropdown.change(
            fn=lambda method: (
                gr.update(visible=(method != "bert")),  # rag_controls_group
                gr.update(visible=(method == "bert"))   # bert_info_group
            ),
            inputs=[method_dropdown],
            outputs=[rag_controls_group, bert_info_group]
        )
        
        # Similar update for batch processing
        batch_method_dropdown.change(
            fn=lambda method: gr.update(visible=(method != "bert")),
            inputs=[batch_method_dropdown],
            outputs=[batch_rag_controls_group]
        )
        
        # Connect functions to buttons
        analyze_button.click(
            fn=integrated_system.hybrid_classification,
            inputs=[
                query_input, 
                model_dropdown, 
                temperature_dropdown, 
                noise_slider, 
                context_slider, 
                method_dropdown
            ],
            outputs=[output_text]
        )
        
        batch_button.click(
            fn=integrated_system.batch_process,
            inputs=[
                batch_file, 
                batch_model_dropdown, 
                batch_temp_dropdown, 
                batch_noise_slider, 
                batch_context_slider, 
                batch_method_dropdown
            ],
            outputs=[batch_output]
        )
        
        analytics_button.click(
            fn=integrated_system.generate_analytics,
            inputs=[],
            outputs=[analytics_output]
        )
        
    return demo

if __name__ == "__main__":
    print("Initializing Integrated Cybersecurity Assessment System...")
    integrated_system = IntegratedCyberSecuritySystem()
    
    print("Creating Gradio interface...")
    demo = create_gradio_interface(integrated_system)
    
    print("Launching interface...")
    demo.launch(share=True)