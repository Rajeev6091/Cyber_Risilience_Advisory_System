🧠 Integrated Cybersecurity Assessment System

A powerful system combining RAG (Retrieval-Augmented Generation) and BERT-based classification for comprehensive cybersecurity assessments.

📘 Project Overview

This system integrates multiple approaches to provide more accurate and reliable cybersecurity assessments:

1. **RAG System** - Uses large language models (OpenAI GPT, Google Gemini) with vector retrieval to analyze cybersecurity contexts against a database of examples
2. **BERT Classification** - Uses a fine-tuned BERT-mini model specifically trained to classify cybersecurity setups as Bad, Good, or Excellent
3. **Hybrid Approach** - Combines both methods.

## Files Structure

- `app.py` - The RAG-based system for retrieving and classifying cybersecurity profiles
- `train_finetune.py` - The script used to fine-tune the BERT model for classification
- `test.py` - Evaluation script for the BERT model on the test dataset
- `single_test.py` - Script for testing a single input with the BERT model
- `integrated_system.py` - **NEW** Integration system combining all components(hybrid) with an enhanced Gradio interface

🔧 Setup Instructions

### 1. Environment Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file with the following variables:

```
PINECONE_API_KEY="your_pinecone_api_key"
GEMINI_API_KEY="your_gemini_api_key"
OPENAI_API_KEY="your_openai_api_key"
LANGCHAIN_API_KEY="your_langchain_api_key"
HF_TOKEN="your_huggingface_token"
GOOGLE_API_KEY="your_google_api_key"
```

### 3. PDF Directory Setup

Ensure your PDF documents are properly organized with appropriate labels:
- Excellent profiles in folders containing 'excellent_profile' in the name
- Good profiles in folders containing 'good_profile' in the name
- Bad profiles in folders containing 'bad_profile' in the name

Update the Config paths in app.py if necessary.

## Running the System

Launch the integrated system:
Note: Ensure you have the necessary API keys set up in your `.env` file and the PDF directory structured correctly. llm_finetune_project directory has train_finetune.py script to fine-tune the BERT model, test.py for evaluating the model, and single_test.py for testing a single input.
```bash
python integrated_system.py
```

This will start the Gradio web interface with all functionality.

🚀 Features

### 1. Single Query Analysis

Analyze individual cybersecurity scenarios using:

- BERT-only classification (fast, based on direct pattern matching)
- RAG-only classification (comprehensive, based on retrieved examples)
- Hybrid classification (combines both approaches for optimal results)

### 2. Batch Processing

Process multiple queries from a CSV file and get consolidated results.

### 3. Analytics

View system performance metrics including:
- Classification distribution
- Agreement rate between models
- Processing times
- Model confidence levels

## Customization

### Adding New Models

To add new LLM models to the RAG system, update the `get_model_config()` method in the Config class in app.py.

### Fine-tuning BERT for New Data

If you have new training data:

1. Format your data as CSV with 'input' and 'label' columns
2. Update the paths in train_finetune.py
3. Run `python train_finetune.py` to retrain the model
4. Update the BERT_MODEL_PATH in your .env file

## Technical Details

### Hybrid Classification Approach

The system uses a weighted decision mechanism:
- Model (BERT) provides a classification and confidence score.
- RAG system retrieves relevant examples and provides a classification with Description for improvements.


### Metrics Tracking

The system tracks comprehensive metrics for each query:
- Classification from each model
- Confidence levels
- Agreement between models
- Processing times
- Final classification result

## Troubleshooting

Common issues:

1. **API Key errors**: Ensure your .env file has valid API keys
2. **Model loading errors**: Check that the BERT model path is correct
3. **Vector store errors**: Delete the vector store directory and let the system rebuild it
4. **Memory issues**: Reduce the context_chunks parameter for large documents

## Future Improvements

Potential enhancements:
- Support for additional LLM providers
- Enhanced visualization of analytics
- Active learning to improve model accuracy over time
- Confidence calibration for more reliable assessments