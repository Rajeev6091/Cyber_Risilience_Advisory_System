import os
import time
import numpy as np
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import gradio as gr
import pandas as pd
from datetime import datetime
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_openai.chat_models import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from enum import Enum
import tiktoken  # For token counting
from collections import defaultdict

load_dotenv()

# Project root = directory containing this file. All paths are derived from it
# so the app runs regardless of where the repo is cloned.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

class MetricsTracker:
    def __init__(self, metrics_file=os.path.join(OUTPUT_DIR, "rag_metrics.csv")):
        self.metrics_file = metrics_file
        # Make sure the output directory exists before writing
        os.makedirs(os.path.dirname(os.path.abspath(metrics_file)), exist_ok=True)
        # Create metrics file with headers if it doesn't exist
        if not os.path.exists(metrics_file):
            headers = [
                "query_id", "query_text", "temperature", "context_chunks", 
                "retrieval_time_ms", "total_time_ms", "num_retrieved_docs", 
                "sources_list", "llm_response", "token_count", 
                "query_doc_euclidean_dist", "precision_at_k", "recall_at_k", 
                "mrr", "groundedness_score", "source_profile_counts"
            ]
            pd.DataFrame(columns=headers).to_csv(metrics_file, index=False)
    
    def log_metrics(self, metrics_data: Dict[str, Any]):
        """Log metrics to CSV file with validation"""
        try:
            # Define expected types for each metric
            metric_types = {
                "query_id": str,
                "query_text": str,
                "temperature": (int, float),
                "context_chunks": int,
                "retrieval_time_ms": (int, float),
                "total_time_ms": (int, float),
                "num_retrieved_docs": int,
                "sources_list": str,
                "llm_response": str,
                "token_count": int,
                "query_doc_euclidean_dist": (int, float),
                "precision_at_k": (int, float, str),
                "recall_at_k": (int, float, str),
                "mrr": (int, float, str),
                "groundedness_score": (int, float, str),
                "source_profile_counts": str
            }
            
            # Validate and clean each metric
            cleaned_metrics = {}
            for metric, expected_type in metric_types.items():
                value = metrics_data.get(metric)
                
                # Handle missing values
                if value is None:
                    if expected_type == str:
                        cleaned_metrics[metric] = ""
                    elif expected_type in (int, float):
                        cleaned_metrics[metric] = 0
                    continue
                
                # Validate type
                if isinstance(value, expected_type) or (
                    isinstance(expected_type, tuple) and 
                    any(isinstance(value, t) for t in expected_type)
                ):
                    cleaned_metrics[metric] = value
                else:
                    # Try to convert to correct type
                    try:
                        if expected_type == int or (isinstance(expected_type, tuple) and int in expected_type):
                            cleaned_metrics[metric] = int(float(value))
                        elif expected_type == float or (isinstance(expected_type, tuple) and float in expected_type):
                            cleaned_metrics[metric] = float(value)
                        else:
                            cleaned_metrics[metric] = str(value)
                    except (ValueError, TypeError):
                        cleaned_metrics[metric] = "" if expected_type == str else 0
            
            # Log the cleaned metrics
            df = pd.DataFrame([cleaned_metrics])
            if os.path.exists(self.metrics_file):
                df.to_csv(self.metrics_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.metrics_file, index=False)
            
            print(f"Metrics logged successfully to {self.metrics_file}")
            return True
            
        except Exception as e:
            print(f"Error logging metrics: {str(e)}")
            return False

class TemperatureLevel(Enum):
    LOW = ("Low", 0.2)
    MEDIUM = ("Medium", 0.5)
    HIGH = ("High", 0.8)

    @classmethod
    def get_temp_value(cls, level_name: str) -> float:
        for temp_level in cls:
            if temp_level.value[0] == level_name:
                return temp_level.value[1]
        return cls.MEDIUM.value[1]  # Default to medium if not found

class Config:
    PDF_DIRECTORY = os.path.join(BASE_DIR, "pdfs")
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vectorstore")
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.csv")

    # @staticmethod
    # def get_model_config():
    #     return {
    #         "gpt-3.5-turbo": {"name": "GPT-3.5", "embedding": "openai"},
    #         "gpt-4": {"name": "GPT-4", "embedding": "openai"},
    #         "gemini-pro": {"name": "Gemini", "embedding": "gemini"},
    #     }

    @staticmethod
    def get_model_config():
        return {
            "gpt-4": {"name": "GPT-4", "embedding": "openai"},
            "deepseek": {"name": "DeepSeek", "embedding": "gemini"},
            "mistral": {"name": "Mistral", "embedding": "gemini"},
            "gemini-pro": {"name": "Gemini", "embedding": "gemini"},
            "claude": {"name": "Claude", "embedding": "gemini"},
        }
    
    @staticmethod
    def get_temperature_levels():
        return [level.value[0] for level in TemperatureLevel]

class DocumentProcessor:
    def __init__(self, pdf_directory: str):
        self.pdf_directory = pdf_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

    def load_and_split_documents(self) -> List[Document]:
        try:
            loader = PyPDFDirectoryLoader(self.pdf_directory)
            documents = loader.load()
            print(f"Documents loaded: {len(documents)}")
            if not documents:
                print("Warning: No documents found in the specified directory.")
                return []
            chunks = self.text_splitter.split_documents(documents)
            print(f"Processed {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            return []

class EmbeddingsManager:
    def __init__(self):
        self.embeddings = {}
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        try:
            # Initialize OpenAI embeddings
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if openai_api_key:
                self.embeddings["openai"] = OpenAIEmbeddings(openai_api_key=openai_api_key)
                print("OpenAI embeddings initialized successfully.")
            else:
                print("Warning: OPENAI_API_KEY is not set in your .env file.")
            
            # -----------------------
            # OpenAI GPT-4 Model
            # -----------------------
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.models["gpt-4o"] = ChatOpenAI(
                    model="gpt-4o",
                    openai_api_key=openai_api_key,
                    temperature=TemperatureLevel.MEDIUM.value[1]
                )
                print("GPT-4o model initialized")
            else:
                print("Warning: OPENAI_API_KEY not found")

            # Initialize Gemini embeddings
            # google_api_key = os.getenv('GOOGLE_API_KEY')
            # if google_api_key:
            #     self.embeddings["gemini"] = GoogleGenerativeAIEmbeddings(
            #         google_api_key=google_api_key,
            #         # model="models/embedding-001"
            #         model="models/text-embedding-004"
            #     )
            #     print("Gemini embeddings initialized successfully.")
            # else:
            #     print("Warning: GOOGLE_API_KEY is not set in your .env file.")
            google_api_key = os.getenv('GOOGLE_API_KEY')
            if google_api_key:
                self.embeddings["gemini"] = GoogleGenerativeAIEmbeddings(
                    google_api_key=google_api_key
                    )
                print("Gemini embeddings initialized successfully.")
            else:
                print("Warning: GOOGLE_API_KEY is not set in your .env file.")
                
        except Exception as e:
            print(f"An error occurred during embeddings initialization: {str(e)}")
    
    def get_embedding(self, embedding_type: str):
        embedding = self.embeddings.get(embedding_type.lower())
        if not embedding:
            print(f"Embedding {embedding_type} is not available.")
            # Return fallback embedding if primary not available
            for emb_type, emb in self.embeddings.items():
                if emb:
                    print(f"Using {emb_type} embeddings as fallback.")
                    return emb
            raise ValueError("No embeddings available. Please check your API keys.")
        return embedding

    def embed_text(self, text: str, embedding_type: str) -> List[float]:
        """Generate embeddings for a text string"""
        embedding_model = self.get_embedding(embedding_type)
        return embedding_model.embed_query(text)

class VectorStoreManager:
    def __init__(self, store_path: str, embeddings_manager: EmbeddingsManager):
        self.store_path = store_path
        self.embeddings_manager = embeddings_manager
        self.vector_stores: Dict[str, FAISS] = {}

    def get_vector_store_path(self, embedding_type: str) -> str:
        return f"{self.store_path}_{embedding_type}"

    def create_or_load_vector_store(self, documents: List[Document], embedding_type: str) -> Optional[FAISS]:
        try:
            # Check if we already have this vector store in memory
            if embedding_type in self.vector_stores:
                return self.vector_stores[embedding_type]
            
            store_path = self.get_vector_store_path(embedding_type)
            embeddings = self.embeddings_manager.get_embedding(embedding_type)
            
            if os.path.exists(store_path):
                try:
                    print(f"Loading existing {embedding_type} vector store...")
                    vector_store = FAISS.load_local(
                        store_path, 
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    print(f"Existing {embedding_type} vector store loaded successfully.")
                except Exception as e:
                    print(f"Error loading {embedding_type} vector store: {e}. Recreating vector store...")
                    vector_store = self._create_vector_store(documents, embeddings)
                    vector_store.save_local(store_path)
                    print(f"New {embedding_type} vector store created and saved successfully.")
            else:
                print(f"Creating new {embedding_type} vector store...")
                vector_store = self._create_vector_store(documents, embeddings)
                vector_store.save_local(store_path)
                print(f"New {embedding_type} vector store created and saved successfully.")
            
            self.vector_stores[embedding_type] = vector_store
            return vector_store
            
        except Exception as e:
            print(f"Critical error creating or loading vector store: {str(e)}")
            return None

    def _create_vector_store(self, documents: List[Document], embeddings) -> FAISS:
        if not documents:
            raise ValueError("No documents provided to create vector store")
        return FAISS.from_documents(documents, embeddings)

# class LLMManager:
#     def __init__(self):
#         self.models = {}
#         self._initialize_models()
#         # Initialize tokenizers for counting tokens
#         try:
#             self.tokenizers = {
#                 "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"),
#                 "gpt-4": tiktoken.encoding_for_model("gpt-4"),
#             }
#         except:
#             print("Warning: Could not initialize tokenizers. Token counting may be approximated.")
#             self.tokenizers = {}

#     def _initialize_models(self):
#         try:
#             # Initialize OpenAI models
#             openai_api_key = os.getenv('OPENAI_API_KEY')
#             if not openai_api_key:
#                 print("Warning: OPENAI_API_KEY is not set in your .env file.")
#             else:
#                 self.models["gpt-3.5-turbo"] = ChatOpenAI(
#                     model_name="gpt-3.5-turbo", 
#                     temperature=TemperatureLevel.MEDIUM.value[1],
#                     openai_api_key=openai_api_key
#                 )
#                 self.models["gpt-4"] = ChatOpenAI(
#                     model_name="gpt-4", 
#                     temperature=TemperatureLevel.MEDIUM.value[1],
#                     openai_api_key=openai_api_key
#                 )
#                 print("OpenAI models initialized successfully.")

#             # Initialize Gemini model
#             google_api_key = os.getenv('GOOGLE_API_KEY')
#             if not google_api_key:
#                 print("Warning: GOOGLE_API_KEY is not set in your .env file.")
#             else:
#                 self.models["gemini-pro"] = ChatGoogleGenerativeAI(
#                     model="gemini-1.5-pro",
#                     temperature=TemperatureLevel.MEDIUM.value[1],
#                     google_api_key=google_api_key,
#                     convert_system_message_to_human=True
#                 )
#                 print("Gemini model initialized successfully.")

#         except Exception as e:
#             print(f"An unexpected error occurred during model initialization: {e}")

#     def get_model(self, model_name: str):
#         model = self.models.get(model_name.lower())
#         if not model:
#             print(f"Model {model_name} is not available.")
#             # Return fallback model if primary not available
#             for model_name, model in self.models.items():
#                 if model:
#                     print(f"Using {model_name} as fallback model.")
#                     return model_name, model
#             raise ValueError("No models available. Please check your API keys.")
#         return model_name, model
    
#     def count_tokens(self, text: str, model_name: str) -> int:
#         """Count tokens in text for a specific model"""
#         if not text:
#             return 0
            
#         try:
#             if model_name.lower() in self.tokenizers:
#                 return len(self.tokenizers[model_name.lower()].encode(text))
#             else:
#                 # Approximate token count for models without tokenizers
#                 return len(text.split()) * 1.3  # Rough approximation
#         except Exception as e:
#             print(f"Error counting tokens: {str(e)}")
#             return 0

class LLMManager:
    def __init__(self):
        self.models = {}
        self._initialize_models()

        try:
            self.tokenizers = {
                "gpt-4": tiktoken.encoding_for_model("gpt-4"),
            }
        except:
            self.tokenizers = {}

    def _initialize_models(self):
        try:

            # Initialize OpenAI models
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                print("Warning: OPENAI_API_KEY is not set in your .env file.")
            else:
                self.models["gpt-4"] = ChatOpenAI(
                    model_name="gpt-4", 
                    temperature=TemperatureLevel.MEDIUM.value[1],
                    openai_api_key=openai_api_key
                )
                print("OpenAI models initialized successfully.")

            # -----------------------
            # DeepSeek Model
            # -----------------------
            deepseek_key = os.getenv("DEEPSEEK_API_KEY")

            if deepseek_key:
                self.models["deepseek"] = ChatOpenAI(
                    model="deepseek-chat",
                    openai_api_key=deepseek_key,
                    openai_api_base="https://api.deepseek.com/v1",
                    temperature=TemperatureLevel.MEDIUM.value[1]
                )
                print("DeepSeek model initialized")

            else:
                print("Warning: DEEPSEEK_API_KEY not found")

            # -----------------------
            # Mistral Model
            # -----------------------
            mistral_key = os.getenv("MISTRAL_API_KEY")

            if mistral_key:
                self.models["mistral"] = ChatMistralAI(
                    model="mistral-large-latest",
                    mistral_api_key=mistral_key,
                    temperature=TemperatureLevel.MEDIUM.value[1]
                )
                print("Mistral model initialized")

            else:
                print("Warning: MISTRAL_API_KEY not found")

            # -----------------------
            # Gemini Model
            # -----------------------
            google_api_key = os.getenv("GOOGLE_API_KEY")

            if google_api_key:
                self.models["gemini-pro"] = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    temperature=TemperatureLevel.MEDIUM.value[1],
                    google_api_key=google_api_key,
                    convert_system_message_to_human=True
                )
                print("Gemini model initialized")

            # -----------------------
            # Claude (Anthropic) Model
            # -----------------------
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")

            if anthropic_key:
                self.models["claude"] = ChatAnthropic(
                    model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                    temperature=TemperatureLevel.MEDIUM.value[1],
                    anthropic_api_key=anthropic_key
                )
                print("Claude model initialized")

            else:
                print("Warning: ANTHROPIC_API_KEY not found")

        except Exception as e:
            print(f"Error initializing models: {e}")

    def get_model(self, model_name: str):

        model = self.models.get(model_name.lower())

        if not model:
            for name, model in self.models.items():
                if model:
                    return name, model
            raise ValueError("No models available")

        return model_name, model

    def count_tokens(self, text: str, model_name: str):

        if not text:
            return 0

        try:
            return int(len(text.split()) * 1.3)
        except:
            return 0

class MetricsCalculator:
    """Utility class for calculating various metrics"""
    
    @staticmethod
    def calculate_euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return -1  # Invalid vectors
        
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    
    @staticmethod
    def calculate_avg_euclidean_distance(vec: List[float], vec_list: List[List[float]]) -> float:
        """Calculate average Euclidean distance between a vector and a list of vectors"""
        if not vec or not vec_list:
            return -1
        
        distances = [MetricsCalculator.calculate_euclidean_distance(vec, v) for v in vec_list]
        return sum(distances) / len(distances)

class RAGApplication:
    def __init__(self):
        self.doc_processor = DocumentProcessor(Config.PDF_DIRECTORY)
        self.embeddings_manager = EmbeddingsManager()
        self.vector_store_manager = VectorStoreManager(Config.VECTOR_STORE_PATH, self.embeddings_manager)
        self.llm_manager = LLMManager()
        self.documents = self.doc_processor.load_and_split_documents()
        self.metrics_tracker = MetricsTracker(Config.METRICS_FILE)
        self.metrics_calculator = MetricsCalculator()
        self.query_counter = 0
        print(f"System initialized with {len(self.documents)} document chunks")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        model_info = Config.get_model_config().get(model_name, {})
        if not model_info:
            # Provide default fallback
            return {"name": "Unknown", "embedding": "gemini"}
        return model_info

    def _count_source_profiles(self, source_documents: List[Document]) -> Dict[str, int]:
        """Count the number of documents from each profile type (Excellent, Good, Bad)"""
        profile_counts = defaultdict(int)
        
        for doc in source_documents:
            source_path = doc.metadata.get('source', '')
            if 'excellent_profile' in source_path.lower():
                profile_counts['Excellent'] += 1
            elif 'good_profile' in source_path.lower():
                profile_counts['Good'] += 1
            elif 'bad_profile' in source_path.lower():
                profile_counts['Bad'] += 1
            else:
                profile_counts['Unknown'] += 1
                
        return dict(profile_counts)

    def _determine_majority_profile(self, profile_counts: Dict[str, int]) -> str:
        """Determine the majority profile based on counts"""
        if not profile_counts:
            return "Unknown"
        
        # Remove unknown counts if other profiles exist
        if len(profile_counts) > 1 and 'Unknown' in profile_counts:
            del profile_counts['Unknown']
            
        if not profile_counts:
            return "Unknown"
            
        # Get the profile with maximum count
        majority_profile = max(profile_counts.items(), key=lambda x: x[1])[0]
        
        # If there's a tie, we'll return the highest quality profile
        max_count = profile_counts[majority_profile]
        profiles_with_max = [k for k, v in profile_counts.items() if v == max_count]
        
        if len(profiles_with_max) > 1:
            # Prefer higher quality profiles in case of tie
            if 'Excellent' in profiles_with_max:
                return 'Excellent'
            elif 'Good' in profiles_with_max:
                return 'Good'
            else:
                return 'Bad'
        
        return majority_profile

    def get_response(self, query: str, model_name: str, temp_level: str, noise_level: float, context_chunks: int) -> str:
        start_time = time.time()
        self.query_counter += 1
        query_id = f"Q{self.query_counter:04d}"
        
        metrics_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query_id": query_id,
            "query_text": query,
            "temperature": TemperatureLevel.get_temp_value(temp_level) + (noise_level * 0.1),
            "context_chunks": context_chunks,
            "retrieval_time_ms": 0,
            "total_time_ms": 0,
            "num_retrieved_docs": 0,
            "sources_list": "",
            "llm_response": "",
            "token_count": 0,
            "query_doc_euclidean_dist": -1,
            "precision_at_k": "NA",
            "recall_at_k": "NA",
            "mrr": "NA",
            "groundedness_score": "NA",
            "source_profile_counts": ""
        }
        
        try:
            # Get embedding type for selected model
            model_info = self.get_model_info(model_name)
            embedding_type = model_info.get("embedding", "gemini")  # Default to gemini if not specified
            
            # Try to get the LLM
            try:
                model_name, llm = self.llm_manager.get_model(model_name)
            except ValueError as e:
                return f"Error: {str(e)}"
            
            # Try to get the vector store
            try:
                vector_store = self.vector_store_manager.create_or_load_vector_store(self.documents, embedding_type)
                if vector_store is None:
                    return "Vector store not initialized. Please check logs for errors."
            except ValueError as e:
                return f"Error with vector store: {str(e)}"

            # Setting temperature based on level and apply noise
            base_temp = TemperatureLevel.get_temp_value(temp_level)
            final_temp = min(1.0, base_temp + (noise_level * 0.1))  # Noise affects temperature by up to 0.1
            
            if isinstance(llm, (ChatOpenAI, ChatGoogleGenerativeAI)):
                llm.temperature = final_temp

            # Generate embeddings for query
            query_embedding = self.embeddings_manager.embed_text(query, embedding_type)

            # Measure retrieval time separately
            retrieval_start = time.time()
            retriever = vector_store.as_retriever(search_kwargs={"k": context_chunks, "score_threshold": 0.60})
            docs = retriever.get_relevant_documents(query)
            retrieval_end = time.time()
            retrieval_time_ms = (retrieval_end - retrieval_start) * 1000
            
            # Store document embeddings for similarity calculations
            doc_embeddings = []
            for doc in docs:
                doc_text = doc.page_content
                doc_embedding = self.embeddings_manager.embed_text(doc_text, embedding_type)
                doc_embeddings.append(doc_embedding)
            
            # Calculate query-document Euclidean distance
            if query_embedding and doc_embeddings:
                query_doc_euclidean = self.metrics_calculator.calculate_avg_euclidean_distance(
                    query_embedding, doc_embeddings
                )
            else:
                query_doc_euclidean = -1
            
            # Count source profiles
            profile_counts = self._count_source_profiles(docs)
            majority_profile = self._determine_majority_profile(profile_counts)
            
            # Update metrics
            metrics_data["retrieval_time_ms"] = retrieval_time_ms
            metrics_data["num_retrieved_docs"] = len(docs)
            metrics_data["sources_list"] = "; ".join([doc.metadata.get('source', 'Unknown') for doc in docs])
            metrics_data["query_doc_euclidean_dist"] = query_doc_euclidean
            metrics_data["source_profile_counts"] = str(profile_counts)
            
            # Create the prompt with source profile information
            prompt_template = """
            As an **advanced cybersecurity analyst**, your task is to evaluate the provided cybersecurity context and question. You will assess the **cyber resilience of the described organization (without mentioning any organization's name)** and classify it as **Excellent, Good, or Bad** based on the given case labels.

### **Source Profile Distribution**
The retrieved documents contain the following profile distribution:
- Excellent: {excellent_count}
- Good: {good_count}
- Bad: {bad_count}
- Majority profile: {majority_profile}

**This distribution is the PRIMARY FACTOR in determining the rating**, as it represents the most relevant examples from our knowledge base. The majority profile should strongly influence your final assessment.

---
### **Evaluation Guidelines for Analysis as Good, Bad or Excellent**
1. **If majority is Excellent (>50%):** 
   - Default to Excellent rating
   - Only downgrade if context contains explicit evidence of poor security practices
   - Focus on how the organization's assets match Excellent examples

2. **If majority is Good (>50%):** 
   - Default to Good rating
   - Consider Excellent if context shows exceptional practices
   - Consider Bad if context shows significant weaknesses

3. **If majority is Bad (>50%):** 
   - Default to Bad rating
   - Only upgrade if context contains explicit evidence of strong security practices

---

---
### **Current Case Analysis**
**Question:** {question}

**Profile Distribution Indicates:** {majority_profile} should be the starting point for rating.

**Context:** {context}

---

### **If and only if the user provides a query related to tool description, respond with:**
- *What is the purpose of this tool? Provide a complete introduction explaining how the cybersecurity analysis is conducted.*

---
### **Evaluation Framework**
#### **1. Profile Analysis**
- **Asset Categorization:** Categorize as **Bad, Good, or Excellent** based on the source profile distribution and context.
- **Identify Key Technical and Non-Technical Aspects.**
- **Highlight key technical and strategic aspects.**

#### **2. Improvement Recommendations (if necessary)**
- **Technical Enhancements:** List critical security and infrastructure improvements.
- **Non-Technical Enhancements:** Non-technical improvements with reasoning.
- **Impact on Category Improvement:** Explain how suggested changes improve cyber resilience.

---
## **Cyber Resilience Evaluation Process**
Your task is to classify the following description:  
**{question}**

To assist in this analysis, here are labeled case examples (**Excellent, Good, Bad**) from cybersecurity assessments:  

**{context}**

---

### **Final Response Format**
- **Analysis:** [Only Bad/Good/Excellent on the basis of majority retrieved file only] 
- **Technical Improvements:** [List with justifications]
- **Non-Technical Improvements:** [List with reasoning]
- **Final Recommendation:** [Steps for category improvement]
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"],
                partial_variables={
                    "excellent_count": str(profile_counts.get('Excellent', 0)),
                    "good_count": str(profile_counts.get('Good', 0)),
                    "bad_count": str(profile_counts.get('Bad', 0)),
                    "majority_profile": majority_profile
                }
            )

            # Using the chain with the retrieved docs
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            print(f"Execution details: Query={query}, Model={model_name}, Embedding={embedding_type}, Temperature={final_temp}")
            
            # Use the newer invoke method
            response = chain.invoke({"query": query})
            
            # Store the LLM response
            llm_response = response['result']
            
            # Add profile analysis to the response
            profile_analysis = f"\n\nProfile Analysis: Majority of retrieved documents are from {majority_profile} profiles (Counts: {profile_counts})"
            metrics_data["llm_response"] = llm_response
            
            # Count tokens in response
            try:
                token_count = self.llm_manager.count_tokens(llm_response, model_name)
                metrics_data["token_count"] = token_count
            except Exception as e:
                print(f"Error counting tokens: {str(e)}")
                metrics_data["token_count"] = 0
            
            # Calculate total time
            end_time = time.time()
            total_time_ms = (end_time - start_time) * 1000
            metrics_data["total_time_ms"] = total_time_ms
            
            # Log metrics
            self.metrics_tracker.log_metrics(metrics_data)
            
            formatted_response = f"Answer: {llm_response}\n\nSources:"
            for i, doc in enumerate(response["source_documents"], 1):
                formatted_response += f"\n{i}. {doc.metadata.get('source', 'Unknown source')}"
                
            return formatted_response
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(f"Error stack trace: {trace}")
            
            # Store error in LLM response field
            metrics_data["llm_response"] = f"Error: {str(e)}"
            
            # Calculate total time even in case of error
            end_time = time.time()
            total_time_ms = (end_time - start_time) * 1000
            metrics_data["total_time_ms"] = total_time_ms
            
            # Log error metrics
            self.metrics_tracker.log_metrics(metrics_data)
            
            return f"Error generating response: {str(e)}"

def create_gradio_interface(rag_app: RAGApplication):
    model_config = Config.get_model_config()
    model_display_names = {k: v["name"] for k, v in model_config.items()}
    
    with gr.Blocks() as demo:
        gr.Markdown("# Cyber Security Framework RAG System")
        
        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Enter your question here...",
                    lines=3
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            choices=list(model_config.keys()),
                            label="Select Model",
                            # value="gpt-5.1"  # Default to Gemini  
                            value="deepseek"
                        )
                    with gr.Column(scale=1):
                        temperature_dropdown = gr.Dropdown(
                            choices=Config.get_temperature_levels(),
                            label="Temperature Level",
                            value=TemperatureLevel.MEDIUM.value[0]
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        noise_slider = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            value=0,
                            step=0.1,
                            label="Randomness"
                        )
                    with gr.Column(scale=1):
                        context_slider = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=5,
                            step=5,
                            label="Context Chunks"
                        )
                
                submit_button = gr.Button("Submit", variant="primary")
                
                # Status display
                status_box = gr.Textbox(
                    label="System Status", 
                    value="System ready. Using default model: Gemini Pro",
                    lines=2
                )
                
            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Response", lines=20)
        
        # Update status when model changes
        def update_status(model):
            model_info = rag_app.get_model_info(model)
            embedding_type = model_info.get("embedding", "unknown")
            model_display = model_info.get("name", model)
            return f"Selected model: {model_display} (using {embedding_type} embeddings)"
        
        model_dropdown.change(
            fn=update_status,
            inputs=[model_dropdown],
            outputs=[status_box]
        )

        submit_button.click(
            fn=rag_app.get_response,
            inputs=[query_input, model_dropdown, temperature_dropdown, noise_slider, context_slider],
            outputs=[output_text]
        )
    
    return demo

if __name__ == "__main__":
    rag_app = RAGApplication()
    demo = create_gradio_interface(rag_app)
    demo.launch(share=True)