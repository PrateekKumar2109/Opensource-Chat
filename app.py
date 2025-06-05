import streamlit as st
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="📚 RAG PDF Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .model-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metrics-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .bot-message {
        background: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Model configurations
MODEL_CONFIGS = {
    "Qwen2.5-7B-Instruct": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "embedding_model": "all-MiniLM-L6-v2",
        "description": "Qwen2.5 7B - Excellent for instruction following and reasoning",
        "color": "#667eea"
    },
    "Llama-3.2-3B-Instruct": {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "embedding_model": "all-MiniLM-L6-v2", 
        "description": "Meta Llama 3.2 3B - Fast and efficient for general tasks",
        "color": "#764ba2"
    }
}

class RAGSystem:
    def __init__(self, model_name: str, embedding_model: str):
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self.load_models()
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def load_models(self):
        """Load the language model and embedding model"""
        try:
            with st.spinner(f"Loading {self.model_name}..."):
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # Set pad_token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Create pipeline
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
            with st.spinner("Loading embedding model..."):
                # Load embedding model
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                
            st.success("✅ Models loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("💡 Tip: Make sure you have enough GPU memory or try running on CPU")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            if end > text_len:
                end = text_len
            
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < text_len:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                if boundary > start + chunk_size // 2:
                    chunk = text[start:start + boundary + 1]
                    end = start + boundary + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def create_vector_store(self, documents: List[str]):
        """Create FAISS vector store from documents"""
        try:
            with st.spinner("Creating vector embeddings..."):
                self.documents = documents
                self.embeddings = self.embedding_model.encode(documents)
                
                # Create FAISS index
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(self.embeddings)
                self.index.add(self.embeddings.astype('float32'))
                
            st.success(f"✅ Created vector store with {len(documents)} chunks")
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
    
    def retrieve_relevant_docs(self, query: str, k: int = 3) -> List[str]:
        """Retrieve most relevant documents for the query"""
        if self.index is None:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Return relevant documents
            relevant_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents) and scores[0][i] > 0.1:  # Threshold for relevance
                    relevant_docs.append(self.documents[idx])
            
            return relevant_docs
            
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using the language model"""
        try:
            # Create prompt
            if "qwen" in self.model_name.lower():
                prompt = f"""<|im_start|>system
You are a helpful AI assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so clearly.
<|im_end|>
<|im_start|>user
Context: {context}

Question: {query}
<|im_end|>
<|im_start|>assistant"""
            else:  # Llama format
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so clearly.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Context: {context}

Question: {query}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>"""
            
            # Generate response
            with st.spinner("Generating answer..."):
                response = self.pipe(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False
                )
            
            answer = response[0]['generated_text'].strip()
            
            # Clean up the answer
            if "<|im_end|>" in answer:
                answer = answer.split("<|im_end|>")[0]
            if "<|eot_id|>" in answer:
                answer = answer.split("<|eot_id|>")[0]
                
            return answer
            
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return "I apologize, but I encountered an error while generating the answer."

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

# Main header
st.markdown("""
<div class="main-header">
    <h1>📚 RAG PDF Assistant</h1>
    <p>Upload your PDF and ask questions powered by Qwen3 & Llama 3</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎯 Model Selection")
    
    selected_model = st.selectbox(
        "Choose your AI model:",
        options=list(MODEL_CONFIGS.keys()),
        index=0
    )
    
    # Display model info
    model_info = MODEL_CONFIGS[selected_model]
    st.markdown(f"""
    <div class="model-card">
        <h4>{selected_model}</h4>
        <p>{model_info['description']}</p>
        <small><strong>Embedding:</strong> {model_info['embedding_model']}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Model initialization
    if st.button("🚀 Initialize Model", key="init_model"):
        st.session_state.rag_system = RAGSystem(
            model_info['model_name'],
            model_info['embedding_model']
        )
    
    st.divider()
    
    # PDF Upload
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to ask questions about"
    )
    
    if uploaded_file and st.session_state.rag_system:
        if st.button("🔄 Process PDF", key="process_pdf"):
            # Extract text
            text = st.session_state.rag_system.extract_text_from_pdf(uploaded_file)
            
            if text:
                # Chunk text
                chunks = st.session_state.rag_system.chunk_text(text)
                
                # Create vector store
                st.session_state.rag_system.create_vector_store(chunks)
                st.session_state.pdf_processed = True
                
                # Show metrics
                st.markdown(f"""
                <div class="metrics-container">
                    <h4>📊 Processing Complete</h4>
                    <p><strong>Total chunks:</strong> {len(chunks)}</p>
                    <p><strong>Average chunk size:</strong> {sum(len(c) for c in chunks) // len(chunks)} chars</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Settings
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    num_chunks = st.slider("Retrieved Chunks", 1, 10, 3)
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat with your PDF")
    
    # Check if system is ready
    if not st.session_state.rag_system:
        st.warning("⚠️ Please initialize a model first from the sidebar")
    elif not st.session_state.pdf_processed:
        st.info("📋 Please upload and process a PDF to start asking questions")
    else:
        # Chat interface
        with st.container():
            # Display chat history
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🙋 You:</strong> {question}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant:</strong> {answer}
                </div>
                """, unsafe_allow_html=True)
        
        # Question input
        with st.form("question_form", clear_on_submit=True):
            question = st.text_area(
                "Ask a question about your PDF:",
                placeholder="What is the main topic of this document?",
                height=100
            )
            
            col_a, col_b = st.columns([1, 4])
            with col_a:
                submitted = st.form_submit_button("🚀 Ask Question")
            
            if submitted and question.strip():
                # Retrieve relevant documents
                relevant_docs = st.session_state.rag_system.retrieve_relevant_docs(question, num_chunks)
                
                if relevant_docs:
                    # Combine context
                    context = "\n\n".join(relevant_docs)
                    
                    # Generate answer
                    answer = st.session_state.rag_system.generate_answer(question, context)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    
                    st.rerun()
                else:
                    st.error("No relevant information found in the document for your question.")

with col2:
    st.header("📈 System Status")
    
    # Status indicators
    if st.session_state.rag_system:
        st.success("✅ Model Ready")
        st.info(f"🧠 Using: {selected_model}")
    else:
        st.error("❌ Model Not Loaded")
    
    if st.session_state.pdf_processed:
        st.success("✅ PDF Processed")
        if st.session_state.rag_system and st.session_state.rag_system.documents:
            st.info(f"📄 {len(st.session_state.rag_system.documents)} chunks ready")
    else:
        st.warning("⏳ No PDF Processed")
    
    # System info
    st.header("💻 System Info")
    device = "GPU" if torch.cuda.is_available() else "CPU"
    st.info(f"🖥️ Running on: {device}")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.info(f"🎮 GPU Memory: {gpu_memory:.1f} GB")
    
    # Tips
    st.header("💡 Tips")
    st.markdown("""
    - **Better Questions**: Be specific and detailed
    - **Context Matters**: Questions about document content work best
    - **Model Selection**: Qwen3 for reasoning, Llama 3 for speed
    - **Chunk Size**: Larger chunks for detailed context
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit • Powered by Qwen3 & Llama 3</p>
</div>
""", unsafe_allow_html=True)
