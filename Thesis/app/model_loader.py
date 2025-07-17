import torch
import gc
import logging
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from configuration import (
    BGE_MODEL_NAME, 
    RERANKER_MODEL_NAME, 
    QWEN_MODEL_NAME, 
    LLAMA_MODEL_NAME, 
    GEMMA_MODEL_NAME, 
    AVAILABLE_MODELS, 
    HUGGINGFACE_TOKEN
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    logger.info("Using CPU")

def unload_model(model):
    """Unload model from GPU memory"""
    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()

def load_llm_model(model_name):
    """Load specific LLM model"""
    logger.info(f"Loading LLM model: {model_name}")
    try:
        # Configure model loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": True
        }

        # Special handling for Gemma
        if "gemma" in model_name.lower():
            model_kwargs["torch_dtype"] = torch.float16
            
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Ensure padding token for Gemma
        if "gemma" in model_name.lower():
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        # Load model with appropriate configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        model = model.eval()
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading LLM model {model_name}: {str(e)}")
        raise

def load_models(selected_model_name=QWEN_MODEL_NAME):
    logger.info("Loading models...")
    try:
        # Load BGE model
        logger.info(f"Loading BGE tokenizer: {BGE_MODEL_NAME}")
        bge_tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME, token=HUGGINGFACE_TOKEN)
        
        logger.info(f"Loading BGE model: {BGE_MODEL_NAME}")
        bge_model = AutoModel.from_pretrained(
            BGE_MODEL_NAME,
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.float16,  # Changed from bfloat16 to float16
            device_map='auto'
        ).eval()
        
        # Load Reranker model
        logger.info(f"Loading Reranker tokenizer: {RERANKER_MODEL_NAME}")
        reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME, token=HUGGINGFACE_TOKEN)
        
        logger.info(f"Loading Reranker model: {RERANKER_MODEL_NAME}")
        reranker_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_NAME,
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.float16,  # Changed from bfloat16 to float16
            device_map='auto'
        ).eval()
        
        # Load LLM model
        llm_model, llm_tokenizer = load_llm_model(selected_model_name)
        
        logger.info("All models loaded successfully.")
        return {
            "bge": (bge_model, bge_tokenizer),
            "reranker": (reranker_model, reranker_tokenizer),
            "llm": (llm_model, llm_tokenizer)
        }
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

