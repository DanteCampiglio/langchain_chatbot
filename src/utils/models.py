"""
Utilidades para manejo de modelos de Hugging Face
"""
import logging
import torch
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline
)

logger = logging.getLogger(__name__)


class ModelManager:
    """Gestor de modelos de Hugging Face"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.pipeline = None
    
    def load_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Cargar modelo de Hugging Face"""
        try:
            logger.info(f"Cargando modelo: {model_name}")
            
            # Determinar tipo de modelo
            if "t5" in model_name.lower():
                return self._load_t5_model(model_name)
            elif "gpt" in model_name.lower():
                return self._load_gpt_model(model_name)
            elif "blender" in model_name.lower():
                return self._load_blender_model(model_name)
            else:
                return self._load_generic_model(model_name)
                
        except Exception as e:
            logger.error(f"Error cargando modelo {model_name}: {e}")
            return None
    
    def _load_t5_model(self, model_name: str) -> Dict[str, Any]:
        """Cargar modelo T5"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.DEVICE == "cuda" else torch.float32,
            device_map="auto" if self.config.DEVICE == "cuda" else None,
            cache_dir=str(self.config.CACHE_DIR)
        )
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=self.config.MAX_LENGTH,
            min_length=self.config.MIN_LENGTH,
            temperature=self.config.TEMPERATURE,
            do_sample=self.config.DO_SAMPLE,
            top_p=self.config.TOP_P,
            top_k=self.config.TOP_K,
            num_beams=self.config.NUM_BEAMS,
            device=0 if self.config.DEVICE == "cuda" else -1
        )
        
        return {
            "tokenizer": tokenizer,
            "model": model,
            "pipeline": pipe,
            "type": "seq2seq"
        }
    
    def _load_gpt_model(self, model_name: str) -> Dict[str, Any]:
        """Cargar modelo GPT"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.DEVICE == "cuda" else torch.float32,
            device_map="auto" if self.config.DEVICE == "cuda" else None,
            cache_dir=str(self.config.CACHE_DIR)
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=self.config.MAX_LENGTH,
            temperature=self.config.TEMPERATURE,
            do_sample=self.config.DO_SAMPLE,
            top_p=self.config.TOP_P,
            device=0 if self.config.DEVICE == "cuda" else -1
        )
        
        return {
            "tokenizer": tokenizer,
            "model": model,
            "pipeline": pipe,
            "type": "causal"
        }
    
    def _load_blender_model(self, model_name: str) -> Dict[str, Any]:
        """Cargar modelo Blender"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.DEVICE == "cuda" else torch.float32,
            cache_dir=str(self.config.CACHE_DIR)
        )
        
        pipe = pipeline(
            "conversational",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.config.DEVICE == "cuda" else -1
        )
        
        return {
            "tokenizer": tokenizer,
            "model": model,
            "pipeline": pipe,
            "type": "conversational"
        }
    
    def _load_generic_model(self, model_name: str) -> Dict[str, Any]:
        """Cargar modelo genérico"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=str(self.config.CACHE_DIR)
        )
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.config.DEVICE == "cuda" else -1
        )
        
        return {
            "tokenizer": tokenizer,
            "model": model,
            "pipeline": pipe,
            "type": "generic"
        }
    
    def get_model_size(self, model) -> str:
        """Obtener tamaño del modelo"""
        try:
            param_count = sum(p.numel() for p in model.parameters())
            if param_count > 1e9:
                return f"{param_count/1e9:.1f}B parámetros"
            elif param_count > 1e6:
                return f"{param_count/1e6:.1f}M parámetros"
            else:
                return f"{param_count/1e3:.1f}K parámetros"
        except:
            return "Tamaño desconocido"