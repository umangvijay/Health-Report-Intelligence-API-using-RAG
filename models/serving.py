"""
Unified Model Serving Layer
Supports local (quantized/LoRA), HuggingFace, and Gemini APIs
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class ModelServer:
    """Unified interface for multiple LLM backends"""

    def __init__(self, model_type: str = "gemini", use_quantization: bool = True, use_lora: bool = False):
        """
        Args:
            model_type: 'gemini', 'huggingface', 'local', 'meditron'
            use_quantization: Enable 4-bit quantization for local models
            use_lora: Enable LoRA for efficient fine-tuning
        """
        self.model_type = model_type
        self.use_quantization = use_quantization
        self.use_lora = use_lora
        self._init_model()

    def _init_model(self):
        """Initialize model based on type"""
        if self.model_type == "gemini":
            import google.generativeai as genai
            self.client = genai.GenerativeModel("gemini-pro")
        elif self.model_type == "huggingface":
            self.hf_token = os.getenv("HF_TOKEN", "")
            self.api_url = "https://api-inference.huggingface.co/models/"
        elif self.model_type == "local":
            self._init_local_model()
        elif self.model_type == "meditron":
            self.hf_token = os.getenv("HF_TOKEN", "")
            self.api_url = "https://api-inference.huggingface.co/models/epfl-llm/meditron-70b"

    def _init_local_model(self):
        """Initialize local model with optional quantization and LoRA"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import get_peft_model, LoraConfig, TaskType
            
            model_name = "mistralai/Mistral-7B-Instruct-v0.1"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with 4-bit quantization if enabled
            if self.use_quantization:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Apply LoRA if enabled
            if self.use_lora:
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                self.model = get_peft_model(self.model, lora_config)
        except Exception as e:
            print(f"Failed to initialize local model: {e}")
            self.model = None

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text from prompt"""
        if self.model_type == "gemini":
            return self._gemini_generate(prompt, max_tokens, temperature)
        elif self.model_type == "huggingface":
            return self._hf_generate(prompt, max_tokens, temperature)
        elif self.model_type == "local":
            return self._local_generate(prompt, max_tokens, temperature)
        elif self.model_type == "meditron":
            return self._meditron_generate(prompt, max_tokens, temperature)
        return ""

    def _gemini_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate via Gemini API"""
        try:
            response = self.client.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": temperature}
            )
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    def _hf_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate via HuggingFace Inference API"""
        import requests
        try:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}
            }
            response = requests.post(self.api_url + "mistralai/Mistral-7B-Instruct-v0.1", 
                                   headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()[0]["generated_text"]
            return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _meditron_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate via Meditron-70B API"""
        import requests
        try:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}
            }
            response = requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()[0]["generated_text"]
            return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _local_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate via local model"""
        if self.model is None:
            return "Local model not initialized"
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error: {str(e)}"


class EmbeddingServer:
    """Unified embedding interface"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            self.model = None

    def embed(self, text: str) -> list:
        """Embed text to vector"""
        if self.model is None:
            return []
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    def embed_batch(self, texts: list) -> list:
        """Embed multiple texts"""
        if self.model is None:
            return []
        try:
            return self.model.encode(texts).tolist()
        except Exception as e:
            print(f"Batch embedding error: {e}")
            return []
