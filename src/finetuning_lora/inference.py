"""Inference module for the finetuned model."""
import os
import logging
import torch
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
)
from peft import PeftModel, PeftConfig

from .config.model import ModelConfig

logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """Container for generation results."""
    text: str
    full_text: str
    input_length: int
    generated_length: int
    finish_reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "full_text": self.full_text,
            "input_length": self.input_length,
            "generated_length": self.generated_length,
            "finish_reason": self.finish_reason,
        }

class TextGenerator:
    """Handles text generation with the finetuned model."""
    
    def __init__(self, model_path: str, config: Optional[ModelConfig] = None):
        """Initialize with model path and optional config."""
        self.model_path = model_path
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=self.config.use_fast_tokenizer,
        )
        
        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        model_config = PeftConfig.from_pretrained(self.model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_config.base_model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            self.model_path,
            device_map="auto",
        )
        self.model.eval()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **generation_kwargs
    ) -> GenerationResult:
        """Generate text from a prompt.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt to use
            **generation_kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with the generated text and metadata
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first")
        
        # Prepare messages for chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_token_type_ids=False,
        ).to(self.device)
        
        # Prepare generation config
        generation_config = self._get_generation_config(**generation_kwargs)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode the generated text
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[0, input_length:]
        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        # Get the full text (prompt + generated)
        full_text = self.tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )
        
        # Get finish reason
        finish_reason = self._get_finish_reason(outputs)
        
        return GenerationResult(
            text=generated_text.strip(),
            full_text=full_text,
            input_length=input_length,
            generated_length=len(generated_tokens),
            finish_reason=finish_reason,
        )
    
    def chat(self, system_prompt: Optional[str] = None):
        """Start an interactive chat session."""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        print("\n" + "=" * 50)
        print("💬 INTERACTIVE CHAT - LoRA Fine-tuned Model")
        print("Type 'quit' or 'exit' to stop")
        print("Type 'help' for commands")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'salir']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self._print_help()
                    continue
                elif user_input.lower() == 'clear':
                    print("🧹 Conversation context cleared")
                    continue
                elif not user_input:
                    continue
                
                # Generate response
                print("\n🤖 Assistant: ", end="", flush=True)
                
                result = self.generate(
                    prompt=user_input,
                    system_prompt=system_prompt,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=256,
                )
                
                print(result.text)
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    def _get_generation_config(self, **kwargs) -> GenerationConfig:
        """Get generation configuration."""
        # Use provided kwargs or fall back to config defaults
        config_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        return GenerationConfig(**config_kwargs)
    
    def _get_finish_reason(self, outputs) -> str:
        """Get the finish reason for the generated sequence."""
        if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
            return "length" if outputs.sequences_scores[0] < 0 else "stop"
        return "length"
    
    @staticmethod
    def _print_help():
        """Print help message."""
        print("""
📚 Available Commands:
- Type your message: Chat with the assistant
- system <message>: Change system prompt
- clear: Clear conversation context
- help: Show this help message
- quit/exit: Exit the chat
        """)
