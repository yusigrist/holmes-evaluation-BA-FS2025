import torch
from sentence_transformers.models import Transformer
from transformers import AutoModel, T5Config, BitsAndBytesConfig


class FourBitTransformer(Transformer):

    def _load_model(self, model_name_or_path, config, cache_dir, backend, is_peft_model, **model_args):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,  # Enable for additional memory savings
                bnb_4bit_compute_dtype=torch.bfloat16  # Match dtype to your training/inference setup
            )
            self.auto_model = AutoModel.from_pretrained(
                model_name_or_path, 
                config=config, 
                quantization_config=quantization_config, 
                device_map="auto", 
                trust_remote_code=True
            )

    def _load_t5_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,  # Enable for additional memory savings
                bnb_4bit_compute_dtype=torch.bfloat16  # Match dtype to your training/inference setup
            )
        self.auto_model = T5EncoderModel.from_pretrained(
            model_name_or_path, 
            config=config,  
            quantization_config=quantization_config, 
            device_map="auto", 
            trust_remote_code=True
        )