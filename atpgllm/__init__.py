
from .utils import *
from .llm import *
from .graph import *
import json

models_causal = {
  'tiny-mistral': "openaccess-ai-collective/tiny-mistral",
  'llama-2': "meta-llama/Llama-2-7b-chat-hf",
  'nousre-llama-2': "NousResearch/Llama-2-7b-chat-hf",
  'llama-2-atpg': "chrivasileiou/LlamaModelForCausalLM-ATPG",
  'llama-2-atpg-lora': "chrivasileiou/LlamaModelForCausalLM-ATPG-LoRA",
}

models_seq2seq = {
  'tiny-t5': "google/t5-efficient-tiny",
  't5': "google-t5/t5-base",
  'new-t5': "google/t5-v1_1-base",
}


with open("/proj/trela/christos/.cache/huggingface/accelerate/deepspeed_config.json", 'r') as f:
  deepspeed_config = json.load(f)
