from typing import Dict, List, Any
import torch
import json
import torch.nn.functional as F
import multiprocessing as mp
from atpgllm.utils import (
  length_less_than_model_max_len, 
  patterns_contains_special_tokens,
  get_user_prompt, 
)
from .tokenizer import tokenize_fn
from ..utils import AttrDict
import torch.distributed as dist

class PinMemoryData(AttrDict):
  def pin_memory(self):
    for key in self.keys():
      if isinstance(self[key], torch.Tensor):
        self[key] = self[key].pin_memory()
    return self

start_of_assistant_responses_mapping = {"meta-llama/Llama-2-7b-chat-hf": "[/INST]", 
                                        "chivasileiou/TestModel": "[/INST]",
                                        "chivasileiou/TestModel-2": "[/INST]",
                                        "chivasileiou/TestModel-3": "[/INST]",
                                        "meta-llama/Llama-3.1-8B-Instruct": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                                        "meta-llama/Llama-3.2-3B-Instruct": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"}

class MyCollate:
  def __init__(self, tokenizer, is_causal=True, lora=False, instruction_training=False):
    self.tokenizer = tokenizer
    self.is_causal = is_causal
    self.lora = lora
    self.tokenize_fn = lambda x, t: tokenize_fn(x, tokenizer=t, is_causal=is_causal)
    self.set_right_padding()
    self.start_of_assistant_response = "[/INST]" if self.tokenizer.name_or_path not in start_of_assistant_responses_mapping \
                                                 else start_of_assistant_responses_mapping[self.tokenizer.name_or_path]
    self.end_of_instr = self.tokenizer.encode(self.start_of_assistant_response, return_tensors='pt', add_special_tokens=False)[0]
    self.instruction_training = instruction_training
    
  def set_instruction_training(self, instruction_training: bool):
    self.instruction_training = instruction_training
    
  def set_right_padding(self):
    self.right_padding = True # sft
    self.left_padding = False # rlft
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.tokenizer.padding_side = 'right'

  def set_left_padding(self):
    self.right_padding = False # sft
    self.left_padding = True   # rlft
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.tokenizer.padding_side = 'left'
    
  def process_batch(self, batch: List[Dict[str, Any]]):
    for item in batch:
      if 'chat' in item.keys():
        try:
          item['chat'] = json.loads(item['chat'])
          item['text'] = self.tokenizer.apply_chat_template(item['chat'], tokenize=False)
        except json.JSONDecodeError:
          item['text'] = item['chat']
    return batch
  
  def __call__(self, batch: List[Dict[str, Any]]):
    """
    Args:
        batch (list): a list of samples to collate. list of dicts.

    Returns:
        dict: a dictionary of collated samples
    """
    # import code; code.interact(local=dict(globals(), **locals()))
    batch = self.process_batch(batch)
    keys = list(batch[0].keys())
    if self.right_padding:
      # convert list of dicts to dict of lists
      batch = {key: [item[key] for item in batch] for key in keys}
      tokenized_batch = self.tokenize_fn(batch, self.tokenizer)
      if self.instruction_training:
        batch_size = tokenized_batch.labels.size(0)
        seq_length = tokenized_batch.labels.size(1)
        end_of_instr_size = self.end_of_instr.size(0)
        for i in range(batch_size):
          for pos in range(seq_length - end_of_instr_size + 1):
            if torch.equal(tokenized_batch.labels[i, pos:pos+end_of_instr_size], self.end_of_instr):
              tokenized_batch.labels[i, :pos+end_of_instr_size].fill_(-100)
              break
    elif self.left_padding:
      # convert list of dicts to dict of lists
      batch = {key: [get_user_prompt([item[key]], system_tag=None, instr_tag="[/INST]")[0]+"[/INST]" if key == 'text' else item[key] for item in batch] for key in keys}
      tokenized_batch = self.tokenize_fn(batch, self.tokenizer)
    tokenized_batch = PinMemoryData(tokenized_batch)
    return tokenized_batch


class ATPGCollate:
  def __init__(self, tokenizer, max_freq, possible_labels2id, max_num_of_patterns_per_circuit):
    self.tokenizer                       = tokenizer
    self.max_freq                        = max_freq
    self.possible_labels2id              = possible_labels2id
    self.max_num_of_patterns_per_circuit = max_num_of_patterns_per_circuit
  
  def __call__(self, batch):
    batch = {key: [item[key] for item in batch] for key in batch[0].keys()}
    tokenized_inputs = self.tokenizer(batch['prompts'], padding=True, truncation=True, return_tensors='pt')
    tokenized_inputs['input_ids_lengths'] = length_less_than_model_max_len(tokenized_inputs, self.tokenizer, train_test=None, key='input_ids', token=0)
    answers_sorted_byline = ['\n'.join(sorted(patterns.split('\n'))) for patterns in batch['patterns']]
    tokenized_targets = self.tokenizer(text_target=answers_sorted_byline, padding=True, truncation=True, return_tensors='pt')
    tokenized_inputs['labels'] = tokenized_targets['input_ids'] if type(self.tokenizer).__name__ == 'ATPGTokenizer' else tokenized_targets['input_ids'][:, 1:-1]
    tokenized_inputs['labels'] = tokenized_inputs['labels'].view(-1, tokenized_inputs['labels'].shape[-1]//self.max_freq, self.max_freq)

    # Create that many samples as required to match the max number of patterns. Not only in the batch!
    temp       = torch.zeros(tuple(dim if i!=1 else self.max_num_of_patterns_per_circuit - tokenized_inputs['labels'].shape[1] for i, dim in enumerate(tokenized_inputs['labels'].shape)))
    # Trace the indices of the labels.
    bool_true  = torch.ones_like(tokenized_inputs['labels'], dtype=torch.bool)
    bool_false = torch.zeros_like(temp, dtype=torch.bool)
    # Concatenate the booleans and the labels-padding
    valid_indices       = torch.cat([bool_true, bool_false], dim=1)
    tokenized_inputs['labels'] = torch.cat((tokenized_inputs['labels'], temp), dim=1)

    assert tokenized_inputs['labels'].shape == valid_indices.shape

    matrix = [torch.full((2**self.max_freq,), 0.)  for _ in range(tokenized_inputs['labels'].shape[0])]
    for b in range(tokenized_inputs['labels'].shape[0]):
      for pattern, _idx in zip(tokenized_inputs['labels'][b], valid_indices[b]):
        if patterns_contains_special_tokens(self.tokenizer, pattern):
          continue
        if _idx.all().item():
          matrix[b][self.possible_labels2id[self.tokenizer.decode(pattern.to(int))]] = 1.
    tokenized_inputs['labels'] = torch.stack(matrix)

    valid_samples = tokenized_inputs['input_ids_lengths'] != -1
    for key in tokenized_inputs:
      tokenized_inputs[key] = tokenized_inputs[key][valid_samples]
    
    return tokenized_inputs
