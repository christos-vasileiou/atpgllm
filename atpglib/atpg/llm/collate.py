import torch
import torch.nn.functional as F
from atpg.utils import length_less_than_model_max_len, patterns_contains_special_tokens


class MyCollate:
  def __init__(self, tokenizer, is_causal=True):
    self.tokenizer = tokenizer
    self.is_causal = is_causal

  def __call__(self, batch):
    if self.is_causal:
      return self.forward_if_causal(batch)
    else:
      return self.forward_if_seq2seq(batch)

  def forward_if_seq2seq(self, batch):
    collated_batch = {}
    collated_batch['netlist'] = []
    collect_netlists = True
    for key in ['input_ids', 'attention_mask', 'labels']:
      temp     = []
      max_size = 0
      for item in batch:
        self.tokenizer
        temp.append(torch.tensor(item[key], dtype=torch.long))
        max_size = max(max_size, temp[-1].size(-1))
        if collect_netlists:
          collated_batch['netlist'].append(item['netlist'])
      collect_netlists = False
      padded_batch = [F.pad(t, (0, max_size - t.size(-1)), "constant", self.tokenizer.pad_token_id if key == 'input_ids' else 0) for t in temp]
      collated_batch[key] = torch.stack(padded_batch)
    return collated_batch

  def forward_if_causal(self, batch):
    collated_batch = {}
    collated_batch['netlist'] = []
    collect_netlists = True
    for key in ['input_ids', 'attention_mask']:
      temp     = []
      max_size = 0
      for item in batch:
        temp.append(torch.tensor(item[key], dtype=torch.long))
        max_size = max(max_size, temp[-1].size(-1))
        if collect_netlists:
          collated_batch['netlist'].append(item['netlist'])
      collect_netlists = False
      padded_batch = [F.pad(t, (0, max_size - t.size(-1)), "constant", self.tokenizer.pad_token_id if key == 'input_ids' else 0) for t in temp]
      collated_batch[key] = torch.stack(padded_batch)
    return collated_batch


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
