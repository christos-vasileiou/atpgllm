import torch
from atpg.utils import length_less_than_model_max_len, patterns_contains_special_tokens

class GraphCollate:
  def __init__(self, tokenizer, max_freq, possible_labels2id, max_num_of_patterns_per_circuit):
    self.tokenizer                       = tokenizer
    self.max_freq                        = max_freq
    self.possible_labels2id              = possible_labels2id
    self.max_num_of_patterns_per_circuit = max_num_of_patterns_per_circuit
  
  def __call__(self, batch):
    batch = {key: [item[key] for item in batch] for key in batch[0].keys()}
    tokenized_inputs  = self.tokenizer(batch['prompts'], padding=True, truncation=True, return_tensors='pt')
    answers_sorted_byline = ['\n'.join(sorted(patterns.split('\n'))) for patterns in batch['patterns']]
    tokenized_targets = self.tokenizer(text_target=answers_sorted_byline, padding=True, truncation=True, return_tensors='pt')

    tokenized_inputs['labels'] = tokenized_targets['input_ids']
    tokenized_inputs['labels'] = tokenized_inputs['labels'].view(-1, tokenized_inputs['labels'].shape[-1]//self.max_freq, self.max_freq)
    #tokenized_inputs['input_ids_lengths'] = length_less_than_model_max_len(tokenized_inputs, self.tokenizer, train_test=None, key='labels', token=0)
    # Create that many samples as required to match the max number of patterns. Not only in the batch!
    temp       = torch.zeros(tuple(dim if i!=1 else self.max_num_of_patterns_per_circuit - tokenized_inputs['labels'].shape[1] for i, dim in enumerate(tokenized_inputs['labels'].shape)))
    #print(f"len:\n{tokenized_inputs['input_ids_lengths'][0]}")
    #print(f"patterns:\n{batch['patterns'][0]}")
    #print(f"sorted_answers:\n{answers_sorted_byline[0]}")
    #print(f"ttargets:\n{tokenized_targets['input_ids'][0]}\n")
   
    # Trace the indices of the labels.
    bool_true  = torch.ones_like(tokenized_inputs['labels'], dtype=torch.bool)
    bool_false = torch.zeros_like(temp, dtype=torch.bool)
    # Concatenate the booleans and the labels-padding
    valid_indices = torch.cat([bool_true, bool_false], dim=1)
    #print('labels:', tokenized_inputs['labels'].shape, 'temp:', temp.shape, 'bool_true:', bool_true.shape, 'bool_false:', bool_false.shape, 'valid_indices:', valid_indices.shape)
    #print('input_lengths:', tokenized_inputs['input_ids_lengths'][0], 'labels:', tokenized_inputs['labels'][0])
    #print()
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

    return tokenized_inputs
    
