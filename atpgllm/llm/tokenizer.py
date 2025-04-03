import numpy as np
import torch

class ATPGTokenizer(object):
  def __init__(self, vocab, max_length=512):
    self.special_tokens_fn()
    self.init_vocab(vocab)
    self.stoi = {word: i for i, word in enumerate(self.vocab)}
    self.itos = {i: word for i, word in enumerate(self.vocab)}
    self.model_max_length = max_length

  def init_vocab(self, vocab):
    self.vocab = sum(vocab.values.tolist(), [])
    self.vocab += ['0', '1', '.']
    self.vocab = sorted(self.vocab)
    self.vocab = self.vocab[::-1] + self.all_special_tokens
    self.vocab = self.vocab[::-1]
    self.vocab_size = len(vocab)

  def __call__(self, s=None, text_target=None, padding=False, truncation=False, return_tensors='pt'):
    s = s if text_target is None else text_target
    s = np.array(s)
    # s = np.char.add(self.bos[0], s)
    for sp_token in self.special_tokens:
      s = np.char.replace(s, sp_token, f' {sp_token} ')
    tokens = np.expand_dims(np.char.split(s, ' '), axis=0)
    # remove empty str in list of strings
    if isinstance(tokens[0][0], str):
      tokens[0] = [token for token in tokens[0] if token not in ['','\n'] and token in self.vocab and token not in self.masked_tokens ] # and token not in self.masked_tokens
    # remove empty str in list of list of strings
    elif isinstance(tokens[0][0], list):
      tokens[0] = [[_token for _token in _tokens if _token not in ['','\n'] and _token in self.vocab and _token not in self.masked_tokens ] for _tokens in tokens[0]] # and _token not in self.masked_tokens
    #print(tokens)
    #exit()
    """
    marks       = 1
    keywords    = 2
    ports       = 3
    output_nets = 4
    input_nets  = 5
    attn_mask   = 0
    token_type_section = 'KEYWORDS'
    """
    self.input_ids      = []
    self.attention_mask = []
    self.token_type_ids = []
    # list of strings
    if isinstance(tokens[0][0], str):
      for token in tokens[0]:
        self.input_ids.append(self.stoi[token])
        self.attention_mask[-1].append(1)
        self.token_type_ids[-1].append(0)
        """
        # add ids, token_type, mask
        if token == 'module':
          token_type_section = 'KEYWORDS'
          module_ports       = True
        elif token == '(':
          if module_ports == True:
            token_type_section = 'PORTS'
          else:
            token_type_section = 'NETS'
        elif token == ')':
          token_type_section = 'KEYWORDS'
          gates_first_net    = True
        
        #if token == ';':
        #  self.input_ids.append(self.stoi[token])
        #  self.token_type_ids.append(marks)
        #  self.attention_mask.append(1)
        if token == 'endmodule' or token == 'module':
          attn_mask += 1
          self.input_ids[-1].append(self.stoi[token])
          self.token_type_ids[-1].append(keywords)
          self.attention_mask[-1].append(1)
          attn_mask += 1
        elif token == ';' or token == ',' or token == '(' or token == ')':
          self.token_type_ids.append(marks)
          self.attention_mask.append(1)
          #if token == ';' and attn_mask == 2:
          #  attn_mask = 3
          #if token == ';':
          #  attn_mask += 1
        elif token_type_section == 'KEYWORDS':
          self.input_ids.append(self.stoi[token])
          self.token_type_ids.append(keywords)
          self.attention_mask.append(1)
        elif token_type_section == 'PORTS':
          self.input_ids.append(self.stoi[token])
          self.token_type_ids.append(ports)
          self.attention_mask.append(1)
        elif token_type_section == 'NETS':
          self.input_ids.append(self.stoi[token])
          if gates_first_net == True:
            self.token_type_ids.append(output_nets)
            self.attention_mask.append(1)
          else:
            self.token_type_ids.append(input_nets)
            self.attention_mask.append(1)
        #print(f"{token}\n{self.token_type_ids}\n{self.attention_mask}")
        """
    # list of list of string
    elif isinstance(tokens[0][0], list):
      for _tokens in tokens[0]:
        self.input_ids.append([])
        self.attention_mask.append([])
        self.token_type_ids.append([])
        for token in _tokens:
          self.input_ids[-1].append(self.stoi[token])
          self.attention_mask[-1].append(1)
          self.token_type_ids[-1].append(0)
          """
          if token == 'module':
            token_type_section = 'KEYWORDS'
            module_ports       = True
          elif token == '(':
            if module_ports == True:
              token_type_section = 'PORTS'
              module_ports = False
            else:
              token_type_section = 'NETS'
          elif token == ')':
            token_type_section = 'KEYWORDS'
            gates_first_net    = True
        
          #if token == ';':
          #  self.input_ids[-1].append(self.stoi[token])
          #  self.token_type_ids[-1].append(marks)
          #  self.attention_mask[-1].append(1)
          if token == 'endmodule' or token == 'module':
            attn_mask += 1
            self.input_ids[-1].append(self.stoi[token])
            self.token_type_ids[-1].append(keywords)
            self.attention_mask[-1].append(1)
            attn_mask += 1
          elif token == ';' or token == ',' or token == '(' or token == ')':
            self.input_ids[-1].append(self.stoi[token])
            self.token_type_ids[-1].append(marks)
            self.attention_mask[-1].append(1)
            #if token == ';' and attn_mask == 2:
            #  attn_mask = 3
            #if token == ';':
            #  attn_mask += 1
          elif token_type_section == 'KEYWORDS':
            self.input_ids[-1].append(self.stoi[token])
            self.token_type_ids[-1].append(keywords)
            self.attention_mask[-1].append(1)
          elif token_type_section == 'PORTS':
            self.input_ids[-1].append(self.stoi[token])
            self.token_type_ids[-1].append(ports)
            self.attention_mask[-1].append(1)
          elif token_type_section == 'NETS':
            self.input_ids[-1].append(self.stoi[token])
            if gates_first_net == True:
              self.token_type_ids[-1].append(output_nets)
              self.attention_mask[-1].append(1)
              gates_first_net = False
            else:
              self.token_type_ids[-1].append(input_nets)
              self.attention_mask[-1].append(1)
          #print(f"{token}\n{self.token_type_ids}\n{self.attention_mask}")
          """
        #print(f"{_tokens}\n{self.input_ids}\n{self.token_type_ids}\n{self.attention_mask}")
        #exit()
        self.input_ids[-1]      = np.array(self.input_ids[-1], dtype=float)
        self.attention_mask[-1] = np.array(self.attention_mask[-1], dtype=float)
        self.token_type_ids[-1] = np.array(self.token_type_ids[-1], dtype=float)

    # set maximum size for padding
    if isinstance(self.input_ids[0], np.ndarray):
      if padding == True:
        max_size_padding = min(max(i.shape[0] for i in self.input_ids), self.model_max_length)
        if text_target:
          max_size_padding = min(max(i.shape[0] for i in self.input_ids), self.model_max_length)
      elif padding == 'max_length':
        max_size_padding = self.model_max_length
      elif padding == 'longest':
        max_size_padding = max([i.shape[0] for i in self.input_ids])

      # set maximum size for truncation
      if truncation == True:
        max_size_truncation = self.model_max_length
        if text_target:
          max_size_truncation = min(max(i.shape[0] for i in self.input_ids), self.model_max_length)
      elif truncation == 'max_length':
        max_size_truncation = self.model_max_length
      elif truncation == 'longest':
        max_size_truncation = max([i.shape[0] for i in self.input_ids])
    else:
      max_size_padding    = len(self.input_ids)
      max_size_truncation = len(self.input_ids)

    # Padding
    if padding:
      # there is no reason to add padding if a single input was given. Padding applied only for batches
      if isinstance(self.input_ids[0], np.ndarray):
        # Add padding
        for i, (input_ids, attention_mask, token_type_ids) in enumerate(zip(self.input_ids, self.attention_mask, self.token_type_ids)):
          while len(input_ids) < max_size_padding:
            input_ids      = np.append(input_ids, self.stoi[self.pad[0]])
            attention_mask = np.append(attention_mask, 0)
            token_type_ids = np.append(token_type_ids, 0)
          self.input_ids[i]      = input_ids
          self.attention_mask[i] = attention_mask
          self.token_type_ids[i] = token_type_ids
      else:
        print(f"padding: {type(self.input_ids[0])}")

    # Truncate samples
    if truncation:
      if isinstance(self.input_ids[0], np.ndarray):
        # truncate
        for i, (input_ids, attention_mask, token_type_ids) in enumerate(zip(self.input_ids, self.attention_mask, self.token_type_ids)):
          if len(input_ids) <= max_size_truncation and attention_mask[-1] == 0:
            continue
          if len(input_ids) >= max_size_truncation:
            input_ids      = input_ids[:max_size_truncation-1]
            attention_mask = attention_mask[:max_size_truncation-1]
            token_type_ids = token_type_ids[:max_size_truncation-1]

            input_ids      = np.append(input_ids, self.stoi[self.eos[0]])
            attention_mask = np.append(attention_mask, 1)
            token_type_ids = np.append(token_type_ids, 0)
          self.input_ids[i]      = input_ids
          self.attention_mask[i] = attention_mask
          self.token_type_ids[i] = token_type_ids
      else:
        print(f"truncation: {type(self.input_ids[0])}")

    self.input_ids      = np.array(self.input_ids, dtype=float)
    self.attention_mask = np.array(self.attention_mask, dtype=float)
    self.token_type_ids = np.array(self.token_type_ids, dtype=float)

    self.input_ids      = np.expand_dims(self.input_ids, axis=0) if not isinstance(self.input_ids[0], np.ndarray) else self.input_ids
    self.attention_mask = np.expand_dims(self.attention_mask, axis=0) if not isinstance(self.attention_mask[0], np.ndarray) else self.attention_mask
    self.token_type_ids = np.expand_dims(self.token_type_ids, axis=0) if not isinstance(self.token_type_ids[0], np.ndarray) else self.token_type_ids

    self.input_ids      = torch.from_numpy(self.input_ids) if return_tensors == 'pt' else self.input_ids
    self.attention_mask = torch.from_numpy(self.attention_mask) if return_tensors == 'pt' else self.attention_mask
    self.token_type_ids = torch.from_numpy(self.token_type_ids) if return_tensors == 'pt' else self.token_type_ids

    res = {'input_ids': self.input_ids, 'attention_mask': self.attention_mask, 'token_type_ids': self.token_type_ids}
    return res

  def special_tokens_fn(self):
    self.all_special_tokens = ['(', ')', ',', ';', '<s>', '</s>', '<unk>', '<pad>']
    self.special_tokens = ['(', ')', ',', ';', '\n']
    self.masked_tokens = ['(', ')', ',']
    self.eol = [';']
    self.nl  = ['\n']
    self.bos = ['<s>']
    self.eos = ['</s>']
    self.unk = ['<unk>']
    self.pad = ['<pad>']
    self.not_printable = ['<s>', '</s>', '<unk>', '<pad>']

  def convert_tokens_to_ids(self, s):
    pass

  def convert_ids_to_tokens(self, input_ids, attention_mask=None):
    input_ids = input_ids.numpy() if isinstance(input_ids, torch.Tensor) else np.array(input_ids)
    if attention_mask is not None:
      attention_mask = attention_mask.numpy() if isinstance(attention_mask, torch.Tensor) else np.array(attention_mask)
      itos = lambda batch_ids, batch_mask: [self.itos[integer] for sample_ids, sample_mask in zip(batch_ids, batch_mask) for integer, mask in zip(sample_ids, sample_mask) if mask != 0]
      return itos(input_ids, attention_mask)
    else:
      if len(input_ids.shape) > 1:
        itos = lambda batch_ids: [self.itos[integer] for sample_ids in batch_ids for integer in sample_ids]
      else:
        itos = lambda sample_ids: [self.itos[integer] for integer in sample_ids]
      return itos(input_ids)

  def decode(self, input_ids, attention_mask=None):
    tokens = self.convert_ids_to_tokens(input_ids, attention_mask)
    # TODO: should I remove the special tokens?
    tokens = [token for token in tokens]
    # concatenate all the tokens in a single string
    res = ' '.join(tokens)
    return res

  def get_attention_mask(self, s):
    return [1 if w in self.stoi.keys() else 0 for w in s]


def tokenize_fn(batched, tokenizer, is_causal=True):
  if is_causal:
    tokenized_inputs = tokenizer(batched['text'], 
                                truncation=True, # truncates to the specified tokenizer's maximum length.
                                padding=True, # add eos-special-tokens-id, i.e. the number 2. ID of '</s>'
                                add_special_tokens=False, # do not add bos-special-token-id, i.e. the number 1. ID of '<s>'
                                return_tensors='pt', # return PyTorch tensors.
                                )
    tokenized_inputs['netlist'] = batched['netlist']
    tokenized_inputs['labels']  = tokenized_inputs['input_ids'].clone().detach()
  else:
    tokenized_inputs = tokenizer([b['prompts'] for b in batched['text']], truncation=True, padding=True, add_special_tokens=False, return_tensors='pt')
    tokenized_labels = tokenizer([b['answers'] for b in batched['text']], truncation=True, padding=True, add_special_tokens=False, return_tensors='pt')
    tokenized_inputs['labels']  = tokenized_labels['input_ids']
    tokenized_inputs['netlist'] = batched['netlist']
  return tokenized_inputs
