import numpy as np
import torch
import re

class GraphTokenizer(object):
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
    if text_target is None:
      s = np.array(s)
      res = self.parse_nodes(s, padding, truncation, return_tensors)
    else:
      s = np.array(text_target)
      res = self.parse_labels(s, padding, truncation, return_tensors)
    return res
  
  def parse_labels(self, s, padding=False, truncation=False, return_tensors='pt'):
    #print('parse_labels:', s[0])
    for sp_token in self.special_tokens:
      s = np.char.replace(s, sp_token, f' {sp_token} ')
    tokens = np.expand_dims(np.char.split(s, ' '), axis=0)
    # remove empty str in list of strings
    if isinstance(tokens[0][0], str):
      tokens[0] = [token for token in tokens[0] if token not in ['','\n'] and token in self.vocab ] # and _token not in self.masked_tokens
    # remove empty str in list of list of strings
    elif isinstance(tokens[0][0], list):
      tokens[0] = [[_token for _token in _tokens if _token not in ['','\n'] and _token in self.vocab ] for _tokens in tokens[0]] # and _token not in self.masked_tokens
    #print(tokens)
    #exit()
    self.input_ids      = []
    self.attention_mask = []
    self.token_type_ids = []
    # list of strings
    if isinstance(tokens[0][0], str):
      for token in tokens[0]:
        self.input_ids.append(self.stoi[token])
        self.attention_mask[-1].append(1)
        self.token_type_ids[-1].append(0)
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
        self.input_ids[-1]      = np.array(self.input_ids[-1], dtype=float)
        self.attention_mask[-1] = np.array(self.attention_mask[-1], dtype=float)
        self.token_type_ids[-1] = np.array(self.token_type_ids[-1], dtype=float)

    # set maximum size for padding
    if isinstance(self.input_ids[0], np.ndarray):
      if padding == True:
        max_size_padding = min(max(i.shape[0] for i in self.input_ids), self.model_max_length)
      elif padding == 'max_length':
        max_size_padding = self.model_max_length
      elif padding == 'longest':
        max_size_padding = max([i.shape[0] for i in self.input_ids])

      # set maximum size for truncation
      if truncation == True:
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
   

  def parse_nodes(self, s, padding=False, truncation=False, return_tensors='pt'):
    regex = re.compile(r'(\w+)\s+(_\w+_\b)\s*\(([\s\w,]+)\);')
    nodes = np.array([np.array(regex.findall(line)) for line in s], dtype=object) # batch_size x Nodes x Features
    features = np.array([np.char.split(features[2], ',') for node in nodes for features in node if isinstance(features[2], str)])
    
    # maximum length of gates ports + 1 gate type
    max_features_size = max([len(f.tolist()) for f in features]) + 1
    # maximum number of nodes
    max_nodes_size = max([node.shape[0] for node in nodes])

    # Batch_Size x Nodes x Features
    data = np.zeros((len(nodes), max_nodes_size, max_features_size))
    for i, node in enumerate(nodes):
      for j, (x, y) in enumerate(zip(node[:, 0], node[:, 2])):
        y = np.char.split(y, ',')
        data[i, j, 0] = self.stoi[x.strip()]
        for k, yy in enumerate(y.tolist()):
          data[i, j, k+1] = self.stoi[yy.strip()]
    #print(data[:3])
    #print(s[:3])
    #print(data.shape)
    # List all nodes per circuit in a batch
    nodes_per_circuit = [np.count_nonzero(nodes[:,0]) for nodes in data]

    # List to hold all adjacency matrices
    adj_matrix  = []
    max_nodes = max(nodes_per_circuit)
    edge_index = []
    i = 0 
    j = 0
    for nodes, d in zip(nodes_per_circuit, data): # iterate over each sample of the batch
      #print(f"nodes: {nodes}, d: {d}")
      #print(self.decode(d))
      edge_indices = []
      adj_m = np.zeros((nodes, nodes))
      np.fill_diagonal(adj_m, 1)
      for output_port in d[:, 1]:
        if output_port != 0:
          # x: nodes indices, 
          # y: nodes' features indices
          x, y = np.where(output_port == d)
          # assert that y[0] equalt to 1 of y list. This index is the feature location of output net, thus should indicate the source node
          assert y[0] == 1
          # This condition filters the cells are not driven by any gate. i.e. output gates.
          if len(x) != 2:
            #edge_indices.append([x[0], x[0]])
            #print(f"len(x) != 2: x-> {edge_indices[-1]}")
            continue
          edge_indices.append(x.tolist())
          #print(f"Properly added to the list -> {edge_indices[-1]}")
          # Get the source node.
          src = x[0]
          # Get destination by filtering out the source from all the other nodes
          dest = list(x)
          dest.remove(src)
          src = [src]
          # Create the adjacency matrix
          for dest_x in dest:
            adj_m[src, dest_x] = 1
       
      # put all adj_m under the same variable. Batch them all together
      adj_m = np.pad(adj_m, (0, max_nodes - nodes), mode='constant') if nodes < max_nodes else adj_m
      if len(adj_m) > 0:
        adj_matrix.append(adj_m)
      else:
        j += 1

      # Get the Directed Acyclic Graph's connections and create a bi-directional graph + self-loops 
      if len(edge_indices) > 0: # drop 
        #print(f"edge_indices: {edge_indices}, source-destination: {list(zip(*edge_indices))}")
        edges_data = []
        x_set = set()
        x = list(zip(*edge_indices))
        
        #print(f"x: {x}")
        x_set = x_set.union(set(x[0]))
        x_set = x_set.union(set(x[1]))
        x_list = list(x_set)
        # Initialize bi-directional graph by adding the self-loops first
        #         [source, destination]
        x_selfloop = [x_list, x_list.copy()]
    
        # Create bi-directional graph
        x_bidirect = x_selfloop.copy()
        x_bidirect[0].extend(x[0])
        x_bidirect[1].extend(x[1])
        x_bidirect[1].extend(x[0])
        x_bidirect[0].extend(x[1])
        edges_data.append(np.array(x_bidirect))
        edge_index.append(np.squeeze(np.array(edges_data)))
        edge_index[-1] = torch.tensor(edge_index[-1]) if return_tensors == 'pt' else edge_index[-1]
        #edge_index.append(np.array(list(zip(*edge_indices))))
      else:
        edge_index.append(np.array([[0], [0]]))
        edge_index[-1] = torch.tensor(edge_index[-1]) if return_tensors == 'pt' else edge_index[-1]
        i += 1
      #print(f"{len(edge_index)}: {len(edge_index[-1])} edge indices:\n{edge_index[-1]}")
    #print(f"total leftovers: {i}, {j}")
    #print(f"{len(edge_index)}, {edge_index[0].shape}, {edge_index[1].shape}")
    # create the adjacency matrix
    adj_matrix = np.stack(adj_matrix, axis=0)
    # Create edge index matrix. batch_size x 2 x num_edges. The second dimension which is equal to 2 contains the source and destination nodes of the edges
    #print(len(edge_index), [i.shape for i in edge_index])
    max_edges  = max([i.shape[1] for i in edge_index])
   
    # pad the array
    #print(f"{edge_index[0]}\n{edge_index[1]}\n{type(edge_index[0])}")
    #print(f"index: {len(edge_index[0]) + len(edge_index[1])}")
    #print(f"matrix: {adj_matrix.shape} {adj_matrix.shape[0] * adj_matrix.shape[1]}")
    
    # The padding format is ((top, bottom), (left, right))
    # As of Dec. 4th I have decided I need to change the way that I calculate 'edge_index'
    #edge_index = np.array([np.pad(e, ((0, 0), (0, max_edges - e.shape[1])), mode='constant', constant_values=0) if e.shape[1] < max_edges else e for e in edge_index])
    #edge_index = np.pad(edge_index, ((0, data.shape[0] - edge_index.shape[0]), (0, 0), (0, 0)), mode='constant', constant_values=0)
    
    
    self.adj_matrix = torch.from_numpy(adj_matrix) if return_tensors == 'pt' else adj_matrix
    self.data       = torch.from_numpy(data)       if return_tensors == 'pt' else data
    self.edge_index = edge_index
    #self.edge_index = torch.from_numpy(edge_index) if return_tensors == 'pt' else edge_index
    #print(self.edge_index.shape, self.data.shape, self.adj_matrix.shape)

    res = {'adj_matrix': self.adj_matrix, 'nodes': self.data, 'edge_index': self.edge_index}
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

