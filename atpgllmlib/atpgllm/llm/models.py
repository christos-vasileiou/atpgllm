from transformers import BertModel, AutoModel, AutoConfig
from atpgllm.utils import *
import torch.nn as nn
import torch.nn.functional as F

models = {
    "t5-small": "google/flan-t5-small",
    "t5-base": "google/flan-t5-base",
    "t5-v1_1-base": "google/t5-v1_1-base",
    "t5-large": "google/flan-t5-large",
    "t5-xl": "google/flan-t5-xl",
    "t5-xxl": "google/flan-t5-xxl",
    "mt5-base": "google/mt5-base",
    "m2m100": "facebook/m2m100_418M",
    "t5-finetuned": "mrm8488/t5-base-finetuned-common_gen",
    "led-base": "allenai/led-base-16384",
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "gpt2": "gpt2" 
}

class GPT(torch.nn.Module):
  def __init__(self, model_checkpoint, dropout_p, patterns_len):
    super(GPT, self).__init__()
    model_checkpoint = 'gpt2'
    self.feat_extr = AutoModel.from_pretrained(models[model_checkpoint])
    print(self.feat_extr)
    layer = next((l for l in reversed(list(self.feat_extr.modules())) if hasattr(l, 'out_features')), None)
    print(layer.__dir__())
    self.head = nn.Sequential(
                       #nn.TransformerEncoderLayer(d_model=layer.out_features, nhead=8, batch_first=True),
                       #nn.TransformerEncoderLayer(d_model=layer.out_features, nhead=8, batch_first=True),
                       #nn.TransformerEncoderLayer(d_model=layer.out_features, nhead=8, batch_first=True),
                       #nn.Linear(layer.out_features, 256, bias=True),
                       #nn.Dropout(0.1),
                       #nn.LayerNorm(256),
                       #LambdaLayer(lambda x: x.squeeze()),
                       #LambdaLayer(lambda x: F.adaptive_avg_pool1d(x, 2**patterns_len)), #nn.Dropout(dropout_p),
                       #nn.TransformerEncoderLayer(d_model=layer.out_features, nhead=8, batch_first=True),
                       #nn.Linear(256, 64, bias=True),
                       #nn.Dropout(0.1),
                       nn.Linear(768, 256),
                       nn.LayerNorm(256),
                       nn.Dropout(dropout_p),
                       nn.Linear(256, 2**patterns_len),
                       #nn.LayerNorm(2**patterns_len),
    )
    #self.initialize_weights(self.feat_extr)
    self.initialize_weights(self.head)
    self.patterns_len = patterns_len

  def forward(self, ids, mask, token_type_ids):
    x, _ = self.feat_extr(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
    x = x[:, 0]
    
    #print(f"last_hidden_state: {x['last_hidden_state'].shape}")
    #print(f"past_key_values: {len(x['past_key_values'])}")
    #print(f"encoder_last_hidden_state: {x['encoder_last_hidden_state'].shape}")
    x = self.head(x)
    #for head_layer in self.head:
    #  x = head_layer(x)
    x = x.view(-1, 2**self.patterns_len)
    return x

  def initialize_weights(self, model):
    """Initialize the weights of the model."""
    for module in model.modules():
      # Linear layers
      if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
          nn.init.constant_(module.bias.data, 0)
        # LayerNorm layers
      elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias.data, 0)
        nn.init.constant_(module.weight.data, 1.0)
      # Embedding layers
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight.data, mean=0, std=0.02)
 
class T5(torch.nn.Module):
  def __init__(self, model_checkpoint, dropout_p, patterns_len):
    super(T5, self).__init__()
    model_checkpoint = 't5-base'
    self.feat_extr = AutoModel.from_pretrained(models[model_checkpoint])
    layer = next((l for l in reversed(list(self.feat_extr.modules())) if hasattr(l, 'out_features')), None)
    self.head = nn.Sequential(
                       #nn.TransformerEncoderLayer(d_model=layer.out_features, nhead=8, batch_first=True),
                       #nn.TransformerEncoderLayer(d_model=layer.out_features, nhead=8, batch_first=True),
                       #nn.TransformerEncoderLayer(d_model=layer.out_features, nhead=8, batch_first=True),
                       #nn.Linear(layer.out_features, 256, bias=True),
                       #nn.Dropout(0.1),
                       #nn.LayerNorm(256),
                       #LambdaLayer(lambda x: x.squeeze()),
                       #LambdaLayer(lambda x: F.adaptive_avg_pool1d(x, 2**patterns_len)), #nn.Dropout(dropout_p),
                       #nn.TransformerEncoderLayer(d_model=layer.out_features, nhead=8, batch_first=True),
                       #nn.Linear(256, 64, bias=True),
                       #nn.Dropout(0.1),
                       #nn.LayerNorm(64),
                       nn.Linear(layer.out_features, 256),
                       nn.Dropout(dropout_p),
                       nn.Linear(256, 2**patterns_len),
                       #nn.LayerNorm(2**patterns_len),
    )
    #self.initialize_weights(self.feat_extr)
    self.initialize_weights(self.head)
    self.patterns_len = patterns_len

  def forward(self, ids, mask, token_type_ids):
    x = self.feat_extr(input_ids=ids, attention_mask=mask, decoder_input_ids=ids, return_dict=True)
    #print(f"last_hidden_state: {x['last_hidden_state'].shape}")
    #print(f"past_key_values: {len(x['past_key_values'])}")
    #print(f"encoder_last_hidden_state: {x['encoder_last_hidden_state'].shape}")
    x = self.head(x['last_hidden_state'])
    #for head_layer in self.head:
    #  x = head_layer(x)
    x = x[:, 0]
    return x

  def initialize_weights(self, model):
    """Initialize the weights of the model."""
    for module in model.modules():
      # Linear layers
      if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
          nn.init.constant_(module.bias.data, 0)
        # LayerNorm layers
      elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias.data, 0)
        nn.init.constant_(module.weight.data, 1.0)
      # Embedding layers
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight.data, mean=0, std=0.02)
 
class LambdaLayer(nn.Module):
  def __init__(self, lambda_function):
    super(LambdaLayer, self).__init__()
    self.lambda_func = lambda_function

  def forward(self, x):
    return self.lambda_func(x)

class ResLinear(nn.Module):
  def __init__(self, in_features, out_features, dropout_p, skip_layers=1, intermediate_features=256):
    super(ResLinear, self).__init__()
    self.skip_layers = skip_layers 
    assert in_features == out_features
    if skip_layers == 1:
      self.linear1 = nn.Linear(in_features, out_features)
      self.dropout1 = nn.Dropout(dropout_p)
    elif skip_layers == 2:
      self.linear1 = nn.Linear(in_features, intermediate_features)
      self.linear2 = nn.Linear(intermediate_features, out_features)
      self.dropout1 = nn.Dropout(dropout_p)
      self.dropout2 = nn.Dropout(dropout_p)

  def forward(self, x):
    x_skip = x
    if self.skip_layers == 1:
      x = self.dropout1(self.linear1(x))
      x += x_skip
    elif self.skip_layers == 2:
      x = self.dropout1(self.linear1(x))
      x = self.dropout2(self.linear2(x))
      x += x_skip
    return x

class BERT(torch.nn.Module):
  def __init__(self, model_checkpoint, dropout_p, patterns_len):
    super(BERT, self).__init__()
    model_checkpoint = 'bert'
    
    configuration = AutoConfig.from_pretrained(models['bert'])
    configuration.hidden_dropout_prob = dropout_p
    configuration.attention_probs_dropout_prob = dropout_p

    self.feat_extr = AutoModel.from_pretrained(models[model_checkpoint], config=configuration )
    #self.l1.embeddings.position_embeddings = nn.Embedding(512, 768) #LambdaLayer(lambda x: x) 
    #print(self.feat_extr.embeddings.__dir__())
    #self.feat_extr.embeddings.token_type_embeddings = nn.Embedding(6, 768) #LambdaLayer(lambda x: x)
    layer = next((l for l in reversed(list(self.feat_extr.modules())) if hasattr(l, 'out_features')), None)
    self.head = nn.Sequential(
                       nn.TransformerEncoderLayer(d_model=layer.out_features, nhead=8, batch_first=True, dropout=dropout_p, activation='gelu'),
                       #LambdaLayer(lambda x: x.squeeze()),
                       #LambdaLayer(lambda x: F.adaptive_avg_pool1d(x, 2**patterns_len)), #nn.Dropout(dropout_p),
                       nn.Linear(layer.out_features, 256),
                       nn.LayerNorm(256),
                       nn.Dropout(dropout_p),
                       #ResLinear(768, 768, dropout_p=dropout_p, skip_layers=1),
                       #ResLinear(768, 768, dropout_p=dropout_p, skip_layers=1),
                       #ResLinear(768, 768, dropout_p=dropout_p, skip_layers=1),
                       nn.Linear(256, 2**patterns_len),
    )
    self.initialize_weights(self.feat_extr)
    self.initialize_weights(self.head)
    self.patterns_len = patterns_len

  def forward(self, ids, mask, token_type_ids):
    last_hidden_state, pooler_output = self.feat_extr(ids, mask, token_type_ids, return_dict=False)
    x = self.head(pooler_output)
    #x = pooler_output
    #for head_layer in self.head:
    #  x = head_layer(x)
    x = x.view(-1, 2**self.patterns_len)
    return x

  def initialize_weights(self, model):
    """Initialize the weights of the model."""
    for module in model.modules():
      # Linear layers
      if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
          nn.init.constant_(module.bias.data, 0)
        # LayerNorm layers
      elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias.data, 0)
        nn.init.constant_(module.weight.data, 1.0)
      # Embedding layers
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight.data, mean=0, std=0.02)
  

class PositionalEncoding(nn.Module):
  def __init__(self, max_len, embedding_dim):
    super(PositionalEncoding, self).__init__()
    self.positional_embeddings = nn.Embedding(max_len, embedding_dim)
    # Initialize the positional embeddings using sine-cosine initialization
    self.positional_embeddings.weight.data = self._init_pos_emb(max_len, embedding_dim)
    # Make sure to tell PyTorch that these embeddings are not learnable
    #self.positional_embeddings.weight.requires_grad = False

  def _init_pos_emb(self, max_len, embedding_dim):
    pos_enc = np.array([
      [pos / np.power(10000.0, (i - i % 2) / embedding_dim) for i in range(embedding_dim)]
       for pos in range(max_len)
    ])
    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # dim 2i
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # dim 2i+1
    return torch.tensor(pos_enc, dtype=torch.float32)

  def forward(self, x):
    """
    x: size (B, T, F)
    """
    # take device
    device = x.device
    # take sequence length
    seq_len = x.size(1)
    # Generate a tensor of position indices
    position_indices = torch.arange(seq_len, dtype=torch.long, device=device)
    # Retrieve the positional embeddings
    return self.positional_embeddings(position_indices)

class TokenTypeEmbeddings(nn.Module):
  def __init__(self, type_vocab_size, embedding_dim):
    super().__init__()
    # Typically type_vocab_size is 2 for BERT: one for each sentence
    self.token_type_embeddings = nn.Embedding(type_vocab_size, embedding_dim)
  
  def forward(self, token_type_ids):
    # token_type_ids is a tensor of the same shape as the input_ids,
    # with zeros in places of Sentence A tokens and ones in places of Sentence B tokens.
    return self.token_type_embeddings(token_type_ids)

class MyModel(nn.Module):
  def __init__(self, patterns_len, vocab_size=10_000, max_len=512, embedding_dim=768, type_vocab_size=5, num_encoder_layers=12, num_decoder_layers=12, d_model=768, nhead=8, dim_feedforward=2048, dropout_p=0.1, activation='gelu'):
    super(MyModel, self).__init__()

    # Encoder
    self.enc_words_embedding       = nn.Embedding(vocab_size, embedding_dim, max_norm=True)
    self.enc_pos_encoding          = PositionalEncoding(max_len, embedding_dim)
    self.enc_token_type_embeddings = TokenTypeEmbeddings(type_vocab_size, embedding_dim)
    assert d_model == embedding_dim
    self.encoder                   = nn.ModuleList([nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_p, activation=activation, batch_first=True)
                                                    for _ in range(num_encoder_layers)])
    layer = next((l for l in reversed(list(self.encoder.modules())) if hasattr(l, 'out_features')), None)
    self.head = nn.Sequential(
                       nn.TransformerEncoderLayer(d_model=layer.out_features, nhead=8, batch_first=True),
                       nn.Linear(layer.out_features, 2048, bias=True),
                       nn.Dropout(dropout_p),
                       nn.Linear(2048, (2**patterns_len), bias=True),
                       nn.LayerNorm(2**patterns_len),
    ) 
    self.initialize_weights(self.encoder)
    self.initialize_weights(self.head)
    self.patterns_len = patterns_len

  # Experimentation
  def forward(self, enc_input, attention_mask, token_type_ids, dec_input=None, enc_mask=None, dec_mask=None):
    x = self.enc_words_embedding(enc_input)
    #print('enc_embedding', x.shape)
    x += self.enc_pos_encoding(enc_input)
    #print('enc_pos_encoding', x.shape)
    if token_type_ids is not None:
      x += self.enc_token_type_embeddings(token_type_ids) 
    #print('enc_token_type_embeddings', x.shape)
    for enc_layer in self.encoder:
      x = enc_layer(x)
    #print(enc_layer)
    #print(x.shape)

    x    = self.head(x)[:, 0, :]
    #print('head output', x.shape)
    x    = x.view(-1, 2**self.patterns_len)

    return x
  
  def initialize_weights(self, model):
    """Initialize the weights of the model."""
    for module in model.modules():
      # Linear layers
      if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
          nn.init.constant_(module.bias.data, 0)
        # LayerNorm layers
      elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias.data, 0)
        nn.init.constant_(module.weight.data, 1.0)
      # Embedding layers
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight.data, mean=0, std=0.02)

