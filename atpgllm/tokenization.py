from typing import List
import regex as re
from atpgllm.utils import load_raw_dataset, AttrDict
from os.path import join, dirname, abspath

# Pattern for netlist split. Find gate types, nets, instance names
NETLIST_PATTERN = r""" ?\p{L}+| ?_\p{N}+_"""
NET_PATTERN = r""" ?_\p{N}+_"""

def get_new_tokens(hps: AttrDict) -> List[str]:
  """
  This function is used to extract new tokens from the given dataset.
  Args:
      hps (AttrDict): Hyperparameters.
  Returns:
      List[str]: List of new tokens.
  """
  from tokenizers import AddedToken
  # load raw dataset
  # raw_dataset = load_raw_dataset(hps.data_file)

  # this csv contains ALL generated netlists 
  # raw_dataset = load_raw_dataset(join(dirname(abspath(__file__)), "../../data/atpg_data_random_pis_v3.csv")) 
  raw_dataset = load_raw_dataset(hps.data_file)

  # compile patterns
  compiled_pattern = re.compile(NETLIST_PATTERN)
  compiled_net_pattern = re.compile(NET_PATTERN)
  # parse dataset -> Netlist
  tokens = set()
  for netlist in raw_dataset['train']['netlist']:
    # split netlist in chunks of text by categories defined in regex pattern
    tokens = tokens.union(set(compiled_pattern.findall(netlist)))
  # parse dataset -> Patterns
  # TODO: 'patterns'
  # text = [patterns for patterns in raw_dataset['train']['patterns']]
  # # split netlist in chunks of text by categories defined in regex pattern
  # tokens = tokens.union(set(compiled_pattern.findall(netlist)))
  tokens = sorted(set([t.strip() for t in tokens]))
  tokens_list = []
  for token in tokens:
    # gate names
    if any(token==gate for gate in ["nor", "xor", "xnor", "nand", "buf"]): 
      tokens_list.append(AddedToken(token, lstrip=False, rstrip=True, normalized=False, single_word=True))
    # gate names
    # NOTE: Be careful with these 3 tokens, since they can be actual english words
    elif any(token==gate for gate in ["or", "and", "not"]): 
      tokens_list.append(AddedToken(token, lstrip=False, rstrip=False, normalized=True, single_word=False))
    # gate names
    # elif any(token==gate for gate in ['AN', 'IBUF', 'ND', 'NR', 'OR', 'XNR', 'XOR']):
    #   tokens_list.append(AddedToken(token, lstrip=False, rstrip=True, normalized=False, single_word=False))
    # gate names
    elif token=='IV': 
      tokens_list.append(AddedToken(token, lstrip=False, rstrip=False, normalized=False, single_word=True))
    # net names
    elif compiled_net_pattern.match(token):
      tokens_list.append(AddedToken(token, lstrip=False, rstrip=False, normalized=False, single_word=False))
    else:
      pass
      # tokens_list.append(AddedToken(token, lstrip=True, rstrip=False, normalized=True, single_word=False))
  return tokens_list
