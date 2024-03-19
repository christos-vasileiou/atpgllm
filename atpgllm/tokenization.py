from typing import List
from tokenizers import AddedToken
import regex as re
from atpgllm.utils import load_raw_dataset, AttrDict

# Pattern for netlist split. Find gate types, nets, instance names
NETLIST_PATTERN = r""" ?\p{L}+| ?_\p{N}+_"""

def get_new_tokens(hps: AttrDict) -> List[str]:
  """
  This function is used to extract new tokens from the given dataset.
  Args:
      hps (AttrDict): Hyperparameters.
  Returns:
      List[str]: List of new tokens.
  """
  # load raw dataset
  raw_dataset = load_raw_dataset(hps.data_file)
  # compile patterns
  compiled_pattern = re.compile(NETLIST_PATTERN)
  # parse dataset -> Netlist
  tokens = set()
  for netlist in raw_dataset['train']['netlist_only_gates']:
    # split netlist in chunks of text by categories defined in regex pattern
    tokens = tokens.union(set(compiled_pattern.findall(netlist)))
  # parse dataset -> Patterns
  # TODO: 'patterns'
  # text = [patterns for patterns in raw_dataset['train']['patterns']]
  # # split netlist in chunks of text by categories defined in regex pattern
  # tokens = tokens.union(set(compiled_pattern.findall(netlist)))
  tokens = set([t.strip() for t in tokens])
  tokens_list = []
  for t in tokens:
    if any(t==gate for gate in ["nor", "xor", "xnor", "nand", "buf"]):
      tokens_list.append(AddedToken(t, lstrip=False, rstrip=True, normalized=False, single_word=True))
    elif any(t==gate for gate in ["or", "and", "not"]):
      tokens_list.append(AddedToken(t, lstrip=True, rstrip=False, normalized=True, single_word=False))
    else:
      tokens_list.append(AddedToken(t, lstrip=True, rstrip=False, normalized=True, single_word=False))
  return tokens_list
