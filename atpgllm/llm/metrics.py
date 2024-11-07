from torchmetrics import Accuracy
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import TrainerCallback
from pprint import pprint
from rouge_score import rouge_scorer
from collections import defaultdict
from .fault_coverage_calc import validate_generated_text
import multiprocessing
import torch
import sacrebleu

def compute_bleu(references, completions):
  """
  Computes BLEU score using sacrebleu.
  Args:
      references (list of str): Reference texts.
      completions (list of str): Generated texts.
  Returns:
      float: BLEU score.
  """
  # sacrebleu expects list of references and list of completions
  # For single references, wrap references in another list
  sacrebleu_bleu = sacrebleu.corpus_bleu(completions, [references])
  return {"sbleu": sacrebleu_bleu.score}

def compute_rouge(references, completions):
  """
  Computes ROUGE scores using rouge-score.
  Args:
      references (list of str): Reference texts.
      completions (list of str): Generated texts.
  Returns:
      dict: ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
  """
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
  rouge1_scores = []
  rouge2_scores = []
  rougel_scores = []
  
  for ref, completion in zip(references, completions):
    scores = scorer.score(ref, completion)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougel_scores.append(scores['rougeL'].fmeasure)
  
  # Compute average scores
  avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
  avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
  avg_rougel = sum(rougel_scores) / len(rougel_scores) if rougel_scores else 0.0
  
  return {'r1': avg_rouge1, 'r2': avg_rouge2, 'rL': avg_rougel}

def compute_distinct_n(completions, n=2):
  """
  Computes Distinct-n metric.
  Args:
      completions (list of str): Generated texts.
      n (int): The 'n' in Distinct-n.
  Returns:
      float: Distinct-n score.
  """
  unique_ngrams = set()
  total_ngrams = 0
  for completion in completions:
    tokens = completion.split()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngrams = [' '.join(gram) for gram in ngrams]
    unique_ngrams.update(ngrams)
    total_ngrams += len(ngrams)
  return {"dstnct_n": (len(unique_ngrams) / total_ngrams) if total_ngrams > 0 else 0.0}


def compute_repetition_rate(completions):
  """
  Computes Repetition Rate.
  Args:
      completions (list of str): Generated texts.
  Returns:
      float: Average repetition rate across all completions.
  """
  repetition_counts = []
  for completion in completions:
    tokens = completion.split()
    token_counts = defaultdict(int)
    repeats = 0
    for token in tokens:
      token_counts[token] += 1
      if token_counts[token] > 1:
        repeats += 1
    repetition_counts.append(repeats / len(tokens) if len(tokens) > 0 else 0.0)
  return {"rpt_rate": (sum(repetition_counts) / len(repetition_counts)) if len(repetition_counts) > 0 else 0.0}


def compute_fault_coverage(completions, netlists):
  """
  Computes Fault Coverage.
  Args:
      completions (list of str): Generated texts.
  Returns:
      float: Average fault coverage across all completions.
  """
  if len(completions) != len(netlists):
    raise ValueError("The lengths of completions and netlists must be the same.")

  cpu_pool = min(len(completions), 4)
  with multiprocessing.Pool(processes=cpu_pool) as pool:
    covered_fault_counts = pool.starmap(validate_generated_text, zip(completions, netlists))

  fault_coverage = sum(covered_fault_counts) / len(covered_fault_counts)
  return {"fc": fault_coverage}



class ATPGAccuracy:
  def __init__(self, model, tokenizer):
    self.heads = [params for name, params in model.named_parameters() if 'head' in name]
    self.acc_metric = [ Accuracy(task="multiclass", num_classes=self.heads[i].shape[0], top_k=1).cpu() for i in range(len(self.heads)) ]
    self.id_one = tokenizer.convert_tokens_to_ids('1')
    self.id_zero = tokenizer.convert_tokens_to_ids('0')
    self.len_acc_metric = len(self.acc_metric)
    self.tokenizer = tokenizer
    _tokens = tokenizer.tokenize('\n10')
    _tokens = _tokens[1:] if len(_tokens) == 4 else _tokens
    assert len(_tokens) == 3, f"The tokens for calculating accuracy are: {_tokens}"
    self.tokens_to_keep = torch.tensor(tokenizer.convert_tokens_to_ids(_tokens))
    self.tokens_to_map_to_1 = torch.tensor(tokenizer.convert_tokens_to_ids(_tokens[1:]))

  def __call__(self, preds, targets):
    res = dict()
    if isinstance(preds, CausalLMOutputWithPast):
      preds = preds.logits
    preds = preds.cpu()
    targets = targets.cpu()
    if self.len_acc_metric == 1:
      assert isinstance(preds, torch.Tensor)
      self.acc_metric[0] = self.acc_metric[0].cpu()

      ids = preds.topk(1).indices
      ids = ids.reshape(-1, ids.shape[1])
      bleu_averages, rouge_averages = self._blue_n_rouge_score(ids, targets)
      
      res.update({'accuracy': self.acc_metric[0](preds.reshape(-1, preds.shape[-1]), targets.reshape(-1)).item()})
      res.update({'patterns_accuracy': self._patterns_accuracy(preds, targets)})
      res.update({'blue': bleu_averages})
      res.update({'rouge': rouge_averages})
    else:
      # TODO: Needs to be tested and debugged...
      assert len(preds) == self.len_acc_metric, f"Length of prediction is different of the length of heads: len of preds: {len(preds)} and number of heads: {self.len_acc_metric}"
      assert isinstance(preds, list) or isinstance(preds, tuple), f"Type of preds should be tuple or list" 
      preds = [pred.reshape(-1, pred.shape[-1]) for pred in preds]
      targets = targets.reshape(-1)
      self.acc_metric = [accm.cpu() for accm in self.acc_metric]
      bleu_averages, rouge_averages = self._blue_n_rouge_score(ids, targets)

      res.update({'accuracy': (self.acc_metric[i](preds[i], targets).item() for i in range(self.len_acc_metric))}) # Accuracy for all token ids
      res.update({'patterns_accuracy': self._patterns_accuracy(preds, targets)})
      res.update({'bleu': bleu_averages})
      res.update({'rouge': rouge_averages})
    return res

  def _patterns_accuracy(self, preds, targets):
    ids = preds.topk(1).indices
    ids = ids.reshape(-1, ids.shape[1])
    b = preds.shape[0]
    valid_patterns_from_completion = self._get_patterns_from_completion(ids)
    valid_patterns_from_targets = self._get_patterns_from_completion(targets)
    valid_patterns_from_completion = valid_patterns_from_completion[valid_patterns_from_completion != 0]
    valid_patterns_from_targets = valid_patterns_from_targets[valid_patterns_from_targets != 0]
    min_size = min(valid_patterns_from_completion.size(0), valid_patterns_from_targets.size(0))
    miss_preds = abs(valid_patterns_from_completion.size(0) - valid_patterns_from_targets.size(0))
    valid_patterns_from_completion = valid_patterns_from_completion[:min_size]
    valid_patterns_from_targets = valid_patterns_from_targets[:min_size]
    correct_predictions = (valid_patterns_from_completion == valid_patterns_from_targets).sum()
    patterns_accuracy = ((correct_predictions - miss_preds)/min_size).item()
    return patterns_accuracy
    
  def _get_patterns_from_completion(self, x):
    self.tokens_to_keep = self.tokens_to_keep.cpu()
    self.tokens_to_map_to_1 = self.tokens_to_map_to_1.cpu()
    mask = (x[..., None] == self.tokens_to_keep).any(-1)
    # Find the start index of the last continuous group of Trues
    # Convert the mask to int to utilize cumsum and diff methods
    diff = torch.diff(mask.int(), prepend=torch.tensor([[0]]).cpu())
    # The following variables (`group_starts`, `group_ends`) consist of 2 tensors.
    # 2. `group_starts`: tuple( tensor(0, 0, ..., 1, 1, ..., batch_size-1, batch_size-1, ...), 
    #                           tensor(2, 14, ..., 1, 5, ..., 10, 30))
    # 2. `group_ends`: tuple( tensor(0, 0, ..., 1, 1, ..., batch_size-1, batch_size-1, ...), 
    #                         tensor(6, 25, ..., 4, 10, ..., 15, 40))
    group_starts = torch.where(diff == 1)  # Start of each group of Trues
    group_ends = torch.where(diff == -1)   # End of each group of Trues
    # Collect the range of 0, 1, ..., batch_size-1
    unique_first_indices = torch.unique(group_starts[0])

    # Create a result tensor based on the found indices
    # Would be similar to iterate over the range(batch_size). 
    # However, since the mask is used there is a possibility that certain batches will not contain 1s and 0s and \n
    valid_patterns = torch.zeros_like(x)
    for idx in unique_first_indices:
      # Get indices where first-axis index matches
      idx_mask = group_starts[0] == idx
      # Extract corresponding starts and ends for this index
      specific_starts = group_starts[1][idx_mask]
      specific_ends = group_ends[1][idx_mask]

      if len(specific_ends) == 0 and mask[idx, -1] == True:
        # Handle case where the last element is True and no False after it
        specific_ends = torch.tensor([len(mask[idx])], dtype=torch.long)
      if len(specific_starts) > 0 and len(specific_ends) > 0:
        # Handle edge case if the sequence ends in Trues
        if specific_ends[-1] <= specific_starts[-1]:
          specific_ends = torch.cat((specific_ends, torch.tensor([len(mask[idx])], dtype=torch.long)))
        # Get the indices for the last group
        last_group_start = specific_starts[-1]
        last_group_end = specific_ends[-1]
      else:
        # No True group found
        last_group_start = 0
        last_group_end = 0

      # Create a result tensor based on the found indices
      _valid_patterns = torch.zeros_like(x[idx])
      if last_group_start < last_group_end:  # There is a valid group
        _valid_patterns[last_group_start:last_group_end] = x[idx][last_group_start:last_group_end]
      valid_patterns[idx] = _valid_patterns
      # _valid_patterns = self.tokenizer.decode(_valid_patterns[idx])
      
    return valid_patterns
  
  def transform_to_binary(self, tensor, tokens_to_one):
    mask = torch.zeros_like(tensor)
    for token_id in tokens_to_one:
        mask |= (tensor == token_id).int()
    return mask.int()

  def _blue_n_rouge_score(self, batch_preds, batch_references):
    from rouge import Rouge
    from nltk.translate.bleu_score import corpus_bleu
    rouge = Rouge()
    # Initialize accumulators
    rouge_scores = {
        'rouge-1': {'f': [], 'p': [], 'r': []},
        'rouge-2': {'f': [], 'p': [], 'r': []},
        'rouge-l': {'f': [], 'p': [], 'r': []}
    }
    bleu_scores = []
    # Iterate over batches axis
    for preds, references in zip(batch_preds, batch_references):
      preds = [self.tokenizer.decode(preds)]
      references = [[self.tokenizer.decode(references)]]
      tokenized_prediction = [p.split() for p in preds]
      tokenized_reference = [[r.split() for r in references[0]]]

      bleu_score = corpus_bleu(tokenized_reference, tokenized_prediction)
      bleu_scores.append(bleu_score)
      scores = rouge.get_scores(preds, references[0], avg=True)
      # Accumulate scores
      for key in rouge_scores.keys():
        rouge_scores[key]['f'].append(scores[key]['f'])
        rouge_scores[key]['p'].append(scores[key]['p'])
        rouge_scores[key]['r'].append(scores[key]['r'])
    rouge_averages = {metric: {submetric: sum(values)/len(values) for submetric, values in series.items()} for metric, series in rouge_scores.items()}
    bleu_averages = sum(bleu_scores)/len(bleu_scores)
    return bleu_averages, rouge_averages


class MetricsCallback(TrainerCallback):
  def __init__(self, accuracy_calculator):
    super().__init__()
    self.accuracy_calculator = accuracy_calculator

  def on_evaluate(self, args, state, control, **kwargs):
    """Called at the end of an evaluation phase."""
    logits = kwargs['logits']
    labels = kwargs['labels']
    metrics = self.accuracy_calculator(logits, labels)
    for key, value in metrics.items():
      print(f"{key}: {value}")

class MetricsLoggingCallback(TrainerCallback):
  def __init__(self, accuracy_calculator, logging_steps):
    super().__init__()
    self.accuracy_calculator = accuracy_calculator
    self.logging_steps = logging_steps

  def on_step_end(self, args, state, control, **kwargs):
    """Called at the end of each training step."""
    if state.global_step % self.logging_steps == 0:
      # print(state.global_step, args)
      # Assuming that logits and labels are available here,
      # this might need adjustment based on how your trainer works
      logits = kwargs.get('logits')
      labels = kwargs.get('labels')
      # print(logits)
      # print(labels)
      # exit()
      if logits is not None and labels is not None:
        metrics = self.accuracy_calculator(logits.detach(), labels.detach())
        for key, value in metrics.items():
          print(f"Step {state.global_step} - {key}: {value}")
