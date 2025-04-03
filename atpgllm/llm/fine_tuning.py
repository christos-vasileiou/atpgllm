import torch


def get_dec_ids_and_mask(targets, tokenizer, is_causal, device):
  # shifts targets forwards to get decoder_input. The opposite of causal
  dec_input, dec_mask = None, None
  if not is_causal:
    dec_input = targets.clone().detach()
    dec_input = torch.roll(dec_input, shifts=1, dims=1)
    dec_input[:, 0] = tokenizer.cls_token_id # assign the last id of the tokenizer which is mapped to cls_token: '<s>'

    # also convert all -100 to pad token id
    dec_input = dec_input.masked_fill(dec_input == -100, tokenizer.pad_token_id).to(device)

    # make decoder input mask
    dec_mask = torch.ones_like(dec_input)
    dec_mask = dec_mask.masked_fill(dec_input == tokenizer.pad_token_id, 0).to(device)
  return dec_input, dec_mask
