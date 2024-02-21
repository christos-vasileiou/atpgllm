from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from atpg.utils import collate_fn_dict, create_data_loader, convert_bytes, seconds_to_dhms, setup_logging, save_checkpoint, model_size_in_bytes, load_checkpoint, penalize, compute_metrics, move_optimizer_to_device
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch
import deepspeed
import os
import time
import io
import re  

timing_template = ("[{epoch}/{total_epochs}]: "
            "GPU ID: {local_rank} | "
            "Time Elapsed: {time_elapsed} | "
            "Train Acc % (>.5, topk): ({train_accuracy:.4f}, {train_accuracy_topk:.4f}) | "
            "Train Prec % (>.5, topk): ({train_precision:.4f}, {train_precision_topk:.4f}) | "
            "Valid Acc [% (>.5, topk), N (norm, topk)]: [({accuracy:.4f}, {accuracy_topk:.4f}), ({non_norm_accuracy}, {non_norm_accuracy_topk}) /{total_samples}] | "
            "Valid Prec % (>.5, topk): ({precision:.4f}, {precision_topk:.4f}) | "
            "Valid F1-Score M (topk): {f1_score_macro:.4f} | "
            "Train Loss: {train_loss:.4f} | "
            "Val Loss: {val_loss:.4f}")

testing_template = ("GPU ID: {local_rank}, "
                    "Test Time Elapsed: {elapsed_time}, "
		    "Test Acc [%, N]: [({accuracy:.4f}, {accuracy_topk:.4f}), {non_norm_accuracy}/{test_labels_size}], "
		    "Test Prec %: {precision:.4f}, "
		    "Test Loss: {avg_test_loss:.4f}")

epoch_data = {"epoch": None,
              "total_epochs": None,
              "time_elapsed": None,
              "local_rank": None,
              "train_accuracy": None,
              "train_precision": None,
              "accuracy": None,
              "non_norm_accuracy": None,
              "total_samples": None,
              "precision": None,
              #"f1_score_micro": None,
              "f1_score_macro": None,
              "train_loss": None,
              "val_loss": None}

test_data = {"local_rank": None,
             "elapsed_time": None,
	     "accuracy": None,
	     "accuracy_topk": None,
	     "non_norm_accuracy": None,
	     "test_labels_size": None,
	     "precision": None,
	     "avg_test_loss": None}


def train_graph(epochs, model, dataset, lr, batch_size, labels_filtering=True, reduction='sum', parallel=False, deepspeed_config=None, fp16=False, stored_checkpoint=None, graph_collate=None, pos_weight=None, train_type=None, free_gpu_id=0, args=None):
  #if train_type is None:
  #  raise ValueError(f"train_type: has to set to 'multilabel' or 'multiclass'")
  if reduction not in ['sum', 'mean', 'none']:
    raise ValueError(f"reduction: has set to wrong value.")
  
  if train_type == 'multilabel':
    criterion = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)
  elif train_type == 'multiclass':
    criterion = nn.CrossEntropyLoss(reduction=reduction)
  else:
    criterion_sigmoid = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)
    criterion_softmax = nn.CrossEntropyLoss(reduction=reduction)
    criterion = [criterion_sigmoid, criterion_softmax]

  optimizer   = optim.AdamW(model.parameters(), lr=lr)
  start_epoch = 0
  if stored_checkpoint is not None:
    # preprocess the checkpoint string
    split_path = stored_checkpoint.split('/') if '/' in stored_checkpoint else stored_checkpoint
    date_time  = split_path[0] if isinstance(split_path, list) else split_path
    local_rank = split_path[1].split('_')[0]
    filename   = '_'.join('_'.join(split_path[1:]).split('_')[1:])
    print(f"{date_time}, {local_rank}, {filename}")
    # load model
    model, optimizer, start_epoch, hyperparameter = load_checkpoint(date_time=date_time, local_rank=local_rank, filename=filename, model=None, optimizer=optimizer)
  start_epoch += 1
  local_rank = free_gpu_id
  #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
  model = model.to(torch.bfloat16) if fp16==True else model
  if parallel==True:
    if deepspeed_config is not None:
      model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=deepspeed_config)
      local_rank = model_engine.local_rank
       
      device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
      logger = setup_logging(local_rank)
      logger.info(f"{device}\n{model_engine}\n{optimizer}\nModel parameters: {sum(p.numel() for p in model_engine.parameters())}, \nModel size: {convert_bytes(model_size_in_bytes(model_engine))}\n{args}")
      print(f"{device}\n{model_engine}\n{optimizer}\nModel parameters: {sum(p.numel() for p in model_engine.parameters())}, \nModel size: {convert_bytes(model_size_in_bytes(model_engine))}\n{args}")
      
    elif deepspeed_config is None:
      #local_rank = int(os.environ['LOCAL_RANK']) 
      local_rank = dist.get_rank()
      print(f"local_rank: {local_rank}")
      logger     = setup_logging(local_rank)
      torch.manual_seed(27)
      torch.cuda.set_device(local_rank)
      # dist.init_process_group(backend="nccl", rank=local_rank, world_size=dist.get_world_size())
      device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
      model  = model.to(device)
      model  = DDP(model, device_ids=[local_rank], output_device=local_rank)
      move_optimizer_to_device(optimizer, device)
      logger.info(device)
      logger.info(f"{model}\n{optimizer}\nModel parameters: {sum(p.numel() for p in model.parameters())}, \nModel size: {convert_bytes(model_size_in_bytes(model))}\n{args}")

      if local_rank == 0:
        print(device)
        print(f"{model}\n{optimizer}\nModel parameters: {sum(p.numel() for p in model.parameters())}, \nModel size: {convert_bytes(model_size_in_bytes(model))}\n{args}")
  else:
    device = torch.device(f'cuda:{free_gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(local_rank)
    model  = model.to(device)
    #print(f"device:{device}\n{model}\n{optimizer}\n{args}")
    print(f"device: {device}\n{model}\n{optimizer}\nModel parameters: {sum(p.numel() for p in model.parameters())}, \nModel size: {convert_bytes(model_size_in_bytes(model))}\n{args}")
  if train_type == 'multilabel' or train_type == 'multiclass':
    criterion   = criterion.to(device)
  else:
    criterion[0] = criterion[0].to(device)
    criterion[1] = criterion[1].to(device)
  # Prepare data_loaders
  train_test  = dataset['train'].shuffle(seed=27).train_test_split(test_size=.3)
  train_val   = train_test['train'].train_test_split(test_size=.15)
  use_sampler = parallel and deepspeed_config is None
  shuffle     = not use_sampler
  
  if use_sampler:
    training_sampler   = DistributedSampler(train_val['train'], rank=local_rank)
    validation_sampler = DistributedSampler(train_val['test'],  rank=local_rank)
    testing_sampler    = DistributedSampler(train_test['test'], rank=local_rank)
  else:
    training_sampler, validation_sampler, testing_sampler = None, None, None
  collate_fn        = collate_fn_dict if graph_collate is None else graph_collate
  training_loader   = create_data_loader(train_val['train'], batch_size, shuffle, collate_fn, training_sampler,   num_workers=0)

  train_loss_to_print = []
  val_loss_to_print   = []
  epoch_data.update({'total_epochs': epochs+start_epoch-1, "local_rank": local_rank})
  start_time = time.time()
  llambda    = 3
  vllambda   = 3
  loss_ratio = None
  for epoch in range(start_epoch, start_epoch + epochs):
    # Training
    train_loss = []
    train_outputs = torch.tensor([])
    train_labels  = torch.tensor([])
    model.train()
    for data in training_loader:
      nodes      = data['nodes'].to(device, dtype=torch.bfloat16)
      edge_index = data['edge_index'] #.to(device, dtype=torch.long)
      labels     = data['labels'].to(device, dtype=torch.bfloat16)
      
      #print(f"nodes; {type(nodes)}, edge_index: {type(edge_index)}, labels: {type(labels)}")
      #print(f"nodes: {type(nodes[0])}, edge_index: {type(edge_index[0])}, labels: {type(labels[0])}")
      #print(f"nodes:\n{nodes[0]}\nedge_index:\n{edge_index[0]}\nlabels:\n{labels[0]}")
      # Forward pass
      outputs       = torch.tensor([]).to(device)
      for node, e_idx in zip(nodes, edge_index):
        #print(node.shape, type(node), e_idx.shape, type(e_idx))
        o = model((node, e_idx), device=device) if deepspeed_config is None else model_engine((node, e_idx), device=device)
        outputs = torch.concat([outputs, o.unsqueeze(0)], dim=0)
      #print(f"nodes: {nodes.shape}, outputs: {outputs.shape}, labels: {labels.shape}")
      
      #print(nodes[0].dtype)
      #print(outputs[0].dtype)
      #print(labels[0].dtype)
      # Zero Gradients
      optimizer.zero_grad()

      train_outputs = torch.concat([train_outputs, torch.sigmoid(outputs).cpu().detach().view(-1) ], dim=0)
      train_labels  = torch.concat([train_labels,  labels.cpu().detach().view(-1)], dim=0)
      #print(outputs.shape, train_outputs.shape, labels.shape, train_labels.shape)
      # Loss Calculation
      #print(f"outputs: {outputs.shape}, labels: {labels.shape}")
      loss, loss_ratio = penalize(outputs, labels, criterion, labels_filtering, llambda, vllambda, train_type, epoch, loss_ratio)
      train_loss.append(loss.item())

      # Gradients Calculation
      if deepspeed_config:
        model_engine.backward(loss)
      else:
        loss.backward()

      # Update weights
      optimizer.step()
      
    train_accuracy, train_non_norm_accuracy, train_precision, train_accuracy_topk, train_non_norm_accuracy_topk, train_precision_topk, train_f1_score_macro = compute_metrics(train_outputs, train_labels, None)
    avg_train_loss = train_loss.mean() if isinstance(train_loss, torch.Tensor) else torch.tensor(train_loss).mean()
    train_loss_to_print.append(avg_train_loss)

    # Validation
    validation_loader = create_data_loader(train_val['test'],  batch_size, shuffle, collate_fn, validation_sampler, num_workers=0)
    model.eval()
    with torch.no_grad():
      val_outputs = torch.tensor([])
      val_labels  = torch.tensor([])
      val_loss    = torch.tensor([])
      for data in training_loader:
        nodes      = data['nodes'].to(device, dtype=torch.bfloat16)
        edge_index = data['edge_index'] #.to(device, dtype=torch.long)
        labels     = data['labels'].to(device, dtype=torch.bfloat16)
        
        #print(f"nodes; type(nodes), edge_index: type(edge_index), labels: type(labels)")
        # Forward pass
        outputs       = torch.tensor([]).to(device)
        for node, e_idx in zip(nodes, edge_index):
          #print(node.shape, type(node), e_idx.shape, type(e_idx))
          o = model((node, e_idx), device=device) if deepspeed_config is None else model_engine((node, e_idx), device=device)
          outputs = torch.concat([outputs, o.unsqueeze(0)], dim=0)

        # Loss Calculation
        loss, loss_ratio = penalize(outputs, labels, criterion, labels_filtering, llambda, vllambda, train_type, epoch, loss_ratio)
        
        val_loss = torch.cat([val_loss, loss.cpu().detach().unsqueeze(0)])

        # Collect outputs and labels
        val_outputs = torch.concat([val_outputs, torch.sigmoid(outputs).cpu().detach().view(-1) ], dim=0)
        val_labels = torch.concat([val_labels, labels.cpu().detach().view(-1)], dim=0)
        
      avg_val_loss = val_loss.mean()
      val_loss_to_print.append(avg_val_loss)

    accuracy, non_norm_accuracy, precision, accuracy_topk, non_norm_accuracy_topk, precision_topk, f1_score_macro = compute_metrics(val_outputs, val_labels, None)
    
    elapsed_time      = time.time() - start_time
    epoch_data.update({'epoch': epoch, 
                       'time_elapsed': seconds_to_dhms(elapsed_time),
                       'train_accuracy': train_accuracy if train_accuracy is not None else 0.0,
                       'train_precision': train_precision if train_precision is not None else 0.0,
                       'train_accuracy_topk': train_accuracy_topk if train_accuracy_topk is not None else 0.0,
                       'train_precision_topk': train_precision_topk if train_precision_topk is not None else 0.0,
                       'accuracy': accuracy if accuracy is not None else 0.0,
                       'non_norm_accuracy': non_norm_accuracy if non_norm_accuracy is not None else 0.0,
                       'accuracy_topk': accuracy_topk if accuracy_topk is not None else 0.0,
                       'non_norm_accuracy_topk': non_norm_accuracy_topk if non_norm_accuracy_topk is not None else 0.0,
                       'total_samples': val_labels.size(0),
                       'precision': precision if precision is not None else 0.0,
                       'precision_topk': precision_topk if precision_topk is not None else 0.0,
                       'f1_score_macro': f1_score_macro if f1_score_macro is not None else 0.0,
                       'train_loss': avg_train_loss.item() if avg_train_loss is not None else 0.0,
                       'val_loss': avg_val_loss.item() if avg_val_loss is not None else 0.0
    })
    #[{strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]:  
    printing_info = timing_template.format(**epoch_data)
    logger.info(printing_info)
    #if (epoch==start_epoch or epoch%5==0):
    print(printing_info)
    #print(f'Epoch: [{epoch}/{epochs}], Accuracy: {accuracy}, f1-m: {f1_score_micro}, f1-M: {f1_score_macro}, Training Loss: {avg_train_loss.item()}, Validation Loss: {avg_val_loss.item()}')
    
    if epoch!=start_epoch and epoch%5!=0:
      continue
    
    # Testing
    test_start_time = time.time()
    printing_info = f"GPU ID {local_rank}, Testing started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}"
    #if epoch == start_epoch + epochs - 1: 
    logger.info(printing_info)
    print(printing_info)
    testing_loader    = create_data_loader(train_test['test'], batch_size, shuffle, collate_fn, testing_sampler,    num_workers=0)
    model.eval()
    with torch.no_grad():
      test_outputs = torch.tensor([])
      test_labels  = torch.tensor([])
      test_loss    = torch.tensor([])
      for data in testing_loader:
        nodes      = data['nodes'].to(device, dtype=torch.bfloat16)
        edge_index = data['edge_index'] #.to(device, dtype=torch.long)
        labels     = data['labels'].to(device, dtype=torch.bfloat16)

        # Forward pass
        outputs       = torch.tensor([]).to(device)
        for node, e_idx in zip(nodes, edge_index):
          #print(node.shape, type(node), e_idx.shape, type(e_idx))
          o = model((node, e_idx), device=device) if deepspeed_config is None else model_engine((node, e_idx), device=device)
          outputs = torch.concat([outputs, o.unsqueeze(0)], dim=0)
 
        # Loss Calculation
        loss, loss_ratio = penalize(outputs, labels, criterion, labels_filtering, llambda, vllambda, train_type, epoch, loss_ratio)

        test_loss    = torch.cat([test_loss, loss.cpu().detach().unsqueeze(0)])
        # Collect outputs and labels
        test_outputs = torch.concat([test_outputs, torch.sigmoid(outputs).cpu().detach().view(-1)], dim=0)
        test_labels  = torch.concat([test_labels, labels.cpu().detach().view(-1)], dim=0)
        
      avg_test_loss = test_loss.mean()
   
    accuracy, non_norm_accuracy, precision, accuracy_topk, non_norm_accuracy_topk, precision_topk, f1_score_macro = compute_metrics(test_outputs, test_labels, None)

    elapsed_time  = time.time() - test_start_time
    elapsed_time  = seconds_to_dhms(elapsed_time)
    
    test_data.update({"local_rank": local_rank,
                      "elapsed_time": elapsed_time,
                      "accuracy": accuracy if accuracy is not None else 0.0,
                      "accuracy_topk": accuracy_topk if accuracy_topk is not None else 0.0,
                      "non_norm_accuracy": non_norm_accuracy if non_norm_accuracy is not None else 0.0,
                      "test_labels_size": test_labels.size(0),
                      "precision": precision if precision is not None else 0.0,
                      "avg_test_loss": avg_test_loss.item()})
    
    printing_info = testing_template.format(**test_data)
    print(printing_info)
    if epoch == start_epoch + epochs - 1: 
      logger.info(printing_info)
  # Let other process to know which among them is the best in order to save info of the best training.
  #avg_test_loss = torch.tensor([avg_test_loss.item()], device=device).float()
  #all_test_losses = [torch.tensor([float('inf')], device=device) for _ in range(dist.get_world_size())]
  #dist.all_gather(all_test_losses, torch.tensor([avg_test_loss]))
  
  # Determine the rank with the minimum tets loss
  #min_loss_value, min_loss_rank = torch.tensor(all_test_losses).min(dim=0)
  #print(f"{min_loss_value} {min_loss_rank}")

  if deepspeed_config is not None:
    model_to_save = model_engine.module
  elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model_to_save = model.module
  else:
    model_to_save = model
  
  return model_to_save, optimizer, epochs, local_rank, train_loss_to_print, val_loss_to_print 
 











def train_gen_task(epochs, model, dataset, lr, batch_size, labels_filtering=True, reduction='sum', parallel=False, deepspeed_config=None, fp16=False, stored_checkpoint=None, graph_collate=None, pos_weight=None, train_type=None, free_gpu_id=0):
  if reduction not in ['sum', 'mean', 'none']:
    raise ValueError(f"reduction: has set to wrong value.")
  
  criterion   = nn.CrossEntropyLoss(reduction=reduction)
  optimizer   = optim.AdamW(model.parameters(), lr=lr)
  start_epoch = 1
  if stored_checkpoint is not None:
    # preprocess the checkpoint string
    split_path = stored_checkpoint.split('/') if '/' in stored_checkpoint else stored_checkpoint
    date_time  = split_path[0] if isinstance(split_path, list) else split_path
    local_rank = split_path[1].split('_')[0]
    filename   = '_'.join('_'.join(split_path[1:]).split('_')[1:])
    print(f"{date_time}, {local_rank}, {filename}")
    # load model
    model, optimizer, start_epoch, hyperparameter = load_checkpoint(date_time=date_time, local_rank=local_rank, filename=filename, model=model, optimizer=optimizer)
  local_rank = 0
  #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
  model = model.to(torch.bfloat16) if fp16==True else model
  if parallel==True:
    if deepspeed_config is not None:
      model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=deepspeed_config)
      logger = setup_logging(local_rank)
      logger.info(f"\n{model_engine}")
      logger.info(f"\n{optimizer}")
    elif deepspeed_config is None:
      local_rank = dist.get_rank()
      logger     = setup_logging(local_rank)
      torch.manual_seed(27)
      torch.cuda.set_device(local_rank)
      # dist.init_process_group(backend="nccl", rank=local_rank, world_size=dist.get_world_size())
      device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
      model  = model.to(device)
      model  = DDP(model, device_ids=[local_rank], output_device=local_rank)
      move_optimizer_to_device(optimizer, device)
      logger.info(device)
      logger.info(f"{model}\nModel parameters: {sum(p.numel() for p in model.parameters())}, \nModel size: {convert_bytes(model_size_in_bytes(model))}")

      if local_rank == 0:
        print(device)
        print(model, '\nModel parameters:', sum(p.numel() for p in model.parameters()), '\nModel size:', convert_bytes(model_size_in_bytes(model)))
  else:
    device = torch.device(f'cuda:{free_gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(local_rank)
    model  = model.to(device)
    print(model, '\nModel parameters:', sum(p.numel() for p in model.parameters()), '\nModel size:', convert_bytes(model_size_in_bytes(model)))
  criterion   = criterion.to(device)
  # Prepare data_loaders
  train_test  = dataset['train'].shuffle(seed=27).train_test_split(test_size=.3)
  train_val   = train_test['train'].train_test_split(test_size=.15)
  use_sampler = parallel and deepspeed_config is None
  shuffle     = not use_sampler
  
  if use_sampler:
    training_sampler   = DistributedSampler(train_val['train'], rank=local_rank)
    validation_sampler = DistributedSampler(train_val['test'],  rank=local_rank)
    testing_sampler    = DistributedSampler(train_test['test'], rank=local_rank)
  else:
    training_sampler, validation_sampler, testing_sampler = None, None, None
  collate_fn        = collate_fn_dict if graph_collate is None else graph_collate
  training_loader   = create_data_loader(train_val['train'], batch_size, shuffle, collate_fn, training_sampler,   num_workers=4)
  validation_loader = create_data_loader(train_val['test'],  batch_size, shuffle, collate_fn, validation_sampler, num_workers=4)
  testing_loader    = create_data_loader(train_test['test'], batch_size, shuffle, collate_fn, testing_sampler,    num_workers=4)

  train_loss_to_print = []
  val_loss_to_print   = []
  epoch_data.update({'total_epochs': epochs, "local_rank": local_rank})
  start_time = time.time()
  llambda    = 3
  vllambda   = 3
  for epoch in range(start_epoch, start_epoch + epochs):
    # Training
    train_loss = []
    model.train()
    for data in training_loader:
      ids            = data['input_ids'].to(device, dtype=torch.long)
      mask           = data['attention_mask'].to(device, dtype=torch.long)
      token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
      labels         = data['labels'].to(device, dtype=torch.float)
      
      assert ids.shape == mask.shape and ids.shape == token_type_ids.shape
      # Forward pass
      outputs = model(ids, mask, token_type_ids) if deepspeed_config is None else model_engine(ids, mask, token_type_ids)
 
