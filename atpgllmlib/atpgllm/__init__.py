# Configuration specifically for deepspeed package. 
# You should install deepspeed package to be useful. 
# Run `pip install deepspeed`

deepspeed_config = {
      "train_batch_size": 1,
      "train_micro_batch_size_per_gpu": 1,
      "steps_per_print": 10,
      "gradient_accumulation_steps": 1,
      "optimizer": {
        "type": "Adam",
        "params": {
          "lr": 1e-4,
          "betas": [0.9, 0.999],
          "eps": 1e-8,
          "weight_decay": 3e-7,
          "torch_adam": False, # Use torch’s implementation of adam instead of DeepSpeed's fused adam implementation
          "adam_w_mode": True  # Apply L2 regularization (also known as AdamW)
        },
      },
      "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
      },
      "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "offload_optimizer": {
          "device": "cpu",
          "pin_memory": True
        },
        "offload_param": {
          "device": "cpu",
          "pin_memory": True
        }
      }
    }
