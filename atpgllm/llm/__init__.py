from .collate import *
from .fine_tuning import *
from .models import *
from .sft import *
from .tokenizer import *

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
		    "Test Prec % (>.5, topk): ({precision:.4f}, {precision_topk:.4f}), "
		    "Test Loss: {avg_test_loss:.4f}")

epoch_data = {"epoch": None,
              "total_epochs": None,
              "local_rank": None,
              "time_elapsed": None,
              "train_accuracy": None,
              "train_accuracy_topk": None,
              "train_precision": None,
              "train_precision_topk": None,
              "accuracy": None,
              "accuracy_topk": None,
              "non_norm_accuracy": None,
              "non_norm_accuracy_topk": None,
              "total_samples": None,
              "precision": None,
              "precision_topk": None,
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
	     "precision_topk": None,
	     "avg_test_loss": None}

