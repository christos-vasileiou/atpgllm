"""Stream-filter a Hugging Face dataset and re-upload matching raw records.

Pulls ``$TRAIN_DATASET`` in streaming mode, formats each record into
``prompt``/``completion`` so :func:`RewardFunctionFactory.create_reward_function`
can score it, and keeps records whose reward dict contains
``fault_detected_by_pred_input_vector_acc_logonly == 1``. Matching records are
written back to ``chrivasileiou/asap7-language-of-test-v2`` **with the original
raw schema** (the formatting step mutates the record in place, so we shallow-copy
before formatting and upload the preserved copies).

Memory + speed strategy:

* Stream consumes ``REWARD_BATCH_SIZE`` rows at a time so the working set stays
  bounded regardless of total dataset size.
* Filtered records buffer up to ``SHARD_SIZE`` rows, then are flushed as one
  parquet file ``data/{split}-{idx:05d}.parquet`` via :class:`HfApi`.
* Uploads run on a single background thread so the stream/score loop continues
  while the previous shard is being pushed to the Hub.
* Stale ``data/{split}-*.parquet`` files in the upload repo are atomically
  deleted at the start of each split via a single commit (clean re-run).
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import Dataset, IterableDataset, load_dataset
from huggingface_hub import CommitOperationDelete, HfApi
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from conversation import ConversationExample
from reward_function_factory import RewardFunctionFactory
from tools import TOOLS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

hf_dataset_path_download = os.environ["TRAIN_DATASET"]
hf_dataset_path_upload = "chrivasileiou/asap7-language-of-test-v2"
model_name = os.environ["MODEL"]
config_path = os.environ["SIM_CONFIG"]

# Records pulled per reward_fn call. Small to bound RAM; reward_fn iterates
# sequentially internally so larger batches mainly reduce Python loop overhead.
REWARD_BATCH_SIZE = 64

# Filtered records buffered before a parquet shard is written + uploaded.
SHARD_SIZE = 2000

# Reward dict key that gates whether a record is kept.
TARGET_KEY = "fault_detected_by_pred_input_vector_acc_logonly"

# Splits to process from the source dataset.
SPLITS = ("train", "test")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)
use_tools = bool(
    getattr(tokenizer, "chat_template", None)
    and ("tools" in tokenizer.chat_template or "tool" in tokenizer.chat_template)
)

reward_factory = RewardFunctionFactory(config_path=config_path)
# return_component_dicts=True so we can read TARGET_KEY directly from each result.
reward_fn = reward_factory.create_reward_function(return_component_dicts=True)

api = HfApi()
api.create_repo(repo_id=hf_dataset_path_upload, repo_type="dataset", exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_record(record: dict) -> dict:
    """Add ``prompt``/``completion`` to ``record`` (mutates ``record`` in place).

    ``ConversationExample.from_record`` itself rewrites several keys (e.g.
    ``netlist`` becomes a ``{doc_id, netlist}`` dict, ``input_vector`` /
    ``expected_output`` become compact strings). The caller is responsible for
    snapshotting the raw record before invoking this if the original schema
    must be preserved.
    """
    convo = ConversationExample.from_record(record, use_tools=use_tools)
    prompt_messages = [m for m in convo.messages if m["role"] not in ("assistant", "tool")]
    completion_messages = [m for m in convo.messages if m["role"] in ("assistant", "tool")]

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        tools=TOOLS if use_tools else None,
        add_generation_prompt=True,
    )
    completion = tokenizer.apply_chat_template(
        completion_messages,
        tokenize=False,
        tools=TOOLS if use_tools else None,
        add_generation_prompt=False,
    )
    # Match the slicing used by training_code.format_dataset_for_training so
    # the reward function sees exactly the completion shape it was trained on.
    completion = (
        "<|im_start|>assistant\n<think>"
        + completion.split("</tool_call><|im_end|>\n<|im_start|>assistant\n<think>")[1]
    )
    record["prompt"] = prompt
    record["completion"] = completion
    return record


def wipe_existing_shards(split: str) -> None:
    """Delete ``data/{split}-XXXXX.parquet`` from the upload repo in one commit."""
    try:
        files = api.list_repo_files(repo_id=hf_dataset_path_upload, repo_type="dataset")
    except Exception as exc:  # network / repo-not-found errors should not abort the run
        print(f"WARN: unable to list repo files ({exc}); skipping wipe for {split}.")
        return
    pattern = re.compile(rf"^data/{re.escape(split)}-\d+\.parquet$")
    to_delete = [f for f in files if pattern.match(f)]
    if not to_delete:
        return
    api.create_commit(
        repo_id=hf_dataset_path_upload,
        repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo=p) for p in to_delete],
        commit_message=f"Wipe {len(to_delete)} stale {split} shard(s) before re-filter",
    )
    print(f"[{split}] wiped {len(to_delete)} stale shard(s).")


def upload_shard(records: list, split: str, shard_idx: int) -> None:
    """Serialise ``records`` to a parquet temp file and upload as one shard."""
    if not records:
        return
    ds = Dataset.from_list(records)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ds.to_parquet(tmp_path)
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=f"data/{split}-{shard_idx:05d}.parquet",
            repo_id=hf_dataset_path_upload,
            repo_type="dataset",
            commit_message=f"Filtered {split} shard {shard_idx} ({len(records)} records)",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Stream / score / shard / upload
# ---------------------------------------------------------------------------

# A single background uploader: keeps the next parquet upload off the critical
# path without serialising too many in-flight HTTP commits.
upload_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shard-upload")

try:
    for split in SPLITS:
        print(f"\n=== [{split}] start: source={hf_dataset_path_download} -> upload={hf_dataset_path_upload} ===")
        wipe_existing_shards(split)

        dataset = load_dataset(hf_dataset_path_download, split=split, streaming=True)
        assert isinstance(dataset, IterableDataset), (
            f"Expected streaming IterableDataset, got {type(dataset)}"
        )
        iterator = iter(dataset)

        kept_buffer: list[dict] = []
        pending: list[Future] = []
        shard_idx = 0
        seen = 0
        kept_total = 0

        print(f"[{split}] streaming open; reward_batch={REWARD_BATCH_SIZE} shard_size={SHARD_SIZE} target={TARGET_KEY}==1")
        print(f"[{split}] beginning stream/score/shard loop...", flush=True)
        pbar = tqdm(desc=f"filter {split}", unit="rec", file=sys.stdout)
        try:
            while True:
                # Pull a mini-batch from the stream.
                raw_batch: list[dict] = []
                for _ in range(REWARD_BATCH_SIZE):
                    try:
                        rec = next(iterator)
                    except StopIteration:
                        break
                    raw_batch.append(rec)
                if not raw_batch:
                    break

                # Snapshot raw records BEFORE format_record mutates them.
                # Top-level shallow copy is sufficient: format_record only
                # reassigns top-level keys, so the preserved dicts retain
                # the original string values.
                preserved = [dict(rec) for rec in raw_batch]
                formatted = [format_record(rec) for rec in raw_batch]

                prompts = [r["prompt"] for r in formatted]
                completions = [r["completion"] for r in formatted]
                # GRPOTrainer-style call: extra dataset columns become
                # list-valued kwargs, one entry per row.
                all_keys = set().union(*[set(r.keys()) for r in formatted])
                kwargs = {
                    k: [r.get(k) for r in formatted]
                    for k in all_keys
                    if k not in ("prompt", "completion")
                }
                rewards = reward_fn(prompts, completions, **kwargs)

                for raw_rec, reward in zip(preserved, rewards):
                    if not isinstance(reward, dict):
                        continue
                    if reward.get(TARGET_KEY, 0.0) == 1:
                        kept_buffer.append(raw_rec)
                        kept_total += 1

                seen += len(raw_batch)
                pbar.update(len(raw_batch))
                pbar.set_postfix(kept=kept_total, shards=shard_idx)

                if len(kept_buffer) >= SHARD_SIZE:
                    pending.append(
                        upload_pool.submit(upload_shard, kept_buffer, split, shard_idx)
                    )
                    shard_idx += 1
                    kept_buffer = []

            if kept_buffer:
                pending.append(
                    upload_pool.submit(upload_shard, kept_buffer, split, shard_idx)
                )
                shard_idx += 1
                kept_buffer = []

            # Block until this split's shards are all up before starting the next.
            for fut in pending:
                fut.result()
        finally:
            pbar.close()

        print(f"[{split}] processed={seen} kept={kept_total} shards={shard_idx}")
finally:
    upload_pool.shutdown(wait=True)
