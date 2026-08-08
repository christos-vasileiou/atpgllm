#%%
from datasets import load_dataset
from transformers import AutoTokenizer
from atpgllm.training.conversation import ConversationExample
from atpgllm.training.tools import TOOLS
import os
import re
import gc
import itertools
import pandas as pd
import multiprocessing
import plotly.graph_objects as go
from multiprocessing.pool import ThreadPool
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

#%%
# --- 1. Setup and Loading ---
model_name = os.environ.get("MODEL", "meta-llama/Llama-2-7b-chat-hf")
dataset_path = "chrivasileiou/asap7-language-of-test"

print(f"Loading tokenizer for model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading dataset from path: {dataset_path}")
data = load_dataset(dataset_path, split="train", streaming=True)

# Compile regex once. re objects are thread-safe.
# Matches Verilog gate/module instantiations: <cell_type> <instance_name> ( ... )
# - instance_name may be a normal identifier or an escaped identifier (starts with '\')
# - escaped identifiers can contain '/', '*', '[', ']', etc. up to the first whitespace
GATE_INGREDIENT = r"(?m)^(?!\s*(?:module|endmodule|input|output|inout|wire|wand|wor|tri|tri0|tri1|trireg|reg|logic|assign|parameter|localparam|specify|endspecify|genvar|generate|endgenerate|always|always_ff|always_comb|always_latch|initial|begin|end|if|else|case|endcase|for|while|repeat|forever)\b)\s*" \
                  r"[^\s(]+\s+(?:\\\S+|[A-Za-z_][A-Za-z0-9_$]*)\s*\("
regex_instances = re.compile(GATE_INGREDIENT)

use_tools = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template and ('tools' in tokenizer.chat_template or 'tool' in tokenizer.chat_template)

#%%

# --- 2. Highly Optimized Parallel Processing ---
def process_chunk(chunk):
    """Processes a raw chunk of records in a single thread."""
    gate_counts = []
    batch_prompt_messages = []
    batch_full_messages = []
    netlists = []
    for record in chunk:
        convo = ConversationExample.from_record(record, use_tools=use_tools)
        
        # Isolate prompt messages
        prompt_messages = [m for m in convo.messages if m["role"] not in ("assistant", "tool")]
        
        batch_prompt_messages.append(prompt_messages)
        batch_full_messages.append(convo.messages)
        
        # EXTREME OPTIMIZATION: len(findall) executes entirely in C, 
        # vastly outperforming Python generator expressions for counting.
        gate_counts.append(len(regex_instances.findall(record["netlist"]["netlist"])))
        netlists.append(record["netlist"]["netlist"])
    # EXTREME OPTIMIZATION: Batched tokenization with padding=False
    # The Rust tokenizer backend releases the Python GIL here, allowing true parallel CPU usage!
    prompt_ids_batch = tokenizer.apply_chat_template(
        batch_prompt_messages, 
        tokenize=True, 
        padding=False,  
        tools=TOOLS if use_tools else None,
        add_generation_prompt=True,
    )
    
    full_convo_ids_batch = tokenizer.apply_chat_template(
        batch_full_messages,
        tokenize=True,
        padding=False,
        tools=TOOLS if use_tools else None,
    )

    # Hash the netlist strings to safely count uniques without exploding main thread RAM
    netlist_hashes = [hash(n) for n in netlists]
    
    # Calculate lengths
    prompt_lengths = [len(ids) for ids in prompt_ids_batch]
    full_lengths = [len(ids) for ids in full_convo_ids_batch]

    # DEBUG: print the netlist that has the unlogical long prompt
    for idx, (gcnt, pl, fl, netlist) in enumerate(zip(gate_counts, prompt_lengths, full_lengths, netlists)):
        if (gcnt == 1 and pl > 2048) or (gcnt == 11 and pl > 4096) or (gcnt == 16 and pl > 4096) or (gcnt == 38 and pl > 6144) or (gcnt == 49 and pl > 8192) or (gcnt == 52 and pl > 8192):
            print(f"Prompt length: {pl} for gate count: {gcnt} at index: {idx}")
            print(f"Netlist:\n{netlist}")
            print("--------------------------------")
    return gate_counts, prompt_lengths, full_lengths, netlist_hashes

def chunked_iterable(iterable, size):
    """Yields chunks of the streaming dataset without loading everything into memory."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

# %%

# Execute mapping over the streaming dataset
print("Processing dataset using multi-threading...")

# OPTIMIZATION: For millions of insertions, a dictionary of lists is vastly 
# faster and more memory-efficient for Pandas to process than a list of dicts.
stats_dict = {
    "gate_count": [], 
    "prompt_token_length": [], 
    "full_convo_token_length": [],
    "netlist_hash": []
}

chunk_size = 500
# Use available CPU cores, keeping one free to prevent OS locking
num_threads = max(1, multiprocessing.cpu_count() - 1) 

with ThreadPool(processes=num_threads) as pool:
    chunks = chunked_iterable(data, chunk_size)
    
    # imap_unordered lazily evaluates chunks only when a thread is free (memory safe!)
    # tqdm updates per chunk of 2,000 records
    task_stream = pool.imap_unordered(process_chunk, chunks)
    
    for gc_list, pl_list, fl_list, hash_list in tqdm(task_stream, desc=f"Processing Batches of {chunk_size}"):
        # EXTREME OPTIMIZATION: .extend() acts in C-level bulk. 
        # This replaces millions of individual .append() calls in a Python loop.
        stats_dict["gate_count"].extend(gc_list)
        stats_dict["prompt_token_length"].extend(pl_list)
        stats_dict["full_convo_token_length"].extend(fl_list)
        stats_dict["netlist_hash"].extend(hash_list)
#%%

# --- 3. Data Aggregation & Plotly Visualization ---
print("Calculating statistics and generating plots...")

# Convert to Pandas DataFrame directly from the dict of lists
df = pd.DataFrame(stats_dict)

# Free the massive dictionary from memory now that Pandas has it
del stats_dict
gc.collect()

# Group by gate count and calculate pre-aggregated box plot stats to avoid browser freeze
grouped = df.groupby("gate_count").agg(
    unique_netlists=("netlist_hash", "nunique"),
    prompt_mean=("prompt_token_length", "mean"),
    prompt_std=("prompt_token_length", "std"),
    prompt_min=("prompt_token_length", "min"),
    prompt_q1=("prompt_token_length", lambda x: x.quantile(0.25)),
    prompt_median=("prompt_token_length", "median"),
    prompt_q3=("prompt_token_length", lambda x: x.quantile(0.75)),
    prompt_max=("prompt_token_length", "max"),
    
    full_mean=("full_convo_token_length", "mean"),
    full_std=("full_convo_token_length", "std"),
    full_min=("full_convo_token_length", "min"),
    full_q1=("full_convo_token_length", lambda x: x.quantile(0.25)),
    full_median=("full_convo_token_length", "median"),
    full_q3=("full_convo_token_length", lambda x: x.quantile(0.75)),
    full_max=("full_convo_token_length", "max")
).reset_index()

# Fill NaN standard deviations with 0 (happens if a gate_count only has 1 sample)
grouped = grouped.fillna(0)

#%%

# Create a figure with 1 row and 2 columns
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Prompt Token Length", "Full Conversation Token Length"),
    horizontal_spacing=0.1
)

# Figure 1: Prompt Tokens (Pre-computed Box Plot with Mean/Std)
fig.add_trace(
    go.Box(
        x=grouped["gate_count"],
        lowerfence=grouped["prompt_min"],
        q1=grouped["prompt_q1"],
        median=grouped["prompt_median"],
        q3=grouped["prompt_q3"],
        upperfence=grouped["prompt_max"],
        mean=grouped["prompt_mean"],
        sd=grouped["prompt_std"],
        boxmean='sd', # Instructs Plotly to draw the mean and standard deviation
        customdata=grouped["unique_netlists"],
        hovertemplate=(
            "<b>Gate Count: %{x}</b><br>"
            "Unique Netlists: %{customdata}<br>"
            "Max: %{upperfence}<br>"
            "Q3: %{q3}<br>"
            "Median: %{median}<br>"
            "Q1: %{q1}<br>"
            "Min: %{lowerfence}<br>"
            "Mean: %{mean:.1f} ± %{sd:.1f}"
            "<extra></extra>"
        ),
        name="Prompt Tokens",
        marker_color="#636EFA" # Plotly blue
    ),
    row=1, col=1
)

# Figure 2: Full Conversation Tokens (Pre-computed Box Plot with Mean/Std)
fig.add_trace(
    go.Box(
        x=grouped["gate_count"],
        lowerfence=grouped["full_min"],
        q1=grouped["full_q1"],
        median=grouped["full_median"],
        q3=grouped["full_q3"],
        upperfence=grouped["full_max"],
        mean=grouped["full_mean"],
        sd=grouped["full_std"],
        boxmean='sd',
        customdata=grouped["unique_netlists"],
        hovertemplate=(
            "<b>Gate Count: %{x}</b><br>"
            "Unique Netlists: %{customdata}<br>"
            "Max: %{upperfence}<br>"
            "Q3: %{q3}<br>"
            "Median: %{median}<br>"
            "Q1: %{q1}<br>"
            "Min: %{lowerfence}<br>"
            "Mean: %{mean:.1f} ± %{sd:.1f}"
            "<extra></extra>"
        ),
        name="Full Convo Tokens",
        marker_color="#EF553B" # Plotly red
    ),
    row=1, col=2
)

# Update layout to add titles and clean up the aesthetic
fig.update_layout(
    title_text="Token Length Analysis by Gate Count (Mean ± Std Dev)",
    title_x=0.5,
    xaxis_title="Number of Gates",
    yaxis_title="Number of Tokens",
    xaxis2_title="Number of Gates",
    yaxis2_title="Number of Tokens",
    template="plotly_white",
    showlegend=False,
    height=600,
    width=1200
)

# Save the interactive plot to disk instead of showing it
output_filename = "token_analysis_boxplot.html"
print(f"Saving optimized plot to {output_filename}...")

# include_plotlyjs="cdn" prevents embedding the ~3MB Plotly library into the HTML file
fig.write_html(output_filename, include_plotlyjs="cdn")

# Explicitly delete the figure object and run garbage collection to free memory immediately
del df, grouped, fig
gc.collect()

print("Plot saved successfully and memory cleared.")
