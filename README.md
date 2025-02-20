# atpgllm

## Installation

### Requirements
- **Python 3.9** or higher
- **pip** package manager
- **CUDA-capable GPU** (recommended)

### Quick Install

**1. Clone the repository**
```bash
git clone https://github.com/christos-vasileiou/atpgllm.git
cd atpgllm
```

**2. Create and activate a virtual environment**
```bash
python -m venv myenv
source myenv/bin/activate  # On Linux/Mac
```

**3. Install the package and dependencies**
```bash
pip install -r requirements.txt --use-pep517 --no-cache-dir
```

### Troubleshooting

If you encounter installation issues:
**1.** Make sure you have **CUDA** installed for GPU support
**2.** Try removing any existing installations:
   ```bash
   pip uninstall atpgllm -y
   pip cache purge
   ```
**3.** Then reinstall with:
   ```bash
   pip install -r requirements.txt --use-pep517 --no-cache-dir
   ```

## Usage

### Training

For **multi/single-gpu** training use:
```bash
torchrun --proc_per_node=<NODES> script_name.py
# Example: torchrun --proc_per_node=4 sequence_multilabel_classification.py
```

For **gpu/cpu** training use:
```bash
python script_name.py
# Example: python sequence_multilabel_classification.py
```

## Project Goals

**Large Language Model for Design Testing and Fault-Modeling:**
1. **Stuck-at**: A specific net is "stuck" at a constant logic value (either 0 or 1)
2. **Transition**: Related to signal transitions between different logic levels (e.g., 0 to 1 or 1 to 0)
3. **Coupling**: ...
4. ...

**Ultimate Goal**: To cover as many Fault Models as possible.

### What may go wrong?
- **Physical Defects**: 
  - Shorts between two points (bridges)
  - Open in a line
  - Improper doping
  - Masking error
  - Particles on surface
  - Corrosion
- **Main Focus**: Safety and Reliability




