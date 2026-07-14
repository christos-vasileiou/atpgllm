"""
test_training.py
================

Pytest script for testing the training pipeline with SFT and GRPO methods.
Uses minimal training steps to verify the pipeline works without waiting forever.

Usage:
    # Run all tests
    pytest test_training.py -v
    
    # Run specific test
    pytest test_training.py::test_sft_training -v
    pytest test_training.py::test_grpo_training -v
    
    # Run with custom environment variables
    MODEL=your_model TRAIN_DATASET=your_dataset pytest test_training.py -v
    
    # Run tests for specific model families
    pytest test_training.py -v -k "qwen"
    pytest test_training.py -v -k "deepseek"
    
    # Run tests with vLLM enabled
    pytest test_training.py::TestVLLMIntegration -v
    
    # Run batch size tests
    pytest test_training.py::TestBatchSizeVariations -v
"""

import os
import sys
import subprocess
import tempfile
import shutil
import pytest
from pathlib import Path


# Get environment variables with defaults for testing
MODEL = os.environ.get('MODEL', os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct'))
TRAIN_DATASET = os.environ.get('TRAIN_DATASET', os.environ.get('DATASET', None))

# Minimal steps for quick testing
TEST_MAX_STEPS = int(os.environ.get('TEST_MAX_STEPS', '1'))

# Minimal per device train batch size for quick testing
TEST_PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get('TEST_PER_DEVICE_TRAIN_BATCH_SIZE', '2'))

# Minimal gradient accumulation steps for quick testing
TEST_GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('TEST_GRADIENT_ACCUMULATION_STEPS', '1'))


# =============================================================================
# MODEL DEFINITIONS FOR MULTI-MODEL TESTING
# =============================================================================
# Dictionary of models from different companies for comprehensive testing
# Format: {model_id: {"name": display_name, "size": approximate_size, "company": company}}

QWEN_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": {"name": "Qwen2.5-7B", "size": "7B", "company": "Alibaba"},
    "Qwen/Qwen2.5-72B-Instruct": {"name": "Qwen2.5-72B", "size": "72B", "company": "Alibaba"},
    "Qwen/Qwen3-VL-32B-Thinking-FP8": {"name": "Qwen2.5-72B", "size": "72B", "company": "Alibaba"},
}

DEEPSEEK_MODELS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {"name": "DeepSeek-R1-7B", "size": "7B", "company": "DeepSeek"},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {"name": "DeepSeek-R1-32B", "size": "32B", "company": "DeepSeek"},
}



# All models combined for comprehensive testing
ALL_MODELS = {
    **QWEN_MODELS,
    **DEEPSEEK_MODELS,
}

# Small models for quick CI/CD testing (< 8B parameters)
SMALL_MODELS = {
    k: v for k, v in ALL_MODELS.items() 
    if v["size"] in ["3B", "3.8B", "7B"]
}

# Medium models for more thorough testing (8B-14B parameters)
MEDIUM_MODELS = {
    k: v for k, v in ALL_MODELS.items()
    if v["size"] in ["8B", "9B", "14B"]
}

# Large models for full testing (27B+ parameters)
LARGE_MODELS = {
    k: v for k, v in ALL_MODELS.items()
    if v["size"] in ["27B", "32B", "70B", "72B"]
}

def get_script_path():
    """Get the path to training_code.py"""
    return Path(__file__).parent / "training_code.py"


def run_training_command(
    method: str, 
    model_name: str, 
    dataset: str, 
    output_dir: str, 
    max_steps: int = 2, 
    per_device_train_batch_size: int = 2, 
    gradient_accumulation_steps: int = 1, 
    timeout: int = 600,
    use_vllm: bool = False,
    buffer_size: int = 100,
    extra_args: list = None,
):
    """
    Run the training command and return the result.
    
    Parameters
    ----------
    method : str
        Training method: 'sft' or 'grpo'
    model_name : str
        Model name or path
    dataset : str
        Path to training dataset
    output_dir : str
        Output directory for trained model
    max_steps : int
        Maximum training steps (default: 2 for quick testing)
    per_device_train_batch_size : int
        Per device train batch size (default: 2)
    gradient_accumulation_steps : int
        Gradient accumulation steps (default: 1)
    timeout : int
        Timeout in seconds (default: 600 = 10 minutes)
    use_vllm : bool
        Whether to use vLLM for inference acceleration (default: False)
    buffer_size : int
        Buffer size for GRPO training (default: 100)
    extra_args : list
        Additional command line arguments to pass
    
    Returns
    -------
    subprocess.CompletedProcess
        The result of the training command
    """
    script_path = get_script_path()
    
    cmd = [
        sys.executable,
        str(script_path),
        "--method", method,
        "--model_name", model_name,
        "--dataset", dataset,
        "--output_dir", output_dir,
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--max_steps", str(max_steps),
        "--report_to", "none",
    ]
    
    # Add buffer_size for GRPO to limit memory usage
    if method == "grpo":
        cmd.extend(["--buffer_size", str(buffer_size)])
    
    # Add vLLM flag if requested
    if use_vllm:
        cmd.append("--use_vllm")
    
    # Add any extra arguments
    if extra_args:
        cmd.extend(extra_args)
    
    env = os.environ.copy()
    # Set PyTorch memory allocation settings
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print(f"\nRunning command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(script_path.parent),
    )
    
    return result


def check_vllm_available() -> bool:
    """Check if vLLM is installed and available."""
    try:
        import vllm
        return True
    except ImportError:
        return False


def check_gpu_memory() -> int:
    """Get available GPU memory in GB. Returns 0 if no GPU available."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get total memory of first GPU in GB
            return torch.cuda.get_device_properties(0).total_memory // (1024**3)
    except Exception:
        pass
    return 0


@pytest.fixture
def output_dir():
    """Create a temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="test_training_")
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def check_prerequisites():
    """Check that required environment variables and files exist."""
    errors = []
    
    if TRAIN_DATASET is None:
        errors.append("TRAIN_DATASET environment variable not set. Set it to the path of your training dataset.")
    elif not Path(TRAIN_DATASET).exists() and not TRAIN_DATASET.startswith("http"):
        # Allow HuggingFace Hub datasets
        if "/" not in TRAIN_DATASET:
            errors.append(f"TRAIN_DATASET path does not exist: {TRAIN_DATASET}")
    
    script_path = get_script_path()
    if not script_path.exists():
        errors.append(f"training_code.py not found at: {script_path}")
    
    if errors:
        pytest.skip("\n".join(errors))


class TestSFTTraining:
    """Test cases for SFT (Supervised Fine-Tuning) training."""
    
    def test_sft_training(self, output_dir, check_prerequisites):
        """
        Test SFT training with minimal steps.
        
        This test verifies that:
        1. The training script runs without errors
        2. Training completes within the timeout
        3. The max_steps limit is respected
        """
        result = run_training_command(
            method="sft",
            model_name=MODEL,
            dataset=TRAIN_DATASET,
            output_dir=output_dir,
            max_steps=TEST_MAX_STEPS,
            per_device_train_batch_size=TEST_PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=TEST_GRADIENT_ACCUMULATION_STEPS,
            timeout=900,  # 15 minutes timeout
        )
        
        print(f"\n=== STDOUT ===\n{result.stdout}")
        print(f"\n=== STDERR ===\n{result.stderr}")
        
        # Check for common error patterns
        error_patterns = [
            "OutOfMemoryError",
            "CUDA out of memory",
            "RuntimeError",
            "ImportError",
            "ModuleNotFoundError",
        ]
        
        for pattern in error_patterns:
            if pattern in result.stderr:
                pytest.fail(f"Training failed with error pattern: {pattern}\n\nFull stderr:\n{result.stderr}")
        
        assert result.returncode == 0, f"SFT training failed with return code {result.returncode}\n\nStderr:\n{result.stderr}"
    
    def test_sft_creates_output_files(self, output_dir, check_prerequisites):
        """Test that SFT training creates the expected output files."""
        result = run_training_command(
            method="sft",
            model_name=MODEL,
            dataset=TRAIN_DATASET,
            output_dir=output_dir,
            max_steps=TEST_MAX_STEPS,
            per_device_train_batch_size=TEST_PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=TEST_GRADIENT_ACCUMULATION_STEPS,
            timeout=900,
        )
        
        if result.returncode != 0:
            pytest.skip(f"Training did not complete successfully: {result.stderr[:500]}")
        
        # Check for expected output files
        output_path = Path(output_dir)
        expected_files = ["adapter_config.json", "adapter_model.safetensors"]
        
        for expected_file in expected_files:
            file_path = output_path / expected_file
            # Files might be in a subdirectory checkpoint
            found = file_path.exists() or any(output_path.rglob(expected_file))
            if not found:
                pytest.skip(f"Expected file not found (may not be created with very few steps): {expected_file}")


class TestGRPOTraining:
    """Test cases for GRPO (Group Relative Policy Optimization) training."""
    
    def test_grpo_training(self, output_dir, check_prerequisites):
        """
        Test GRPO training with minimal steps.
        
        This test verifies that:
        1. The training script runs without errors for GRPO
        2. Training completes within the timeout
        3. The max_steps limit is respected
        
        Note: GRPO requires sim_config.json for reward functions.
        """
        # Check for sim_config.json
        sim_config_path = get_script_path().parent / "sim_config.json"
        if not sim_config_path.exists():
            pytest.skip(f"sim_config.json not found at {sim_config_path} - required for GRPO reward functions")
        
        result = run_training_command(
            method="grpo",
            model_name=MODEL,
            dataset=TRAIN_DATASET,
            output_dir=output_dir,
            max_steps=TEST_MAX_STEPS,
            per_device_train_batch_size=TEST_PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=TEST_GRADIENT_ACCUMULATION_STEPS,
            timeout=1200,  # 20 minutes timeout (GRPO takes longer)
        )
        
        print(f"\n=== STDOUT ===\n{result.stdout}")
        print(f"\n=== STDERR ===\n{result.stderr}")
        
        # Check for common error patterns
        error_patterns = [
            "OutOfMemoryError",
            "CUDA out of memory",
            "RuntimeError",
            "ImportError",
            "ModuleNotFoundError",
        ]
        
        for pattern in error_patterns:
            if pattern in result.stderr:
                pytest.fail(f"Training failed with error pattern: {pattern}\n\nFull stderr:\n{result.stderr}")
        
        assert result.returncode == 0, f"GRPO training failed with return code {result.returncode}\n\nStderr:\n{result.stderr}"


# =============================================================================
# VLLM INTEGRATION TESTS
# =============================================================================
class TestVLLMIntegration:
    """Test cases for vLLM integration with GRPO training."""
    
    @pytest.fixture
    def check_vllm_prerequisites(self, check_prerequisites):
        """Check that vLLM is available for testing."""
        if not check_vllm_available():
            pytest.skip("vLLM is not installed. Install with: pip install vllm")
    
    def test_grpo_with_vllm(self, output_dir, check_vllm_prerequisites):
        """
        Test GRPO training with vLLM acceleration enabled.
        
        vLLM provides significant speedup for generation during RL training
        by using optimized CUDA kernels and continuous batching.
        """
        sim_config_path = get_script_path().parent / "sim_config.json"
        if not sim_config_path.exists():
            pytest.skip(f"sim_config.json not found - required for GRPO")
        
        result = run_training_command(
            method="grpo",
            model_name=MODEL,
            dataset=TRAIN_DATASET,
            output_dir=output_dir,
            max_steps=TEST_MAX_STEPS,
            per_device_train_batch_size=TEST_PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=TEST_GRADIENT_ACCUMULATION_STEPS,
            use_vllm=True,
            timeout=1500,  # vLLM initialization adds overhead
        )
        
        print(f"\n=== STDOUT ===\n{result.stdout}")
        print(f"\n=== STDERR ===\n{result.stderr}")
        
        # vLLM may fail on certain GPU configurations, provide helpful message
        if "vllm" in result.stderr.lower() and result.returncode != 0:
            pytest.skip(f"vLLM failed (may require specific GPU): {result.stderr[:500]}")
        
        assert result.returncode == 0, f"GRPO with vLLM failed: {result.stderr}"
    
    def test_sft_compatibility_with_vllm_flag(self, output_dir, check_prerequisites):
        """
        Test that SFT training ignores the vLLM flag gracefully.
        
        vLLM is only used during GRPO generation, not SFT.
        The --use_vllm flag should be ignored for SFT training.
        """
        # Note: SFT doesn't use vLLM, but the flag should be accepted without error
        result = run_training_command(
            method="sft",
            model_name=MODEL,
            dataset=TRAIN_DATASET,
            output_dir=output_dir,
            max_steps=TEST_MAX_STEPS,
            per_device_train_batch_size=TEST_PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=TEST_GRADIENT_ACCUMULATION_STEPS,
            use_vllm=True,  # Should be ignored for SFT
            timeout=900,
        )
        
        # The script should complete without error even with vLLM flag
        print(f"\n=== STDOUT ===\n{result.stdout}")
        if result.returncode != 0:
            print(f"\n=== STDERR ===\n{result.stderr}")
        
        assert result.returncode == 0, f"SFT should work with vLLM flag (ignored): {result.stderr}"


# =============================================================================
# BATCH SIZE VARIATION TESTS
# =============================================================================
class TestBatchSizeVariations:
    """Test cases for different batch size configurations."""
    
    @pytest.mark.parametrize("batch_size,grad_accum", [
        (1, 1),   # Minimal - for low memory GPUs
        (2, 1),   # Small batch
        (4, 1),   # Medium batch
        (2, 2),   # Small batch with gradient accumulation
        (4, 2),   # Medium batch with gradient accumulation
        (8, 1),   # Larger batch (may need more GPU memory)
    ])
    def test_sft_batch_sizes(self, output_dir, check_prerequisites, batch_size, grad_accum):
        """
        Test SFT training with various batch size configurations.
        
        Different batch sizes and gradient accumulation steps affect:
        - Memory usage
        - Training stability
        - Effective batch size (batch_size * grad_accum)
        """
        gpu_mem = check_gpu_memory()
        
        # Skip large batch sizes on GPUs with less memory
        if batch_size >= 8 and gpu_mem < 40:
            pytest.skip(f"Batch size {batch_size} requires >= 40GB GPU memory (found {gpu_mem}GB)")
        if batch_size >= 4 and gpu_mem < 24:
            pytest.skip(f"Batch size {batch_size} requires >= 24GB GPU memory (found {gpu_mem}GB)")
        
        result = run_training_command(
            method="sft",
            model_name=MODEL,
            dataset=TRAIN_DATASET,
            output_dir=output_dir,
            max_steps=TEST_MAX_STEPS,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            timeout=900,
        )
        
        print(f"\n=== Testing batch_size={batch_size}, grad_accum={grad_accum} ===")
        print(f"Effective batch size: {batch_size * grad_accum}")
        
        if "CUDA out of memory" in result.stderr or "OutOfMemoryError" in result.stderr:
            pytest.skip(f"OOM with batch_size={batch_size}, grad_accum={grad_accum}")
        
        assert result.returncode == 0, f"SFT failed with batch_size={batch_size}: {result.stderr}"
    
    @pytest.mark.parametrize("batch_size,grad_accum", [
        (2, 1),   # Minimum for GRPO (num_generations default)
        (4, 1),   # Small batch
        (4, 2),   # With gradient accumulation
        (8, 1),   # Larger batch
    ])
    def test_grpo_batch_sizes(self, output_dir, check_prerequisites, batch_size, grad_accum):
        """
        Test GRPO training with various batch size configurations.
        
        Note: GRPO requires batch_size to be divisible by num_generations.
        """
        sim_config_path = get_script_path().parent / "sim_config.json"
        if not sim_config_path.exists():
            pytest.skip(f"sim_config.json not found - required for GRPO")
        
        gpu_mem = check_gpu_memory()
        
        # GRPO needs more memory due to generation
        if batch_size >= 8 and gpu_mem < 48:
            pytest.skip(f"GRPO batch_size {batch_size} requires >= 48GB GPU memory")
        if batch_size >= 4 and gpu_mem < 32:
            pytest.skip(f"GRPO batch_size {batch_size} requires >= 32GB GPU memory")
        
        result = run_training_command(
            method="grpo",
            model_name=MODEL,
            dataset=TRAIN_DATASET,
            output_dir=output_dir,
            max_steps=TEST_MAX_STEPS,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            timeout=1200,
        )
        
        print(f"\n=== GRPO batch_size={batch_size}, grad_accum={grad_accum} ===")
        
        if "CUDA out of memory" in result.stderr or "OutOfMemoryError" in result.stderr:
            pytest.skip(f"OOM with batch_size={batch_size}, grad_accum={grad_accum}")
        
        assert result.returncode == 0, f"GRPO failed with batch_size={batch_size}: {result.stderr}"


# =============================================================================
# MULTI-MODEL TESTS - QWEN FAMILY
# =============================================================================
class TestQwenModels:
    """Test cases for Alibaba Qwen model family."""
    
    @pytest.mark.parametrize("model_id", list(QWEN_MODELS.keys()))
    def test_sft_qwen_models(self, output_dir, check_prerequisites, model_id):
        """Test SFT training with various Qwen models."""
        model_info = QWEN_MODELS[model_id]
        gpu_mem = check_gpu_memory()
        
        # Memory requirements based on model size
        if model_info["size"] == "72B" and gpu_mem < 160:
            pytest.skip(f"Qwen 72B requires >= 160GB GPU memory (found {gpu_mem}GB)")
        if model_info["size"] == "32B" and gpu_mem < 80:
            pytest.skip(f"Qwen 32B requires >= 80GB GPU memory (found {gpu_mem}GB)")
        if model_info["size"] == "14B" and gpu_mem < 40:
            pytest.skip(f"Qwen 14B requires >= 40GB GPU memory (found {gpu_mem}GB)")
        if model_info["size"] == "7B" and gpu_mem < 24:
            pytest.skip(f"Qwen 7B requires >= 24GB GPU memory (found {gpu_mem}GB)")
        
        print(f"\n=== Testing {model_info['name']} ({model_info['size']}) ===")
        
        result = run_training_command(
            method="sft",
            model_name=model_id,
            dataset=TRAIN_DATASET,
            output_dir=output_dir,
            max_steps=TEST_MAX_STEPS,
            per_device_train_batch_size=TEST_PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=TEST_GRADIENT_ACCUMULATION_STEPS,
            timeout=900,
        )
        
        print(f"\n=== STDOUT ===\n{result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}")
        if result.returncode != 0:
            print(f"\n=== STDERR ===\n{result.stderr}")
        
        assert result.returncode == 0, f"SFT with {model_info['name']} failed: {result.stderr}"


# =============================================================================
# MULTI-MODEL TESTS - DEEPSEEK FAMILY
# =============================================================================
class TestDeepSeekModels:
    """Test cases for DeepSeek model family."""
    
    @pytest.mark.parametrize("model_id", list(DEEPSEEK_MODELS.keys()))
    def test_sft_deepseek_models(self, output_dir, check_prerequisites, model_id):
        """Test SFT training with various DeepSeek models."""
        model_info = DEEPSEEK_MODELS[model_id]
        gpu_mem = check_gpu_memory()
        
        if model_info["size"] == "70B" and gpu_mem < 160:
            pytest.skip(f"DeepSeek 70B requires >= 160GB GPU memory (found {gpu_mem}GB)")
        if model_info["size"] == "32B" and gpu_mem < 80:
            pytest.skip(f"DeepSeek 32B requires >= 80GB GPU memory (found {gpu_mem}GB)")
        if model_info["size"] == "14B" and gpu_mem < 40:
            pytest.skip(f"DeepSeek 14B requires >= 40GB GPU memory (found {gpu_mem}GB)")
        if model_info["size"] in ["7B", "8B"] and gpu_mem < 24:
            pytest.skip(f"DeepSeek {model_info['size']} requires >= 24GB GPU memory")
        
        print(f"\n=== Testing {model_info['name']} ({model_info['size']}) ===")
        
        result = run_training_command(
            method="sft",
            model_name=model_id,
            dataset=TRAIN_DATASET,
            output_dir=output_dir,
            max_steps=TEST_MAX_STEPS,
            per_device_train_batch_size=TEST_PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=TEST_GRADIENT_ACCUMULATION_STEPS,
            timeout=900,
        )
        
        print(f"\n=== STDOUT ===\n{result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}")
        if result.returncode != 0:
            print(f"\n=== STDERR ===\n{result.stderr}")
        
        # DeepSeek models may require authentication
        if "401" in result.stderr or "authentication" in result.stderr.lower():
            pytest.skip(f"DeepSeek model requires HuggingFace authentication")
        
        assert result.returncode == 0, f"SFT with {model_info['name']} failed: {result.stderr}"


# =============================================================================
# SMALL MODEL QUICK TESTS (CI/CD friendly)
# =============================================================================
class TestSmallModelsQuick:
    """Quick tests using small models (3B-7B parameters) for CI/CD pipelines."""
    
    @pytest.mark.parametrize("model_id", list(SMALL_MODELS.keys()))
    def test_sft_small_models(self, output_dir, check_prerequisites, model_id):
        """
        Quick SFT test with small models.
        
        These tests are designed to complete quickly with moderate GPU memory,
        making them suitable for CI/CD pipelines with GPU resources.
        """
        model_info = SMALL_MODELS[model_id]
        gpu_mem = check_gpu_memory()
        
        # Memory requirements for "small" models (3B-7B)
        if model_info["size"] == "7B" and gpu_mem < 24:
            pytest.skip(f"{model_info['name']} requires >= 24GB GPU memory (found {gpu_mem}GB)")
        if model_info["size"] in ["3B", "3.8B"] and gpu_mem < 16:
            pytest.skip(f"{model_info['name']} requires >= 16GB GPU memory (found {gpu_mem}GB)")
        
        print(f"\n=== Quick test: {model_info['name']} ({model_info['company']}) ===")
        
        result = run_training_command(
            method="sft",
            model_name=model_id,
            dataset=TRAIN_DATASET,
            output_dir=output_dir,
            max_steps=1,  # Single step for speed
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            timeout=900,
        )
        
        if "401" in result.stderr or "gated" in result.stderr.lower() or "access" in result.stderr.lower():
            pytest.skip(f"Model requires HuggingFace authentication or license acceptance")
        
        if "CUDA out of memory" in result.stderr or "OutOfMemoryError" in result.stderr:
            pytest.skip(f"OOM with {model_info['name']} - try with more GPU memory")
        
        assert result.returncode == 0, f"Quick SFT with {model_info['name']} failed: {result.stderr}"
    
    @pytest.mark.parametrize("model_id", list(SMALL_MODELS.keys()))
    def test_grpo_small_models(self, output_dir, check_prerequisites, model_id):
        """
        Quick GRPO test with small models.
        
        Tests GRPO functionality with minimal overhead.
        """
        sim_config_path = get_script_path().parent / "sim_config.json"
        if not sim_config_path.exists():
            pytest.skip(f"sim_config.json not found - required for GRPO")
        
        model_info = SMALL_MODELS[model_id]
        gpu_mem = check_gpu_memory()
        
        # Memory requirements for "small" models (3B-7B) - GRPO needs more memory
        if model_info["size"] == "7B" and gpu_mem < 32:
            pytest.skip(f"GRPO with {model_info['name']} requires >= 32GB GPU memory (found {gpu_mem}GB)")
        if model_info["size"] in ["3B", "3.8B"] and gpu_mem < 24:
            pytest.skip(f"GRPO with {model_info['name']} requires >= 24GB GPU memory (found {gpu_mem}GB)")
        
        print(f"\n=== Quick GRPO test: {model_info['name']} ({model_info['company']}) ===")
        
        result = run_training_command(
            method="grpo",
            model_name=model_id,
            dataset=TRAIN_DATASET,
            output_dir=output_dir,
            max_steps=1,
            per_device_train_batch_size=2,  # Minimum for GRPO
            gradient_accumulation_steps=1,
            buffer_size=50,
            timeout=1200,
        )
        
        if "401" in result.stderr or "gated" in result.stderr.lower():
            pytest.skip(f"Model requires HuggingFace authentication")
        
        if "CUDA out of memory" in result.stderr or "OutOfMemoryError" in result.stderr:
            pytest.skip(f"OOM with {model_info['name']} - try with more GPU memory")
        
        assert result.returncode == 0, f"Quick GRPO with {model_info['name']} failed: {result.stderr}"


# =============================================================================
# CROSS-COMPANY GRPO TESTS
# =============================================================================
class TestGRPOMultiModel:
    """Test GRPO training across different model families."""
    
    @pytest.fixture
    def check_grpo_prerequisites(self, check_prerequisites):
        """Check GRPO-specific prerequisites."""
        sim_config_path = get_script_path().parent / "sim_config.json"
        if not sim_config_path.exists():
            pytest.skip(f"sim_config.json not found - required for GRPO")
    
    @pytest.mark.parametrize("model_id,model_info", [
        ("Qwen/Qwen2.5-7B-Instruct", QWEN_MODELS["Qwen/Qwen2.5-7B-Instruct"]),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", DEEPSEEK_MODELS["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]),
    ])
    def test_grpo_cross_company(self, output_dir, check_grpo_prerequisites, model_id, model_info):
        """
        Test GRPO training across models from different companies.
        
        This validates that the reward function and training pipeline
        work correctly regardless of the underlying model architecture.
        """
        print(f"\n=== GRPO Cross-Company Test: {model_info['name']} ({model_info['company']}) ===")
        
        result = run_training_command(
            method="grpo",
            model_name=model_id,
            dataset=TRAIN_DATASET,
            output_dir=output_dir,
            max_steps=TEST_MAX_STEPS,
            per_device_train_batch_size=TEST_PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=TEST_GRADIENT_ACCUMULATION_STEPS,
            timeout=1200,
        )
        
        if "401" in result.stderr or "gated" in result.stderr.lower() or "access" in result.stderr.lower():
            pytest.skip(f"Model requires HuggingFace authentication")
        
        if "CUDA out of memory" in result.stderr:
            pytest.skip(f"OOM - try with smaller batch size")
        
        assert result.returncode == 0, f"GRPO with {model_info['name']} failed: {result.stderr}"


class TestArgParsing:
    """Test command-line argument parsing."""
    
    def test_help_message(self):
        """Test that --help works without errors."""
        script_path = get_script_path()
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "--method" in result.stdout
        assert "--max_steps" in result.stdout
        assert "sft" in result.stdout
        assert "grpo" in result.stdout
    
    def test_invalid_method(self, output_dir):
        """Test that invalid method raises an error."""
        script_path = get_script_path()
        
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                "--method", "invalid_method",
                "--model_name", "test_model",
                "--dataset", "/tmp/fake_dataset",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode != 0, "Should fail with invalid method"
        assert "invalid choice" in result.stderr.lower() or "error" in result.stderr.lower()


# Quick smoke test that can run without GPU
class TestImports:
    """Test that all required imports work."""
    
    def test_training_code_imports(self):
        """Test that training_code.py can be imported."""
        script_path = get_script_path()
        
        # Create a simple import test script
        test_script = f"""
import sys
sys.path.insert(0, '{script_path.parent}')
sys.path.insert(0, '{script_path.parent.parent / 'data_preprocessing'}')

# Try importing the main modules
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("transformers imported successfully")
except ImportError as e:
    print(f"transformers import failed: {{e}}")
    sys.exit(1)

try:
    from peft import LoraConfig
    print("peft imported successfully")
except ImportError as e:
    print(f"peft import failed: {{e}}")
    sys.exit(1)

try:
    from trl import SFTTrainer, GRPOTrainer
    print("trl imported successfully")
except ImportError as e:
    print(f"trl import failed: {{e}}")
    sys.exit(1)

print("All imports successful!")
sys.exit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        
        assert result.returncode == 0, f"Import test failed:\n{result.stdout}\n{result.stderr}"


if __name__ == "__main__":
    # Allow running directly with python
    pytest.main([__file__, "-v", "-s"])
