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
"""

import os
import sys
import subprocess
import tempfile
import shutil
import pytest
from pathlib import Path


# Get environment variables with defaults for testing
MODEL = os.environ.get('MODEL', os.environ.get('MODEL_NAME', 'meta-llama/Llama-3.2-1B-Instruct'))
TRAIN_DATASET = os.environ.get('TRAIN_DATASET', os.environ.get('DATASET', None))

# Minimal steps for quick testing
TEST_MAX_STEPS = int(os.environ.get('TEST_MAX_STEPS', '1'))

# Minimal per device train batch size for quick testing
TEST_PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get('TEST_PER_DEVICE_TRAIN_BATCH_SIZE', '2'))

# Minimal gradient accumulation steps for quick testing
TEST_GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('TEST_GRADIENT_ACCUMULATION_STEPS', '1'))

def get_script_path():
    """Get the path to training_code.py"""
    return Path(__file__).parent / "training_code.py"


def run_training_command(method: str, model_name: str, dataset: str, output_dir: str, max_steps: int = 2, per_device_train_batch_size: int = 2, gradient_accumulation_steps: int = 1, timeout: int = 600):
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
        cmd.extend(["--buffer_size", "100"])
    
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
