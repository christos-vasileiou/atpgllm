import torch

print("1. PyTorch imported successfully.")

# Define a simple function that does some math
def compute_heavy_math(x, y):
    return torch.sin(x) * torch.cos(y) + torch.nn.functional.relu(x)

# Wrap the function with torch.compile (this uses the Inductor backend by default)
compiled_fn = torch.compile(compute_heavy_math)

# Create some dummy data on the CPU 
x = torch.randn(1024, 1024, device="cpu")
y = torch.randn(1024, 1024, device="cpu")

print("2. Triggering Inductor C++ compilation (this might take a few seconds)...")

# The first execution is what actually triggers the C++ code generation and GCC/Clang compilation
try:
    result = compiled_fn(x, y)
    print("3. Success! Compilation finished without fatal errors.")
    print(f"   Output tensor shape: {result.shape}")
except Exception as e:
    print(f"\nCompilation failed with error:\n{e}")