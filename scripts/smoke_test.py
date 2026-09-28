import sys

import torch
import transformers
import peft
import accelerate
import datasets


def main():
    print("=== Third Eye Environment Smoke Test ===")

    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"PEFT:         {peft.__version__}")
    print(f"Accelerate:   {accelerate.__version__}")
    print(f"Datasets:     {datasets.__version__}")

    print()

    if torch.cuda.is_available():
print("Device: CUDA")
    print("GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    print("Device: MPS / Apple Silicon")
else:
    print("Device: CPU")
    print()
    print("Environment OK")


if __name__ == "__main__":
    main()
