"""Pytest configuration and fixtures."""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set test environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"

# Mock tensorboard before any imports that might use it
# This prevents ModuleNotFoundError when tensorboard is not installed
import importlib.util
if "tensorboard" not in sys.modules:
    # Create a proper mock module with __spec__ attribute
    mock_tensorboard = MagicMock()
    mock_tensorboard.__spec__ = MagicMock()
    mock_tensorboard.__spec__.name = "tensorboard"
    sys.modules["tensorboard"] = mock_tensorboard
    
    # Mock torch.utils.tensorboard
    if "torch.utils.tensorboard" not in sys.modules:
        mock_torch_tb = MagicMock()
        mock_summary_writer = MagicMock()
        mock_torch_tb.SummaryWriter = mock_summary_writer
        sys.modules["torch.utils.tensorboard"] = mock_torch_tb

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "data"

# Create test data directory if it doesn't exist
TEST_DATA_DIR.mkdir(exist_ok=True)

# Fixtures can be defined here or imported from other modules
# They will be automatically discovered by pytest
