#!/usr/bin/env python3
"""
Environment Validation Script for LoRA Fine-tuning
Version: 1.0.2
Purpose: Validate all system requirements before fine-tuning
Optimized for: RTX 4060 Ti (16GB VRAM)

Usage: python scripts/validate_environment.py

Changelog:
- v1.0.2: Enhanced dataset validation and error reporting
- v1.0.1: Initial comprehensive validation script
"""

import os
import sys
import torch
import json
from datetime import datetime

# Version information
SCRIPT_VERSION = "1.0.2"
SCRIPT_NAME = "validate_environment.py"

def print_header():
    """Print script header"""
    print(f"🚀 {SCRIPT_NAME} v{SCRIPT_VERSION}")
    print(f"📅 Validation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def check_python_version():
    """Check Python version compatibility"""
    print("\n🐍 Python Version Check:")
    print(f"   Version: {sys.version}")
    print(f"   Executable: {sys.executable}")
    
    version_info = sys.version_info
    if version_info.major == 3 and version_info.minor >= 8:
        print("   ✅ Python version compatible (3.8+)")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False

def check_libraries():
    """Check if all required libraries are installed and compatible"""
    print("\n📚 Library Compatibility Check:")
    
    required_libs = {
        "torch": "PyTorch",
        "transformers": "Transformers", 
        "datasets": "Datasets",
        "peft": "PEFT",
        "accelerate": "Accelerate",
        "trl": "TRL"
    }
    
    optional_libs = {
        "numpy": "NumPy",
        "pandas": "Pandas", 
        "pyarrow": "PyArrow"
    }
    
    all_good = True
    
    for lib_name, display_name in required_libs.items():
        try:
            lib = __import__(lib_name)
            version = getattr(lib, '__version__', 'Unknown')
            print(f"   ✅ {display_name}: {version}")
        except ImportError:
            print(f"   ❌ {display_name}: NOT INSTALLED")
            all_good = False
    
    print("\n   Optional Libraries:")
    for lib_name, display_name in optional_libs.items():
        try:
            lib = __import__(lib_name)
            version = getattr(lib, '__version__', 'Unknown')
            print(f"   ✅ {display_name}: {version}")
        except ImportError:
            print(f"   ⚠️  {display_name}: NOT INSTALLED (optional)")
    
    return all_good

def check_cuda_availability():
    """Check CUDA and GPU availability"""
    print("\n🔥 CUDA & GPU Check:")
    
    if not torch.cuda.is_available():
        print("   ❌ CUDA not available")
        return False
    
    # CUDA available
    print(f"   ✅ CUDA available")
    print(f"   📊 CUDA version: {torch.version.cuda}")
    print(f"   🖥️ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"   🏗️ Compute capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    
    # Check for RTX 4060 Ti specifically
    gpu_name = torch.cuda.get_device_name(0)
    if "4060" in gpu_name:
        print(f"   ✅ RTX 4060 Ti detected")
    else:
        print(f"   ⚠️  Different GPU detected: {gpu_name}")
    
    return True

def check_memory_requirements():
    """Check if VRAM is sufficient for fine-tuning"""
    print("\n🧠 Memory Requirements Check:")
    
    if not torch.cuda.is_available():
        print("   ❌ Cannot check memory without CUDA")
        return False
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   📊 Total VRAM: {vram_gb:.1f}GB")
    
    # Test memory allocation
    try:
        # Allocate test tensors to check available memory
        test_size = 2048
        x = torch.randn(test_size, test_size).cuda()
        y = torch.randn(test_size, test_size).cuda()
        
        # Perform a simple operation
        z = torch.mm(x, y)
        torch.cuda.synchronize()
        
        print(f"   ✅ Memory test successful ({test_size}x{test_size} matrix)")
        del x, y, z
        torch.cuda.empty_cache()
        
        # Recommend batch size based on VRAM
        if vram_gb >= 16:
            print(f"   ✅ VRAM sufficient for batch_size=6-8")
            return True
        elif vram_gb >= 12:
            print(f"   ⚠️  VRAM moderate - recommend batch_size=4-6")
            return True
        else:
            print(f"   ❌ VRAM insufficient - need at least 12GB")
            return False
            
    except RuntimeError as e:
        print(f"   ❌ Memory test failed: {e}")
        return False

def check_dataset():
    """Check if dataset file exists and is valid"""
    print("\n📁 Dataset Check:")
    
    dataset_path = "data/instructions.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"   ❌ Dataset file not found: {dataset_path}")
        print("   📝 Create dataset with:")
        print('   cat > data/instructions.jsonl << \'JSONL\'')
        print('   {"system":"Eres un asistente experto.","input":"¿Cómo hacer X?","output":"Pasos para hacer X."}')
        print('   JSONL')
        return False
    
    # Check dataset format
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"   ✅ Dataset found: {len(lines)} examples")
        
        # Validate JSON format
        import json
        valid_examples = 0
        for i, line in enumerate(lines[:5]):  # Check first 5 examples
            try:
                example = json.loads(line.strip())
                if all(key in example for key in ['system', 'input', 'output']):
                    valid_examples += 1
            except json.JSONDecodeError:
                print(f"   ❌ Invalid JSON on line {i+1}")
                return False
        
        print(f"   ✅ Dataset format valid ({valid_examples}/5 examples checked)")
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading dataset: {e}")
        return False

def check_paths_and_permissions():
    """Check if paths exist and have proper permissions"""
    print("\n📂 Path & Permission Check:")
    
    required_paths = [
        "data",
        "scripts", 
        ".",
    ]
    
    for path in required_paths:
        if os.path.exists(path):
            print(f"   ✅ Path exists: {path}")
            if os.access(path, os.W_OK):
                print(f"      ✅ Write permission: {path}")
            else:
                print(f"      ❌ No write permission: {path}")
        else:
            print(f"   ❌ Path missing: {path}")
    
    # Check output directory
    output_dir = "models"
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"   ✅ Created output directory: {output_dir}")
        except Exception as e:
            print(f"   ❌ Cannot create output directory: {e}")
            return False
    
    return True

def run_performance_test():
    """Run a simple performance test"""
    print("\n⚡ Performance Test:")
    
    if not torch.cuda.is_available():
        print("   ❌ Cannot run performance test without CUDA")
        return False
    
    try:
        import time
        
        # Test 1: Matrix multiplication (good for checking raw GPU performance)
        size = 3000
        x = torch.randn(size, size).cuda()
        y = torch.randn(size, size).cuda()
        
        # Warm-up
        for _ in range(5):
            z = torch.mm(x, y)
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            z = torch.mm(x, y)
            torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / 10 * 1000
        print(f"   ✅ Matrix multiplication ({size}x{size}): {avg_time:.2f}ms")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
        
        # Provide performance rating
        if avg_time < 100:
            print(f"   🏆 Excellent GPU performance")
        elif avg_time < 200:
            print(f"   ✅ Good GPU performance") 
        else:
            print(f"   ⚠️  Moderate GPU performance")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def print_summary(results):
    """Print validation summary"""
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY:")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check}: {status}")
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Ready for fine-tuning!")
        print("\n🚀 You can now run:")
        print("   python scripts/finetune_lora.py")
        return True
    else:
        print("⚠️  SOME CHECKS FAILED - Please fix issues before fine-tuning")
        print("\n🔧 Common solutions:")
        print("   - Install missing libraries: pip install -r requirements.txt")
        print("   - Create dataset: cat > data/instructions.jsonl")
        print("   - Check CUDA installation")
        return False

def main():
    """Main validation function"""
    print_header()
    
    # Run all validation checks
    results = {
        "Python Version": check_python_version(),
        "Libraries": check_libraries(),
        "CUDA & GPU": check_cuda_availability(),
        "Memory": check_memory_requirements(),
        "Dataset": check_dataset(),
        "Paths": check_paths_and_permissions(),
        "Performance": run_performance_test()
    }
    
    # Print summary and return result
    ready = print_summary(results)
    
    # Save validation report
    report_path = "logs/validation_report.json"
    os.makedirs("logs", exist_ok=True)
    
    validation_report = {
        "timestamp": datetime.now().isoformat(),
        "script_version": SCRIPT_VERSION,
        "system_info": {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0
        },
        "validation_results": results,
        "ready_for_training": ready
    }
    
    try:
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        print(f"\n📄 Validation report saved: {report_path}")
    except Exception as e:
        print(f"\n❌ Could not save validation report: {e}")
    
    return ready

if __name__ == "__main__":
    main()
