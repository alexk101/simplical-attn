#!/usr/bin/env python3
"""
Comprehensive test runner for 2-Simplicial Attention implementation.

This script runs all available tests:
- Correctness tests (forward/backward pass validation)
- Performance benchmarks vs traditional attention
- Quick validation tests
"""

import sys
import torch

def main():
    print("🧪 2-Simplicial Attention - Comprehensive Test Suite")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - some tests will be skipped")
        print("   Triton kernels require CUDA to run")
    else:
        device = torch.cuda.get_device_name()
        print(f"🚀 Using CUDA device: {device}")
    
    print()
    
    # Import test functions
    try:
        from simplicial import (
            run_all_correctness_tests,
            run_all_performance_tests, 
            quick_performance_test,
            run_all_tests
        )
    except ImportError as e:
        print(f"❌ Failed to import test functions: {e}")
        print("   Make sure you're in the correct directory and all dependencies are installed")
        return False
    
    success = True
    
    # 1. Quick validation first
    print("1️⃣  QUICK VALIDATION TESTS")
    print("-" * 30)
    try:
        quick_success = run_all_tests()
        if not quick_success:
            print("   ⚠️  Quick validation had issues")
            success = False
    except Exception as e:
        print(f"   ❌ Quick tests failed: {e}")
        success = False
    
    print("\n")
    
    # 2. Comprehensive correctness tests
    print("2️⃣  COMPREHENSIVE CORRECTNESS TESTS")
    print("-" * 40)
    try:
        correctness_success = run_all_correctness_tests()
        if not correctness_success:
            print("   ⚠️  Some correctness tests failed")
            success = False
    except Exception as e:
        print(f"   ❌ Correctness tests failed: {e}")
        success = False
    
    print("\n")
    
    # 3. Performance benchmarks
    print("3️⃣  PERFORMANCE BENCHMARKS")
    print("-" * 30)
    try:
        # Start with quick performance test
        print("Running quick performance test...")
        quick_performance_test()
        
        print("\nRunning comprehensive performance benchmarks...")
        print("(This may take several minutes...)")
        performance_success = run_all_performance_tests()
        if not performance_success:
            print("   ⚠️  Some performance tests had issues")
            success = False
    except Exception as e:
        print(f"   ❌ Performance tests failed: {e}")
        success = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    
    if success:
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n✅ Your 2-simplicial attention implementation is working correctly")
        print("✅ Performance benchmarks completed")
        print("\nNext steps:")
        print("  • Integration into larger models")
        print("  • Production deployment")
        print("  • Further optimization if needed")
    else:
        print("⚠️  SOME TESTS HAD ISSUES")
        print("\n📋 Troubleshooting:")
        print("  • Check CUDA availability for Triton kernels")
        print("  • Verify all dependencies are installed")
        print("  • Check GPU memory availability")
        print("  • Review error messages above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 