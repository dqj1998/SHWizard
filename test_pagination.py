#!/usr/bin/env python3
"""
Test script for pagination functionality.
Creates test files and demonstrates pagination with cat, ls, etc.
"""

import os
import tempfile
import subprocess

def create_test_files():
    """Create test files with varying sizes."""
    test_dir = tempfile.mkdtemp(prefix="shwizard_test_")
    print(f"Created test directory: {test_dir}")
    
    # Create a small file (should not paginate)
    small_file = os.path.join(test_dir, "small.txt")
    with open(small_file, 'w') as f:
        for i in range(20):
            f.write(f"Line {i+1}: This is a small file for testing\n")
    
    # Create a medium file (should paginate)
    medium_file = os.path.join(test_dir, "medium.txt")
    with open(medium_file, 'w') as f:
        for i in range(100):
            f.write(f"Line {i+1}: This is a medium-sized file that should trigger pagination\n")
    
    # Create a large file (should definitely paginate)
    large_file = os.path.join(test_dir, "large.txt")
    with open(large_file, 'w') as f:
        for i in range(500):
            f.write(f"Line {i+1}: This is a large file with lots of content for testing pagination feature\n")
    
    print(f"\nTest files created:")
    print(f"  Small file (20 lines): {small_file}")
    print(f"  Medium file (100 lines): {medium_file}")
    print(f"  Large file (500 lines): {large_file}")
    print(f"\nTest directory: {test_dir}")
    
    return test_dir

def print_test_commands(test_dir):
    """Print test commands to try."""
    print("\n" + "="*70)
    print("Test Commands to Try in SHWizard:")
    print("="*70)
    print("\n1. Small file (no pagination):")
    print(f"   cat {test_dir}/small.txt")
    print("\n2. Medium file (should paginate):")
    print(f"   cat {test_dir}/medium.txt")
    print("\n3. Large file (should paginate):")
    print(f"   cat {test_dir}/large.txt")
    print("\n4. List directory with long format:")
    print(f"   ls -la {test_dir}")
    print("\n5. Find all files recursively:")
    print(f"   find {test_dir}")
    print("\n6. Show file with line numbers:")
    print(f"   cat -n {test_dir}/large.txt")
    print("\n7. Tail large file:")
    print(f"   tail -100 {test_dir}/large.txt")
    print("\n" + "="*70)
    print("\nStart SHWizard in interactive mode and try these commands:")
    print("  shwizard -i")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_dir = create_test_files()
    print_test_commands(test_dir)
    
    # Test should_paginate_output function
    print("\nTesting should_paginate_output function:")
    from shwizard.utils.output_utils import should_paginate_output
    
    small_output = "Line 1\n" * 20
    medium_output = "Line 1\n" * 100
    large_output = "Line 1\n" * 500
    
    print(f"  Small output (20 lines): should_paginate = {should_paginate_output(small_output)}")
    print(f"  Medium output (100 lines): should_paginate = {should_paginate_output(medium_output)}")
    print(f"  Large output (500 lines): should_paginate = {should_paginate_output(large_output)}")
    
    print(f"\nTest directory will be kept at: {test_dir}")
    print("Remember to clean it up when done: rm -rf", test_dir)
