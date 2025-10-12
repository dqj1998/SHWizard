import os
import sys
import platform
import PyInstaller.__main__

def build_binary():
    system = platform.system().lower()
    
    name = "shwizard"
    if system == "windows":
        name += ".exe"
    
    data_files = [
        ('shwizard/data/*.yaml', 'shwizard/data'),
    ]
    
    args = [
        'shwizard/__main__.py',
        '--name', name,
        '--onefile',
        '--console',
        '--clean',
    ]
    
    for src, dst in data_files:
        args.extend(['--add-data', f'{src}{os.pathsep}{dst}'])
    
    args.extend([
        '--hidden-import', 'shwizard.core',
        '--hidden-import', 'shwizard.safety',
        '--hidden-import', 'shwizard.storage',
        '--hidden-import', 'shwizard.utils',
        '--hidden-import', 'shwizard.llm',
    ])
    
    print(f"Building binary for {system}...")
    print(f"Command: {' '.join(args)}")
    
    PyInstaller.__main__.run(args)
    
    print(f"\n✅ Binary built successfully!")
    print(f"Location: dist/{name}")

if __name__ == "__main__":
    build_binary()
