# install_dependencies.py
import subprocess
import sys

def install_packages():
    """Install required packages"""
    packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'plotly',
        'openpyxl'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

if __name__ == "__main__":
    print("Installing required packages...")
    install_packages()
    print("All packages installed! You can now run store_analysis.py")