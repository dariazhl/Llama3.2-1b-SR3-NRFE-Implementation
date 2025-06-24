
import subprocess
import sys

def install_transformers():
    print("Installing transformers using pip...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "transformers==4.21.3", 
            "--no-deps"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" Transformers installed successfully")
            
            import transformers
            print(f" Transformers version: {transformers.__version__}")
            return True
        else:
            print(f" Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f" Installation error: {e}")
        return False

def install_tokenizers():
    print("Installing tokenizers...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "tokenizers==0.13.3"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" Tokenizers installed successfully")
            return True
        else:
            print(f" Tokenizers installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f" Tokenizers installation error: {e}")
        return False

if __name__ == "__main__":
    print("Installing ML dependencies for Enhanced NRFE training...")
    
    tokenizers_ok = install_tokenizers()
    
    transformers_ok = install_transformers()
    
    if transformers_ok and tokenizers_ok:
        print("\n All ML dependencies installed successfully")
        print("Ready to run full training commands with immediate epoch stopping")
    else:
        print("\n Some dependencies failed to install")
        print("Will proceed with available dependencies")