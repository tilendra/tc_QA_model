1. Create or update your conda environment:

# From your project directory:
conda env update -f environment.yml
# Or, if creating new:
conda env create -f environment.yml



2. Activate the environment:

conda activate tc-qa_model

3. (Optional, but recommended) Ensure Tesseract is on your PATH:

which tesseract
# Should print a path like /opt/homebrew/bin/tesseract or /usr/local/bin/tesseract
# If not, install via conda or Homebrew:
brew install tesseract

4. (Optional) If you want drag-and-drop and tkinterdnd2 fails to install via pip, try:

pip install tkinterdnd2
