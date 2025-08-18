# LLM-Based QA Model for Time-Course Data Interpretation

This repository provides a Question Answering (QA) model designed to interpret and analyze time-course (TC) data, with a focus on clinical and physiological datasets. The model leverages large language models (LLMs) to answer user queries about time-series data, such as patient vital signs, laboratory results, and other clinical measurements.

## Features
- **Interactive GUI**: A graphical user interface (`tc_QA_model_GUI.py`) for user-friendly interaction with the QA model.
- **LLM Integration**: Utilizes LLMs to interpret and explain time-course data, including sepsis phenotypes and other clinical scenarios.
- **Template and Example Scripts**: Includes example notebooks and template scripts for customization and extension.
- **Output Management**: Stores model outputs and interpretations in the `Output/` directory for easy access and review.

## Main Files
- `tc_QA_model.py`: Core logic for the QA model.
- `tc_QA_model_GUI.py`: GUI for interacting with the QA model.
phenotype analysis.
- `example_QA.ipynb`: Example Jupyter notebook demonstrating model usage.
- `environment.yml`: Conda environment file with required dependencies.

## Getting Started
1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd tc_QA_model
   ```
2. **Set up the environment**:
   ```bash
   conda env create -f environment.yml
   conda activate tc_QA_model
   ```
3. **Run the GUI**:
   ```bash
   python tc_QA_model_GUI.py
   ```
4. **Explore Examples**:
   Open `example_QA.ipynb` in Jupyter Notebook for sample usage.

### My Tips for Environment Setup:
1. Create or update your conda environment:

    **From your project directory:**
    ```bash
    conda env update -f environment.yml
    ```
    **Or, if creating new:**
    ```bash
    conda env create -f environment.yml
    ```

2. Activate the environment:
    ```bash
    conda activate tc-qa_model
    ```
3. (Optional, but recommended) Ensure Tesseract is on your PATH:

    which tesseract

    **Should print a path like /opt/homebrew/bin/tesseract or /usr/local/bin/tesseract**

    **If not, install via conda or Homebrew:**
    ```bash
    brew install tesseract
    ```

4. (Optional) If you want drag-and-drop and tkinterdnd2 fails to install via pip, try:
    ```bash
    pip install tkinterdnd2
    ```

## Output
- Model outputs and answers are saved in the `Output/` directory.

## License
This project is for research and educational purposes. Please see the repository for license details.

## Contact
For questions or contributions, please open an issue or contact the repository maintainer.
