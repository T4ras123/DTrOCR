# LaTeXTrOCR 📝➡️📄

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.8%2B-orange.svg)
![GitHub Issues](https://img.shields.io/github/issues/YourUsername/LaTeXTrOCR)
![GitHub Stars](https://img.shields.io/github/stars/YourUsername/LaTeXTrOCR?style=social)

LaTeXTrOCR is a cutting-edge Transformer-based OCR (Optical Character Recognition) model designed to convert images of handwritten and printed mathematical equations directly into LaTeX code. By leveraging the power of deep learning and advanced tokenization techniques, LaTeXTrOCR aims to streamline the process of digitizing and editing mathematical content, making it an invaluable tool for researchers, educators, and students.

![LaTeXTrOCR Demo](https://github.com/YourUsername/LaTeXTrOCR/assets/demo.gif)

## Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Inference](#inference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Features
- **Transformer-Based Architecture**: Utilizes state-of-the-art Transformer models for accurate and efficient OCR.
- **Custom Tokenizer**: Specialized tokenizer tailored for LaTeX syntax and mathematical symbols.
- **ArXiv Scraper**: Automated tools to scrape and preprocess LaTeX documents from [arXiv](https://arxiv.org/) for training.
- **Flexible Dataset Handling**: Supports various image formats and preprocesses them for optimal model performance.
- **Interactive Training Loop**: Incorporates robust training scripts with logging and checkpointing.
- **Comprehensive Evaluation**: Tools to assess model performance with detailed metrics and visualizations.
- **Easy Integration**: Designed to be easily integrated into larger projects or used as a standalone tool.

## Demo
Check out our [demo video](https://github.com/YourUsername/LaTeXTrOCR/assets/demo-video.mp4) showcasing the model's capabilities in real-time.

![Model Architecture](https://github.com/YourUsername/LaTeXTrOCR/assets/model_architecture.png)

## Installation

### Prerequisites
- **Python 3.8+**
- **PyTorch 1.8+**
- **CUDA 10.2+** (for GPU support)

### Clone the Repository
<|code|>bash
git clone https://github.com/YourUsername/LaTeXTrOCR.git
cd LaTeXTrOCR
<|code|>

### Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

<|code|>bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
<|code|>

### Install Dependencies
<|code|>bash
pip install -r requirements.txt
<|code|>

### Additional Requirements
- **Tesseract OCR**: Install Tesseract OCR for preprocessing images.
  - **Ubuntu**:
    <|code|>bash
    sudo apt-get update
    sudo apt-get install tesseract-ocr
    <|code|>
  - **macOS**:
    <|code|>bash
    brew install tesseract
    <|code|>
  - **Windows**: [Download Installer](https://github.com/tesseract-ocr/tesseract/wiki)

## Usage

### 1. Preparing the Dataset
LaTeXTrOCR includes a scraper to download and preprocess LaTeX documents from arXiv.

<|code|>bash
python dataset/arxiv_scraper.py
<|code|>

This will:
- **Download**: Fetch `.tar.gz` archives of papers based on predefined queries.
- **Extract**: Unpack and extract `.tex` files from the archives.
- **Process**: Clean and prepare LaTeX content for training.

### 2. Tokenizing LaTeX Content
Train the custom tokenizer to handle LaTeX syntax effectively.

<|code|>bash
python tokenizer.py --text data/raw_la.tex --vocab_size 1000
<|code|>

This will generate a `tokenizer.json` file used during training and inference.

### 3. Training the Model
Start training the Transformer-based OCR model.

<|code|>bash
python models/trOCR.py
<|code|>

**Training Parameters**:
- Adjust hyperparameters in `config/config.yaml` as needed.
- Utilize GPU acceleration for faster training.

### 4. Running Inference
Convert an image of a handwritten equation to LaTeX.

<|code|>bash
python inference.py --image path/to/equation.png --model weights/ocr_model.pth
<|code|>

**Output**:
<|code|>latex
\frac{d}{dx}e^{x} = e^{x}
<|code|>

### 5. Evaluating the Model
Assess model performance with evaluation scripts.

<|code|>bash
python evaluate.py --model weights/ocr_model.pth --dataset data/test_images/
<|code|>

## Project Structure
<|code|>
LaTeXTrOCR/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── data/
│   ├── external/
│   ├── arxiv_papers/
│   └── data.txt
├── notebooks/
│   └── analysis.ipynb
├── models/
│   ├── trOCR.py
│   ├── encoder.py
│   └── infer.py
├── dataset/
│   ├── transforms.py
│   ├── dataset.py
│   ├── arxiv_scraper.py
│   └── extract_latex.py
├── tokenizer.py
├── utils/
│   └── utils.py
├── docker/
│   └── Dockerfile
└── config/
    └── config.yaml
<|code|>

## Contributing
Contributions are welcome and greatly appreciated! To contribute to LaTeXTrOCR, please follow these steps:

1. **Fork the repository**.
2. **Create a new branch**:
    <|code|>bash
    git checkout -b feature/YourFeature
    <|code|>
3. **Make your changes** and **commit them**:
    <|code|>bash
    git commit -m "Add your feature"
    <|code|>
4. **Push to the branch**:
    <|code|>bash
    git push origin feature/YourFeature
    <|code|>
5. **Create a Pull Request** detailing your changes.

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, suggestions, or feedback, feel free to reach out:

- **Twitter**: [@Vover163](https://twitter.com/Vover163)
- **Email**: <vovatara123@gmail.com>
- **GitHub Issues**: [Open an Issue](https://github.com/YourUsername/LaTeXTrOCR/issues)

## Acknowledgements
- **[arXiv](https://arxiv.org/)**: For providing access to a vast repository of academic papers.
- **[PyTorch](https://pytorch.org/)**: For the powerful deep learning framework.
- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)**: For the open-source OCR engine.
- **[tiktoken](https://github.com/openai/tiktoken)**: For efficient tokenization.
- **[Seaborn](https://seaborn.pydata.org/)**: For beautiful statistical data visualization.

---

Made with ❤️ by [Vover](https://github.com/YourUsername)