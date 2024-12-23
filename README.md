# MRZ OCR Project

This project demonstrates Optical Character Recognition (OCR) capabilities using various libraries and frameworks including OpenCV, Pillow, Tesseract, EasyOCR, and PaddleOCR.

## Overview

The project aims to provide a comprehensive solution for extracting text from images. It includes features for skew correction, noise removal, and text enhancement to improve OCR accuracy.

## Features

- **Multiple OCR Engines**: Supports EasyOCR and PaddleOCR for text extraction.
- **Skew Correction**: Utilizes image processing techniques to align text properly.
- **Noise Removal**: Cleans up images to enhance text readability.
- **Text Enhancement**: Adjusts font thickness and clarity for improved OCR performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ariel-ml/python-ocr.git
   ```
2. Navigate to the project directory:
   ```bash
   cd python-ocr
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Tesseract:
   ```bash
   sudo apt-get install tesseract-ocr
   sudo apt-get install libtesseract-dev
   ```
## Usage

1. Place your image file in the `assets` directory.
2. Run the notebooks.

## Acknowledgments

- Thanks to the contributors of the OpenCV, Pillow, Tesseract, EasyOCR, and PaddleOCR projects for their fantastic tools.
