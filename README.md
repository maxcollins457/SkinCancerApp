# Skin Lesion Classification App

Skin Lesion Classification App is a Flask web application that allows users to select different models trained using TensorFlow to classify images of skin lesions. Users can upload their own images and choose a model to generate a report classifying the image. The app also includes tabs for Exploratory Data Analysis (EDA) and model comparison to provide users with more information about the available ML models.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)


## Introduction

The Skin Lesion Classification App is designed to assist users in classifying skin lesions through machine learning models. It provides an interactive platform to upload images, select a model, and generate classification reports.

## Features

- Upload images for classification
- Select from various TensorFlow models
- Generate classification reports
- Exploratory Data Analysis (EDA) tab
- Model comparison tab

## Getting Started

### Prerequisites

- Python 3.11
- Flask
- TensorFlow
- Other dependencies (specified in requirements.txt)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/maxcollins457/SkinCancerApp.git
   cd SkinCancerApp
   ```
2. Install dependencies

    ```bash 
    pip install -r requirements.txt
    ```

### Usage
1. Run the flask app using 

    ```bash
    python main.py
    ```
2. Open your browser and navigate to http://localhost:5000.

3. Explore the different tabs for image classification, EDA, and model comparison.

### Project Struture

Certainly! Here's an example of a project structure in Markdown:

markdown
Copy code
### Project Structure

This is a brief overview of the project structure.

#### Directory Structure

SkinCancerApp/

│

├── app/

│   ├── data/

│   │   └── CancerData.csv

│   ├── networks/

│   │   └── h5 keras models

│   ├── static/

│   │   ├── css/

│   │   ├── img/

│   │   └── js/

│   ├── templates/

│   │   └── html templates

│   └── views/

│       └── flask views

│   ├── `__init__.py`

│   ├── models.py

│

├── COLAB NOTEBOOKS/

│   └── ipynb notebooks

│

├── requirements.txt

├── README.md

└── main.py



#### Description

- **data:** Processed metadata from the dataset.
- **networks:** Stores trained machine learning models and related files.
- **static:** Images, CSS and JS for aesthetic of the flask app.
- **templates:** HTML templates for the flask app.
- **COLAB NOTEBOOKS:** Jupyter notebooks for exploratory data analysis and model training.
- **requirements.txt:** Lists project dependencies.
- **README.md:** Project documentation.

