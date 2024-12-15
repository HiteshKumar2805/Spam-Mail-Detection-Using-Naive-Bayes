# Spam Email Detection Using Naive Bayes

This project demonstrates a machine learning model built to classify emails/SMS messages as either **Spam** or **Ham** (non-spam). The model uses the **Naive Bayes** algorithm and processes the text data using the **Bag of Words** approach with **unigrams** and **bigrams** for better feature extraction.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [License](#license)

## Project Overview

This project uses a Naive Bayes classifier to classify SMS and email messages as spam or not. The dataset contains labeled messages, where each message is classified as "spam" or "ham." The model is trained on the dataset, evaluated using cross-validation, and tested on unseen data to check its accuracy and effectiveness.

## Dataset

The dataset used in this project is the **SMS Spam Collection** dataset, which is available in `.csv` format. It contains two columns:
- **label**: `spam` or `ham`
- **message**: The content of the email or SMS message.

The dataset is pre-processed to encode the labels as `1` for spam and `0` for ham.

The dataset file `spam.csv` is included in this repository.

## Installation

To run the code on your local machine, ensure that you have the following Python libraries installed:

- `pandas`
- `scikit-learn`

You can install the necessary libraries by running:

```bash
pip install pandas scikit-learn
