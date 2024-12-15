# Spam Email Detection using Naive Bayes

This project demonstrates how to build a **Spam Email Detection** model using **Naive Bayes**, a probabilistic machine learning algorithm, on the famous **SMS Spam Collection Dataset**.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data Description](#data-description)
- [Model Explanation](#model-explanation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to classify SMS messages as **spam** or **ham** (not spam). It utilizes the **Naive Bayes** classification algorithm, which is ideal for text classification tasks. The dataset is pre-processed and vectorized using the **CountVectorizer** method, and the model is trained using the **Multinomial Naive Bayes** algorithm.

The steps involved in this project include:
1. **Loading and Preprocessing** the dataset.
2. **Vectorizing** the text data (Bag of Words approach).
3. **Training** a Naive Bayes classifier.
4. **Evaluating** the model’s performance using accuracy and classification metrics.
5. **Testing** the model with new sample data.

## Installation

### Clone the repository:
```bash
git clone https://github.com/your-username/spam-email-detection.git
```

### Install required packages:
Make sure you have Python 3.6 or higher installed, then install the necessary dependencies using **pip**:

```bash
pip install -r requirements.txt
```

### Requirements:
- Python 3.6+
- pandas
- scikit-learn
- matplotlib
- seaborn

## Usage

### 1. Load and Preprocess the Data
The project uses the **SMS Spam Collection Dataset**. You can download the dataset and place it in the root directory of the project.

```bash
# Run the program
python spam_detection.py
```

### 2. Train the Model
The model is trained using the **Multinomial Naive Bayes** classifier. The text data is vectorized using **CountVectorizer**, which converts the SMS messages into a matrix of token counts. The model is then trained on this matrix.

### 3. Evaluate the Model
The model’s performance is evaluated on the test data, and the accuracy, precision, recall, and F1-score are displayed in the terminal.

### 4. Test the Model with New Data
You can test the trained model with new SMS messages by modifying the sample input in the `spam_detection.py` file.

```python
sample_email = ["Congratulations! You've won a $1000 gift card. Click here to claim."]
```

The program will output whether the message is **spam** or **ham**.

## Data Description

The dataset used in this project is the **SMS Spam Collection Dataset**. It contains 5,574 SMS messages labeled as **spam** or **ham** (not spam).

- **Columns**:
  - **v1**: Label (spam/ham)
  - **v2**: Message content (SMS or email message)

Here is a sample of the dataset:

| label | message                                      |
|-------|----------------------------------------------|
| ham   | Go until jurong point, crazy.. available only ... |
| spam  | Free entry in 2 a wkly comp to win FA Cup fina... |
| ham   | Ok lar... Joking wif u oni...                |

## Model Explanation

### Naive Bayes Algorithm
Naive Bayes is a probabilistic classifier based on **Bayes' Theorem**, which calculates the probability of a message belonging to each class (spam or ham). The algorithm assumes that the features (words) are **conditionally independent** of each other, which simplifies the computation of class probabilities.

### Steps:
1. **Preprocessing**: The text data is cleaned and converted into a format that can be processed by the machine learning model.
2. **Vectorization**: We use **CountVectorizer** to convert the SMS messages into numerical feature vectors (bag of words).
3. **Training**: The **Multinomial Naive Bayes** classifier is used to train the model on the vectorized text data.
4. **Evaluation**: The model is evaluated using metrics like accuracy, precision, recall, and F1-score.

## Results

The performance of the trained model can be seen in the following metrics:

- **Accuracy**: The overall accuracy of the model on the test data.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positives.
- **F1-score**: The harmonic mean of precision and recall, giving a balance between the two.

Here is an example of a classification report:

```
Accuracy: 0.9758

Classification Report:
              precision    recall  f1-score   support

          ham       0.98      0.99      0.99       965
         spam       0.94      0.85      0.89       146

    accuracy                           0.98      1111
   macro avg       0.96      0.92      0.94      1111
weighted avg       0.98      0.98      0.98      1111
```

## Contributing

Contributions are welcome! If you find a bug or want to improve the code, feel free to create a pull request. Please ensure that your changes are well-documented and tested.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Additional Notes:**
- You can adjust the code or expand it to use other models like **Logistic Regression**, if you’d like to compare performance.
- If you are uploading to GitHub, remember to include your **requirements.txt** with the necessary libraries.
