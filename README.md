# Hate Speech Detection with LSTM

This repository contains code for a Hate Speech Detection model using Long Short-Term Memory (LSTM) neural networks. The model is built using TensorFlow and Keras libraries and is trained on a dataset with labeled tweets to classify whether a tweet contains hate speech or not.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- TensorFlow (`tensorflow`)
- Keras (`keras`)
- pandas (`pandas`)
- numpy (`numpy`)
- scikit-learn (`sklearn`)
- matplotlib (`matplotlib`)
- seaborn (`seaborn`)

You can install the required libraries using `pip` by running:

```bash
pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn
```

## Dataset

The code assumes you have a dataset in CSV format, containing labeled tweets. In this example, the dataset is loaded from a file named "train.csv." The dataset should have two columns: "tweet" (containing the text of the tweet) and "label" (containing the binary label, where 0 represents non-hate speech and 1 represents hate speech).

## Data Preprocessing

The code includes data preprocessing steps, such as:

- Balancing the dataset by oversampling the minority class (hate speech).
- Removing emojis and special characters from tweets.
- Tokenizing and padding the text sequences for model input.

## Model Architecture

The Hate Speech Detection model is built using LSTM layers. The architecture consists of the following layers:

1. Embedding Layer: Converts text tokens to dense vectors.
2. Dropout Layer: Reduces overfitting.
3. LSTM Layer: Long Short-Term Memory layer for sequence processing.
4. Flatten Layer: Converts the LSTM output to a 2D tensor.
5. Dense Layers: Fully connected layers with ReLU activation and L2 regularization.
6. Output Layer: Sigmoid activation for binary classification.

## Training and Evaluation

The code includes training the model on the preprocessed dataset and evaluating its performance on a validation set. Metrics such as binary accuracy and loss are monitored during training.

After training, the model is used to make predictions on a test dataset, and the predictions are visualized using scatter plots and distribution plots. A cutoff threshold is applied to classify the predictions as hate speech or non-hate speech.

Finally, the code can be used to predict hate speech in a separate test dataset, and the results are displayed, showing the tweets categorized as hate speech.

## Usage

1. Prepare your labeled dataset in CSV format with columns "tweet" and "label."
2. Update the dataset file name in the code (e.g., `"train.csv"`).
3. Run the code to train the Hate Speech Detection model and make predictions.

Please note that this README provides an overview of the code. It's essential to understand the data and model performance before deploying this model in real-world applications.
