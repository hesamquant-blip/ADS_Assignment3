# ADS_Assignment3
# 🧠 Deep Learning Architectures: MLP, CNN, RNN, and Transformers

This repository contains the implementation and analysis for **Assignment 3** of the **Applied Data Analysis** course. The project explores various deep learning architectures using **TensorFlow/Keras** to solve Regression, Classification, and Time-Series Forecasting problems.

## 📌 Project Overview

The goal of this assignment is to understand the inner workings of modern neural networks, compare their performance, and analyze the effect of different hyperparameters.

### 🛠️ Technologies Used
* **Python**
* **TensorFlow & Keras** (Deep Learning Framework)
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib & Seaborn** (Visualization)
* **Google Colab** (Training Environment)

---

## 📂 Project Structure

### 1️⃣ Part 1: Multilayer Perceptron (MLP)
We implemented Fully Connected Neural Networks (FCNN) to analyze **Tesla (TSLA)** stock market data.
* **Tasks:**
    * **Regression:** Predicting the exact `Close` price for the next day.
    * **Classification:** Predicting market direction (Up/Down).
* **Experiments:**
    * Compared optimizers (**Adam** vs. **SGD**).
    * Analyzed the effect of **Dropout** to prevent overfitting.
    * Studied the impact of network depth (Number of layers).

### 2️⃣ Part 2: Convolutional Neural Networks (CNN)
We focused on Image Classification using the **Fashion MNIST** dataset.
* **Tasks:**
    * Built a custom CNN architecture from scratch.
    * Implemented **Data Augmentation** (Rotation, Flip, Zoom) to improve generalization.
    * **Transfer Learning:** Utilized a pre-trained **VGG16** model (feature extraction) and fine-tuned it for the dataset.
* **Analysis:**
    * Investigated the impact of kernel sizes and pooling types (Max vs. Average).

### 3️⃣ Part 3: Recurrent Neural Networks (RNN)
We returned to the **Tesla Stock** time-series data to handle sequential dependencies.
* **Models Implemented:**
    * **Vanilla RNN:** Simple Recurrent Units.
    * **LSTM:** Long Short-Term Memory networks to handle vanishing gradients.
    * **GRU:** Gated Recurrent Units for efficient training.
* **Key Findings:**
    * LSTMs and GRUs significantly outperformed simple RNNs in capturing long-term dependencies.
    * Analyzed **Bidirectional LSTMs** for sequence modeling.

### 4️⃣ Part 4: Transformers (Attention Mechanism)
We implemented a **Transformer Encoder** block from scratch for Time-Series Forecasting.
* **Approach:**
    * Used **Multi-Head Self-Attention** to capture global dependencies in stock price history.
    * Compared the performance of the Transformer model against the LSTM model.

---

## 🚀 How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    ```
2.  Install the required dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
    ```
3.  Open the Jupyter Notebook (`.ipynb` file) via **VS Code** or **Google Colab**.
4.  Ensure `tesla_stock_history.csv` is in the same directory.
5.  Run all cells to reproduce the experiments.

---

## 📊 Results & Highlights
* **MLP:** Achieved convergence on regression tasks; Dropout significantly reduced overfitting gaps.
* **CNN:** Custom CNN achieved ~90% accuracy on Fashion MNIST. Data Augmentation improved validation stability.
* **RNN:** LSTM provided the most stable predictions for stock prices among recurrent models.
* **Transformers:** Demonstrated that attention mechanisms can effectively model time-series data without recurrence.

---

## 👤 Author
* **Name:** [Your Name]
* **Course:** Applied Data Analysis (Tehran Institute for Advanced Studies)
* **Instructor:** Dr. Salavati
* **TA:** Peyman Naseri
