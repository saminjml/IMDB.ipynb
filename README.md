# 🎬 IMDB Movie Review Sentiment Analysis with LSTM (NLP)

This repository contains an end-to-end Natural Language Processing (NLP) pipeline for binary sentiment classification (Positive/Negative) on the **IMDB 50K Movie Reviews Dataset** using **TensorFlow / Keras** and **Recurrent Neural Networks (LSTM)**.

---

## 📌 Project Overview
- **Objective:** Classify movie reviews into **Positive** or **Negative** sentiments based on textual semantics.
- **Dataset:** 50,000 highly polar movie reviews (25,000 Positive, 25,000 Negative).
- **Core Technology:** Deep Learning (Keras/TensorFlow), Word Embeddings, Long Short-Term Memory (LSTM), Scikit-Learn.

---

## 🏗 Model Architecture

The deep learning model is built using Keras `Sequential` API:

| Layer | Type | Specifications / Output Dimension | Notes |
| :--- | :--- | :--- | :--- |
| **1** | `Embedding` | `input_dim=10,000`, `output_dim=128`, `input_length=200` | Word vector representation |
| **2** | `LSTM` | 64 units | Captures temporal & sequential text context |
| **3** | `Dropout` | Rate = 0.5 | Prevents overfitting |
| **4** | `Dense` | 1 unit, `activation='sigmoid'` | Binary sentiment output probability |

---

## ⚙️ Data Preprocessing & Pipeline

1. **Label Encoding:** Encoded target labels (`positive` $\to$ 1, `negative` $\to$ 0).
2. **Train/Test Split:** Stratified 90/10 split (45,000 train samples / 5,000 test samples).
3. **Tokenization:** Top 10,000 most frequent words fitted on the training corpus.
4. **Padding & Truncation:** Padded and truncated review sequences to a fixed length of 200 tokens (`post-padding`).
5. **Training Callbacks:** 
   - `EarlyStopping` (monitored `val_loss`, restored best weights).
   - `ModelCheckpoint` (saved best performing `.keras` model artifact).

---

## 📊 Performance & Evaluation

- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Evaluation Metrics:**
  - Evaluated with Precision, Recall, F1-Score, and Confusion Matrix on an unseen test set (5,000 samples).
  - Includes an interactive text classification loop for real-time sentiment inference.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
