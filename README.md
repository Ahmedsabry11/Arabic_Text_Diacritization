# Arabic Text Diacritization 📝🇸🇦

This project focuses on **automatically adding diacritics (tashkeel)** to Arabic text. Given an undiacritized Arabic sentence, the model outputs the same sentence with appropriate diacritics applied **after each character**.

---

## 🔍 Project Overview

Diacritics in Arabic are crucial for proper pronunciation, disambiguation, and meaning. However, most Arabic texts omit them, creating challenges in language understanding tasks. This project aims to bridge that gap by building a deep learning model capable of restoring diacritics automatically.

---

## ✨ Features

- Takes Arabic sentences as input and predicts diacritics character-by-character.
- Supports character-level modeling for high precision in diacritic placement.
- Multiple deep learning architectures implemented and compared:
  - ✅ RNN
  - ✅ LSTM
  - ✅ CBHG (Convolutional Bank + Highway + GRU)
- Preprocessing pipeline to clean, filter, and extract training labels from raw Arabic text.

---

## 🧠 Models Used

- **RNN (Recurrent Neural Networks)**: Baseline sequential model.
- **LSTM (Long Short-Term Memory)**: Handles long-term dependencies better.
- **CBHG**: Combines convolutional, highway, and GRU layers to extract richer features and context.

---

## 📦 Tech Stack

- **Frameworks & Libraries:**
  - `PyTorch`, `TorchVision`, `TorchAudio`, `CUDA Toolkit 11.1`
  - `Gensim`, `NLTK`, `Tokenizers`, `SentencePiece`
  - `python-bidi`, `arabic-reshaper`, `PyArabic`

- **Languages**:
  - Python

- **Environment Management**:
  - `conda` with channels: `pytorch`, `conda-forge`

---

## 📚 Dataset

- Custom dataset of Arabic text **with diacritics**.
- Preprocessing includes:
  - Text normalization and reshaping
  - Filtering non-Arabic or ambiguous samples
  - Splitting into training, validation, and test sets
  - Extracting target labels for character-level diacritization

---


## 📊 Evaluation

- Evaluation metrics: **Accuracy**, **Character-level F1**, and **Diacritic error rate**
- Visual inspection of output samples
- Performance compared across different model architectures

> *Note: Add example results or graphs here if available.*

---

## 🛠️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Arabic_Text_Diacritization.git
cd Arabic_Text_Diacritization
```

### 2. Create and activate the conda environment

```bash
conda create -n arabic-diacritics python=3.9
conda activate arabic-diacritics
```

### 3. Install dependencies

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
pip install gensim nltk tokenizers sentencepiece python-bidi arabic-reshaper PyArabic
```

### 4. Run training

```bash
python LSTM_Training
```


---

## 🚧 Future Work

- Improve generalization to unseen texts
- Add Transformer-based models for comparison
- Optimize for real-time inference
- Build a web or API interface for public use

---

## 👤 Author

**Your Name**  
GitHub: [@Ahmedsabry11](https://github.com/Ahmedsabry11)
