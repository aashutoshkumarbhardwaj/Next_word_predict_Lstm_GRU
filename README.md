<div align="center">

# 📖 Next Word Prediction
### LSTM vs GRU · Trained on Project Gutenberg · 76.50% Accuracy

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-RNN-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-1d9e75?style=for-the-badge)](LICENSE)

**Two RNN architectures — LSTM & GRU — trained on the timeless prose of Project Gutenberg to learn the rhythm and flow of the English language.**

[📓 LSTM Notebook](#) · [📓 GRU Notebook](#) · [📊 Results](#model-comparison) · [🚀 Quick Start](#quick-start)

</div>

---

## 🧠 What This Project Does

Given a sequence of words, the model predicts the most likely **next word** — just like autocomplete on your phone, but trained on centuries of classic literature instead of social media.

```
Input:  "It was the best of times, it was the worst of"
Output: "times"  ✅  (89% confidence)
```

Two architectures are trained and compared head-to-head:

| Model | Accuracy | Parameters | Training Speed |
|-------|----------|------------|----------------|
| **LSTM** | ~74% | Higher | Slower |
| **GRU** ⭐ | **76.50%** | Lower | **Faster** |

> GRU wins — fewer parameters, faster to train, and better accuracy on this task.

---

## 📊 Model Comparison

### LSTM — Long Short-Term Memory
Uses **three gates** (input, forget, output) plus a dedicated **cell state** to carry information across long sequences. Powerful for capturing long-range dependencies, but heavier to train.

```
Input → Embedding → LSTM Layer → Dense (Softmax) → Next Word
```

### GRU — Gated Recurrent Unit ⭐ Best Model
Combines the forget and input gates into a single **update gate**, plus a **reset gate**. Fewer parameters than LSTM, trains faster, and achieves higher accuracy on this dataset.

```
Input → Embedding → GRU Layer → Dense (Softmax) → Next Word
```

**Why GRU won here:** Project Gutenberg texts have long but predictable structure. GRU's streamlined gating is enough to capture these patterns without LSTM's extra complexity.

---

## 🏗️ How It Works

```
Raw Text  (Project Gutenberg corpus)
    │
    ▼
Tokenization  →  Every word mapped to a unique integer index
    │
    ▼
N-gram Sequences  →  Sliding window creates (context → next_word) pairs
    │
    ▼
Padding  →  All sequences padded to equal length
    │
    ▼
Embedding Layer  →  Word indices → dense 100-dim vectors
    │
    ├──── LSTM Branch:  LSTM(150) → Dropout → Dense(vocab, softmax)
    └────  GRU Branch:   GRU(150) → Dropout → Dense(vocab, softmax)
    │
    ▼
Prediction  →  argmax over vocab → next word
```

---

## 📚 Dataset — Project Gutenberg

[Project Gutenberg](https://www.gutenberg.org/) is a free digital library of **60,000+ classic literary works** whose copyright has expired. Novels, essays, philosophy — all in rich, complex English prose.

Why this dataset is interesting for NLP:
- **Long sentences** with complex grammar push RNNs to develop real memory
- **Rare vocabulary** forces a richer embedding space
- **Varied authors** (Dickens, Austen, Tolstoy, etc.) create diverse language patterns
- **No modern slang** — the model learns proper literary English

---

## ⚙️ Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/aashutoshkumarbhardwaj/Next_word_predict_Lstm_GRU.git
cd Next_word_predict_Lstm_GRU
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run inference**
```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
model = load_model('model_gru.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def predict_next_word(seed_text, model, tokenizer, max_seq_len=20):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word

seed = "It was the best of times it was the worst of"
print(predict_next_word(seed, model, tokenizer))
# → "times"
```

---

## 📁 Project Structure

```
Next_word_predict_Lstm_GRU/
├── next_word_lstm.ipynb      # LSTM: training, evaluation, plots
├── next_word_gru.ipynb       # GRU: training, evaluation, plots
├── model_lstm.h5             # Trained LSTM model weights
├── model_gru.h5              # Trained GRU model weights  ← use this one
├── tokenizer.pkl             # Fitted Keras tokenizer
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 📈 Training Results

<details>
<summary><b>Accuracy & loss curves (click to expand)</b></summary>

```
Add your matplotlib training plots here.
Save as: assets/lstm_training.png and assets/gru_training.png
```

| Epoch | LSTM Acc | GRU Acc |
|-------|----------|---------|
| 10    | ~45%     | ~47%    |
| 30    | ~62%     | ~65%    |
| 50    | ~71%     | ~73%    |
| 80    | ~74%     | **76.50%** |

</details>

---

## 🔥 Roadmap

- [x] LSTM next-word prediction model
- [x] GRU next-word prediction model
- [x] Trained on Project Gutenberg corpus
- [ ] Streamlit app — type a seed, watch it autocomplete
- [ ] Bidirectional RNN for richer context
- [ ] Temperature sampling — control creativity vs accuracy
- [ ] Beam search decoding — generate full sentences
- [ ] Transformer / GPT-2 upgrade

---

## 🧪 Key Learnings

- How RNNs handle **sequential text data** through time steps
- Difference between **LSTM cell state** and **GRU hidden state**
- Why **GRU can outperform LSTM** with fewer parameters on shorter sequences
- How **n-gram windowing** creates supervised training pairs from raw text
- The challenge of **softmax over large vocabularies** in language models

---

## 🤝 Contributing

Ideas worth contributing: Streamlit demo, beam search, BLEU score evaluation, temperature sampling, or a Transformer comparison.

```bash
git checkout -b feature/your-feature
git commit -m "feat: add temperature sampling"
git push origin feature/your-feature
# Open a Pull Request
```

---

## 👨‍💻 Author

**Aashutosh Kumar Bhardwaj** — AI/ML Developer · Open Source Contributor

If this helped you understand LSTMs or GRUs, a ⭐ star goes further than you think.

---

<div align="center">
<sub>Built with TensorFlow · Keras · Project Gutenberg · MIT License</sub>
</div>
