#  LLM Guardrail & Prompt Injection Classifier

This repository contains the implementation for **Section 2: Text Classification Model Building & Deployment**. The project implements an active **AI Security Firewall** that analyzes user prompts in real time to detect and block **Prompt Injections** and **Jailbreak Attacks** targeting Large Language Models.

---

##  Project Structure

```text
LLM Prompt Injector Detection/
│
├── src/
│   └── text_classifier/                # Part 2: Prompt Injection Classifier
│       ├── data/
│       │   └── prompt_injection_dataset.csv  # Downloaded HuggingFace dataset
│       ├── models/
│       │   ├── model.pkl               # Exported best-performing model
│       │   └── tfidf_vectorizer.pkl     # Exported TF-IDF vectorizer
│       ├── download_data.py            # Script to download dataset automatically
│       ├── text_classification.ipynb   # EDA, Training, Cross-Validation & Tuning
│       └── app.py                      # Streamlit deployment web app
│
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation & execution instructions

```

---

##  Setup Instructions

### 1. Prerequisites

Ensure you have **Python 3.9** or higher installed on your system.

### 2. Environment Setup & Dependencies Installation

Open your terminal or PowerShell in the root directory of this project and run:

```bash
pip install -r requirements.txt

```

---

##  How to Run the Application

### Option A: Launch the Streamlit Web Application directly (Pre-trained)

The pre-trained model artifacts (`model.pkl` and `tfidf_vectorizer.pkl`) are included in `src/text_classifier/models/`. You can immediately start the Streamlit firewall application using:

```bash
python -m streamlit run src/text_classifier/app.py

```

> **Note:** Executing via `python -m streamlit` ensures Streamlit runs using the correct active Python environment across Windows, macOS, and Linux.

---

### Option B: Re-downloading Data & Training Models from Scratch (Optional)

If you wish to re-execute the complete training pipeline and evaluate the models:

1. **(Optional) Re-download Dataset:**
```bash
python src/text_classifier/download_data.py

```


2. **Run Model Training Notebook:**
Open and execute all cells in `src/text_classifier/text_classification.ipynb`.

This notebook performs:

* **Exploratory Data Analysis (EDA):** Class distribution, text length distributions, and N-gram analysis.
* **Text Preprocessing & Vectorization:** Cleaning text and extracting TF-IDF features (`TfidfVectorizer`).
* **Model Evaluation:** Compares Multinomial Naive Bayes, Logistic Regression, Support Vector Machine (SVM), and Random Forest.
* **Hyperparameter Tuning:** Evaluates cross-validation performance to select the best model.
* **Artifact Export:** Exports the final model and vectorizer into `src/text_classifier/models/`.

```

```