# Text Summarization App

A web-based text summarization tool built with **Flask** and **Hugging Face Transformers**. Users can input articles and select from multiple fine-tuned models — T5 (CNN/DailyMail, PubMed, XSum), BART, and Pegasus — to generate concise summaries.

---

## Features

- Summarize free-form articles  
- Choose from multiple Transformer models:
  - **T5**
    - CNN/DailyMail
    - PubMed
    - XSum
  - **BART** (trained on all datasets combined)
  - **Pegasus** (trained on all datasets combined)
- Clean, responsive user interface  
- Efficient backend using PyTorch and Transformers

---

## Project Structure

Deep_Project/
├── app.py # Flask app
├── bart_summarizer.py # BART model loader and summarizer
├── pegasus_summarizer.py # Pegasus model loader and summarizer
├── t5_summarizer.py # T5 model loader and summarizer
├── templates/
│ └── index.html # Web frontend
├── static/
│ └── style.css # CSS styles
├── t5_cnn_dailymail_finetuned/ # Fine-tuned T5 CNN model
├── t5_pubmed_finetuned/ # Fine-tuned T5 PubMed model
├── t5_xsum_finetuned/ # Fine-tuned T5 XSum model
├── bart_finetuned_model/ # Fine-tuned BART model
├── pegasus_finetuned_all/ # Fine-tuned Pegasus model

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/HassanSiddique2967/Abstractive-Text-Summarization-for-News-Research-Papers.git
   cd text-summarization-app
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv summarizer
   summarizer\Scripts\activate  # On Windows
   # OR
   source summarizer/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or fine-tune models**  
   Place your fine-tuned models in the following directories:
   - `t5_cnn_dailymail_finetuned/`
   - `t5_pubmed_finetuned/`
   - `t5_xsum_finetuned/`
   - `bart_finetuned_all/`
   - `pegasus_finetuned_all/`

5. **Run the app**
   ```bash
   python app.py
   ```
   The app will start at: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🧪 Model Details

| Model   | Dataset Used  | Description                                                  |
| ------- | ------------- | ------------------------------------------------------------ |
| T5      | CNN/DailyMail | News articles summarization                                  |
| T5      | PubMed        | Scientific/biomedical summarization                          |
| T5      | XSum          | Extreme summarization with a single sentence                 |
| BART    | All datasets  | Combined training for general summarization                  |
| Pegasus | All datasets  | Combined training for high-quality abstractive summarization |

---

## Requirements

- Python 3.8+
- Flask
- torch
- transformers

Refer to `requirements.txt` for the full list.

---

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Pretrained datasets: CNN/DailyMail, PubMed, XSum

---