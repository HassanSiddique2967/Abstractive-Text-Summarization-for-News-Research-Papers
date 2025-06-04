from transformers import T5Tokenizer, T5ForConditionalGeneration

# Paths to fine-tuned T5 model directories
MODEL_PATHS = {
    "cnn": "E:/Courses/Deep Learning/Deep_Project/t5_cnn_dailymail_finetuned",
    "pubmed": "E:/Courses/Deep Learning/Deep_Project/t5_pubmed_finetuned",
    "xsum": "E:/Courses/Deep Learning/Deep_Project/t5_xsum_finetuned"
}


# Load models into memory
models = {name: T5ForConditionalGeneration.from_pretrained(path) for name, path in MODEL_PATHS.items()}
tokenizer = T5Tokenizer.from_pretrained("t5-base")  # Single tokenizer instance

def summarize_t5(article, model_choice="cnn"):
    # Ensure model_choice is valid
    if model_choice not in models:
        raise ValueError(f"Invalid model_choice '{model_choice}'. Must be one of {list(models.keys())}.")

    model = models[model_choice]
    
    input_text = "summarize: " + article.strip()
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary
