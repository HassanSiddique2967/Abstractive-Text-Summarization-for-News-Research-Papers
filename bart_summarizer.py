from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Path to the fine-tuned BART model
path = "./bart_finetuned_model"

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained(path)
model = BartForConditionalGeneration.from_pretrained(path).to(device)

def summarize_bart(text):
    try:
        # Tokenize input
        inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(device)

        # Generate summary with beam search
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode and return summary
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        return f"Error generating summary: {str(e)}"
