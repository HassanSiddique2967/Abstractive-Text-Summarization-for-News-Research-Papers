from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

# Path to the fine-tuned Pegasus model
path = "./pegasus_finetuned_all/checkpoint-500"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained(path)
model = PegasusForConditionalGeneration.from_pretrained(path).to(device)

# Max input tokens Pegasus can handle
MAX_INPUT_LENGTH = 512

def summarize_pegasus(text):
    try:
        # Truncate raw input if it’s too long before tokenizing
        tokens = tokenizer.tokenize(text)
        if len(tokens) > MAX_INPUT_LENGTH:
            tokens = tokens[:MAX_INPUT_LENGTH]
            text = tokenizer.convert_tokens_to_string(tokens)

        # Tokenize for model input
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH).to(device)

        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode and return
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        return f"Error during Pegasus summarization: {str(e)}"
