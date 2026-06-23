from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class BartSummarizer:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only = True
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            local_files_only = True
        ).to(self.device)


    def summarize(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        summary_ids = self.model.generate(
            **inputs,
            max_length=25,
            min_length=20,
            num_beams=4,
            early_stopping=True
        )

        return self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

summarizer = BartSummarizer("./models/bart_summarizer")