# +
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DetoxModel():
    def __init__(self) -> None:
        self.label_map = {0: "Non-toxic", 1: "Toxic"}
        self.model_name = "unitary/multilingual-toxic-xlm-roberta"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def classify_toxicity(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
        return probabilities.item()
    
    def get_label(self, text: str) -> str:
        probability = self.classify_toxicity(text)
        prediction = 1 if probability > 0.5 else 0
        class_label = self.label_map[prediction]
        return class_label
