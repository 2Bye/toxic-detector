# +
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class FilterModel():
    def __init__(self) -> None:
        self.toxic_label_map = ["Non-toxic", "Toxic"]
        self.adult_content_label_map = ["Non-sex", "Sex"]

        self.toxicity_model_name = "unitary/multilingual-toxic-xlm-roberta"
        self.adult_content_model_name = "ziadA123/adultcontentclassifier"

        self.toxicity_tokenizer = AutoTokenizer.from_pretrained(self.toxicity_model_name)
        self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(self.toxicity_model_name)

        self.adult_content_tokenizer = AutoTokenizer.from_pretrained(self.adult_content_model_name)
        self.adult_content_model = AutoModelForSequenceClassification.from_pretrained(self.adult_content_model_name)

    def classify_toxicity(self, text):
        inputs = self.toxicity_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.toxicity_model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
        return probabilities.item()
    
    def classify_adult_content(self, text):
        inputs = self.adult_content_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.adult_content_model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
        return probabilities.tolist()[0]
    
    def get_label(self, text: str) -> str:
        toxic_probability = self.classify_toxicity(text)
        toxic_prediction = 1 if toxic_probability > 0.5 else 0
        toxic_class_label = self.toxic_label_map[toxic_prediction]

        adult_content_probability = self.classify_adult_content(text)
        adult_content_prediction = 1 if adult_content_probability[0] < 0.5 else 0
        adult_content_class_label = self.adult_content_label_map[adult_content_prediction]

        return [toxic_class_label, adult_content_class_label]
    


