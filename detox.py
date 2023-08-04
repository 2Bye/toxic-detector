# +
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class FilterModel():
    def __init__(self) -> None:
        self.toxic_label_map = ["Non-toxic", "Toxic"]
        self.adult_content_label_map = ["Non-sex", "Sex"]

        self.toxicity_model_name = "unitary/multilingual-toxic-xlm-roberta"
        self.adult_content_model_name = "ziadA123/autotrain-adult-classification-3642997339"

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
            probabilities_sigmoid, probabilities_argmax = torch.sigmoid(logits), torch.argmax(logits)
        return probabilities_sigmoid.tolist(), probabilities_argmax
    
    def get_label(self, text: str) -> list:
        toxic_probability = self.classify_toxicity(text)
        toxic_prediction = 1 if toxic_probability > 0.65 else 0
        toxic_class_label = self.toxic_label_map[toxic_prediction]

        adult_content_probability_sigmoid, adult_content_probability_argmax = self.classify_adult_content(text)
        adult_content_label = 1 if adult_content_probability_sigmoid[0][1] > 0.75 else 0
        adult_content_class_label = self.adult_content_label_map[adult_content_label]
        
        print('')
        print((f'// Input Text\t\t\t\t| {text}'))
        print('_' * 10)
        print('// Toxic outputs\t\t\t|', 'Hugging Face model')
        print('// Toxic probability\t\t\t|', toxic_probability)
        print('// Result\t\t\t\t|', toxic_class_label)
        print('_' * 10)
        print('// Adult outputs\t\t\t|', 'Hugging Face model')
        print('// ArgMax Function\t\t\t|', adult_content_probability_argmax)
        print('// Sigmoid Function\t\t\t|', adult_content_probability_sigmoid)
        print('// Result\t\t\t\t|', adult_content_class_label)
        print('')
        return [toxic_class_label, adult_content_class_label]
