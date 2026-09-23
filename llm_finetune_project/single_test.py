import os
from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_name = "prajjwal1/bert-mini"
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentence = "Services: Smart grid monitoring, renewable energy consultations. Security Assets: Firewalls, intrusion detection systems, encryption tools. Tech critical analytics. nical technical and Disaster Recovery Systems: Energy data backups, contingency plans for grid failures. Non Technology: Green energy infrastructure, opera hardware. Servers: Physical and virtual energy Assets non technical monitoring servers. assets. Compliance Systems: Environmental regulations compliance monitoring. Software: Energy management systems, predictive maintenance software. Networking and Infrastructure: IoT Development Tools: Renewable energy simulation tools, CI/CD pipelines. Hosting Services: Cloud platforms for energy Non People: Energy engineers, customer support sta Intellectual Property (IP): Proprietary renewable designs, algorithms for efficiency. Data: Energy production data, grid performance data, customer usage analytics. Business Processes: Energy production plannin regulatory reporting. connected energy devices, routers, secure networks. 1 Critical Solutions Models: Energy load forecasting, renewable resource optimization. Databases: Solar, wind, and hydroelectric performance databases. Communication Systems: Secure communication tools, collaboration platforms. Data Centres: Renewable energy data Technical should repositories. and identify their"
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
model_path = os.path.join(BASE_DIR, "bert-mini-finetuned")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer
)
device = trainer.model.device  # Automatically use the same device as model
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = trainer.model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()

id2label = {0: "bad", 1: "good", 2: "excellent"}
print("Prediction:", id2label[predicted_class_id])