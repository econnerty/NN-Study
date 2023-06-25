# Erik Connerty
# 6/24/2023
# USC - AI institute
from transformers import pipeline

pred_model = pipeline("fill-mask", model = "TrainedModels/GPT2_TrainedModels")

text = "This is an [MASK] movie."

preds = pred_model(text)

for pred in preds:
    print(f">>> {pred['sequence']}")


