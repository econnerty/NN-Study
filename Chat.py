# Erik Connerty
# 6/24/2023
# USC - AI institute
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./Models/GPT2_TrainedModels")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Create a text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer,pad_token_id = 50256)

# Start a chat prompt
while True:
    prompt = input("You: ")
    response = text_generator(prompt, max_length=200, do_sample=True, temperature=0.7)
    print("Chatbot:", response[0]['generated_text'])
