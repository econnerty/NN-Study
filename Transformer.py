# Erik Connerty
# 6/24/2023
# USC - AI institute
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load and process dataset
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(data):
    result = tokenizer(data["text"], padding="max_length", truncation=True, max_length=128)
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./TrainedModels/GPT2_TrainedModels",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # batch size should be as large as you can afford on your hardware
    save_steps=10_000,
    save_total_limit=2,
    optim="adamw_torch",
)

# Define trainer
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()

model.resize_token_embeddings(len(tokenizer))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./TrainedModels/GPT2_TrainedModels")
tokenizer.save_pretrained("./TrainedModels/GPT2_TrainedModels")
