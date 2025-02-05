import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Step 1: Load Dataset
dataset = load_dataset("yahma/alpaca-cleaned")

def format_prompt(example):
    return {
        "prompt": f"### Instruction: {example['instruction']}\n\n### Input: {example['input']}\n\n### Response: {example['output']}",
        "response": example["output"]
    }

dataset = dataset.map(format_prompt)

# Step 2: Load Model with 4-bit Quantization
model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Apply LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, lora_config)

# Step 4: Define Training Arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_steps=100,
    max_steps=1000,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="./results",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# Step 5: Train Model
trainer.train()

# Step 6: Save Model
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Fine-tuning complete. Model saved!")
