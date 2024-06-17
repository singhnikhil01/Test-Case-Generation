import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser,
    TrainingArguments, pipeline, logging, TrainerCallback
)
from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from torchinfo import summary

# Load dataset
dataset = load_dataset("deepmind/code_contests")

def create_prompt(data):
    return {
        'input_text': f"{data['name']} {data['description']} {data['public_tests']}",
        'target_text': data['generated_tests']
    }

original_columns = dataset['train'].column_names
dataset = dataset.map(create_prompt, remove_columns=original_columns, batched=True)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model = model.half()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Model summary
summary(model)

def generate_output(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, 
                                    max_length=512,
                                    num_return_sequences=1,
                                    attention_mask=input_ids.ne(tokenizer.pad_token_id),
                                    pad_token_id=tokenizer.pad_token_id,
                                    temperature=0.8)
    generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_output

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

class SaveLogsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        epoch = state.epoch
        logs_dir = os.path.join(args.logging_dir, f"epoch_{epoch}_logs.txt")
        with open(logs_dir, "w") as f:
            f.write(str(logs))

def tokenize_function(examples):
    inputs = tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(examples['target_text'], max_length=512, truncation=True, padding="max_length")
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": targets["input_ids"]}

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['valid']
test_dataset = tokenized_datasets['test']

args = TrainingArguments(
    output_dir="./codellama_testcase_generation",
    num_train_epochs=20,
    per_device_train_batch_size=4,
    warmup_steps=0.03,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=20,
    learning_rate=2e-4,
    lr_scheduler_type='constant',
    logging_dir="./logs_generation",
)

max_seq_length = None
trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[SaveLogsCallback()]
)
trainer.train()
trainer.model.save_pretrained("./code_lamma_testcase_generation")
tokenizer.save_pretrained("./code_lamma_testcase_generation")

results = trainer.evaluate()
eval_results_path = "./code_lamma_testcase_generation/evaluation_results.txt"
with open(eval_results_path, "w") as f:
    f.write(str(results))
print(f"Evaluation results saved to {eval_results_path}")


test_results_path = "./code_lamma_testcase_generation/test_results.txt"
with open(test_results_path, "w") as f:
    for example in test_dataset:
        prompt = example['input_text']
        generated_output = generate_output(model, tokenizer, prompt)
        result_str = f"Prompt: {prompt}\nGenerated Output: {generated_output}\n\n"
        f.write(result_str)
print(f"Test results saved to {test_results_path}")