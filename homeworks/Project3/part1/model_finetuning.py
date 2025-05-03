sft_filename='sft_dataset.json'
train_dataset = load_dataset("json", data_files=sft_filename)['train']

############################# your code begins ##############################
# your code begins
from transformers import DataCollatorForLanguageModeling

def tokenize_sft_dataset(dataset, tokenizer, max_length=512):
    def tokenize(example):
        prompt = example["prompt"].strip()
        response = example["response"].strip()
        merged = prompt + "\n" + response
        tokenized = tokenizer(merged, truncation=True, padding=True, max_length=max_length)

        # Now compute loss mask
        prompt_tokenized = tokenizer(prompt, truncation=True, max_length=max_length)
        prompt_len = len(prompt_tokenized["input_ids"])

        # Mask out the prompt
        tokenized["labels"] = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
        return tokenized

    tokenized = dataset.map(tokenize, batched=False)
    return tokenized.remove_columns([col for col in tokenized.column_names if col not in tokenizer.model_input_names + ["labels"]])

# def tokenize_sft_dataset(dataset, tokenizer, max_length=512):
#     def tokenize(example):
#         merged = example["prompt"] + "\n" + example["response"]
#         return tokenizer(merged, truncation=True, max_length=max_length)
    
#     tokenized = dataset.map(tokenize, batched=False)
#     return tokenized.remove_columns([col for col in tokenized.column_names if col not in tokenizer.model_input_names])

tokenized_train_dataset = tokenize_sft_dataset(train_dataset, tokenizer)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
base_model = prepare_model_for_kbit_training(base_model)  # if using quantized model
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
ft_model = get_peft_model(base_model, lora_config)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora-llama3",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=100,
    save_total_limit=1,
    save_strategy="epoch",
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # important: use causal language modeling
)
trainer = Trainer(
    model=ft_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()



############################# your code ends ##############################

# don't edit this cell
from tqdm import tqdm
predictions_finetune = []
references = [example["response"].strip() for example in eval_dataset]

for batch in tqdm(eval_dataloader):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    batch_outputs = generate_batch_responses(
        input_ids, attention_mask, ft_model, tokenizer
    )

    predictions_finetune.extend(batch_outputs)

# don't edit this cell
bleu_result_finetune = bleu.compute(predictions=predictions_finetune, references=[[r] for r in references])
rouge_result_finetune = rouge.compute(predictions=predictions_finetune, references=references)
bertscore_result_finetune = bertscore.compute(predictions=predictions_finetune, references=references, lang="en")
print(f"BLEU: {bleu_result_finetune['bleu'] * 100:.2f}")
print(f"ROUGE-L: {rouge_result_finetune['rougeL'] * 100:.2f}")
print(f"BERTScore (F1): {sum(bertscore_result_finetune['f1']) / len(bertscore_result_finetune['f1']) * 100:.2f}")

assert bleu_result_finetune['bleu'] > bleu_result['bleu']
assert rouge_result_finetune['rougeL'] > rouge_result['rougeL']
assert sum(bertscore_result_finetune['f1']) / len(bertscore_result_finetune['f1']) > sum(bertscore_result['f1']) / len(bertscore_result['f1'])
print("Model performance improved on all metrics! Well done!")