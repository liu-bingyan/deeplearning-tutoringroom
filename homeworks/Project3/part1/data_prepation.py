# load dataset
from datasets import Dataset, load_dataset
eval_filename='eval_dataset.json'
eval_dataset = load_dataset("json",data_files=eval_filename)['train']

# tokenize data
def tokenize_prompts(dataset, tokenizer, max_length=512):
    def tokenize(example):
        return tokenizer(
            example["prompt"],
            truncation=True,
            max_length=max_length,
        )
    return dataset.map(tokenize, batched=True)

# wrap dataset with dataloader for batch inference
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
def create_dataloader(tokenized_dataset, tokenizer, batch_size=8):
    # your code begins
    pass
    # your code ends

# test if your data loader is successfully created
# Don't change the code below:
from transformers import AutoTokenizer
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenized_eval_dataset = tokenize_prompts(eval_dataset, tokenizer)
eval_dataloader = create_dataloader(tokenized_eval_dataset, tokenizer, batch_size=8)
total_examples = 0
total_batches = len(eval_dataloader)
for i, batch in enumerate(eval_dataloader):
    if i != total_batches - 1:
      assert batch["input_ids"].shape[0] == 8
      assert batch["attention_mask"].shape[0] == 8
    total_examples += batch["input_ids"].shape[0]
assert total_examples == len(eval_dataset)
print("Dataloader successfully created! well done!")
# Don't change the code above