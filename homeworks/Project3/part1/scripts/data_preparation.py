import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

def setup_device():
    """Set up and return the device (CUDA if available, else CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def load_eval_dataset(filename='eval_dataset.json'):
    """Load the evaluation dataset from a JSON file"""
    dataset = load_dataset("json", data_files=filename)['train']
    return dataset

def tokenize_prompts(dataset, tokenizer, max_length=512):
    """Tokenize the prompts in the dataset"""
    def tokenize(example):
        return tokenizer(
            example["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None
        )
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    return tokenized

def create_dataloader(tokenized_dataset, tokenizer, batch_size=8):
    """Create a DataLoader for batch processing"""
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False  # False for evaluation data
    )
    return loader

def prepare_data(model_name, eval_filename='eval_dataset.json', max_length=512, batch_size=8):
    """Main function to prepare the data for training/evaluation"""
    # Load dataset
    eval_dataset = load_eval_dataset(eval_filename)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize dataset
    tokenized_dataset = tokenize_prompts(eval_dataset, tokenizer, max_length)
    
    # Create dataloader
    dataloader = create_dataloader(tokenized_dataset, tokenizer, batch_size)
    
    return tokenized_dataset, dataloader, tokenizer

if __name__ == "__main__":
    # Example usage
    device = setup_device()
    model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your model
    tokenized_dataset, dataloader, tokenizer = prepare_data(model_name)
    print("Data preparation completed successfully!") 