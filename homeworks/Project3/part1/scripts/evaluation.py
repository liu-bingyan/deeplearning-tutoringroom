import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_preparation import prepare_data, setup_device
from evaluate import load
import numpy as np

def load_fine_tuned_model(base_model_name, fine_tuned_path):
    """Load the fine-tuned model"""
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, fine_tuned_path)
    return model

def generate_response(model, tokenizer, prompt, max_length=100):
    """Generate a response for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def evaluate_model(model, tokenizer, eval_dataloader, metric_name="rouge"):
    """Evaluate the model using the specified metric"""
    metric = load(metric_name)
    all_predictions = []
    all_references = []
    
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            # Generate predictions
            prompts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            predictions = [generate_response(model, tokenizer, prompt) for prompt in prompts]
            
            # Get references (assuming they're in the dataset)
            references = batch.get("labels", [""] * len(predictions))
            if isinstance(references, torch.Tensor):
                references = tokenizer.batch_decode(references, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # Calculate metrics
    results = metric.compute(predictions=all_predictions, references=all_references)
    return results

def main():
    # Setup
    device = setup_device()
    base_model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your model
    fine_tuned_path = "fine_tuned_model"
    
    # Load model and tokenizer
    model = load_fine_tuned_model(base_model_name, fine_tuned_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Prepare evaluation data
    _, eval_dataloader, _ = prepare_data(base_model_name)
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, eval_dataloader)
    print("Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main() 