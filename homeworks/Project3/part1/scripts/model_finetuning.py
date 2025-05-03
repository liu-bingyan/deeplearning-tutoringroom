import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from data_preparation import prepare_data, setup_device

def setup_model(model_name, device):
    """Initialize the base model and move it to the specified device"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model

def setup_lora(model, r=8, lora_alpha=32, lora_dropout=0.1):
    """Configure and apply LoRA to the model"""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def train_model(model, train_dataloader, tokenizer, num_epochs=3, learning_rate=2e-4):
    """Train the model using the provided dataloader"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def main():
    # Setup
    device = setup_device()
    model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your model
    
    # Prepare data
    tokenized_dataset, train_dataloader, tokenizer = prepare_data(model_name)
    
    # Setup model
    model = setup_model(model_name, device)
    
    # Apply LoRA
    model = setup_lora(model)
    
    # Train model
    train_model(model, train_dataloader, tokenizer)
    
    # Save the model
    model.save_pretrained("fine_tuned_model")
    print("Model training completed and saved!")

if __name__ == "__main__":
    main() 