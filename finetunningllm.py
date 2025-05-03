import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, PeftModel

def main():
    # --- 1. Initial Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Device: {device} ===")
    print(f"=== GPU Detected: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ===\n")

    # --- 2. Data Loading and Preparation ---
    print("Loading and preparing data...")
    def load_and_prepare_data():
        try:
            # Load English dataset (using wikitext instead of scientific_papers)
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:2000]")
            dataset = dataset.filter(lambda x: len(x["text"]) > 100 and not x["text"].startswith(" ="))  # Filter short texts and headers
            
            # Create train/validation split
            train_val = dataset.train_test_split(test_size=0.2, seed=42)
            return DatasetDict({
                'train': train_val['train'],
                'validation': train_val['test']
            })
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    dataset = load_and_prepare_data()

    # --- 3. Tokenization ---
    print("\nTokenizing data...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],  # Remove original column
        num_proc=1  # Single process for Windows compatibility
    )

    # --- 4. Data Collator ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
        return_tensors="pt"
    )

    # --- 5. Model Configuration with LoRA ---
    print("\nConfiguring model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=32,
        target_modules=["c_attn", "c_proj", "c_fc"],  # GPT-2 attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        fan_in_fan_out=True  # Required for GPT-2
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 6. Optimized Training Configuration ---
    training_args = TrainingArguments(
        output_dir="./gpt2-english-finetuned",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_steps=50,
        save_steps=200,
        learning_rate=5e-5,
        warmup_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        fp16=True,  # Mixed precision training
        remove_unused_columns=False,
        report_to="wandb",
        run_name="gpt2-english-lora-run"
    )

    # --- 7. Trainer Setup ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # --- 8. Training ---
    print("\n=== Starting Training ===")
    trainer.train()

    # --- 9. Model Evaluation ---
    print("\n=== Evaluating Model ===")
    eval_results = trainer.evaluate()
    print(f"\nEvaluation Results:")
    print(f"- Validation Loss: {eval_results['eval_loss']:.4f}")
    print(f"- Evaluation Time: {eval_results['eval_runtime']:.2f}s")
    print(f"- Samples Per Second: {eval_results['eval_samples_per_second']:.2f}")

    # --- 10. Text Generation Test ---
    def generate_text(prompt, max_length=150, temperature=0.7):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_k=50
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n=== Generation Example ===")
    test_prompt = "The treatment for diabetes involves"
    generated_text = generate_text(test_prompt)
    print(f"Prompt: '{test_prompt}'\nGenerated: '{generated_text}'\n")

    # --- 11. Saving the Model ---
    print("Saving model...")
    save_path = "./english_finetuned_gpt2"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved at: {save_path}")

    # --- 12. Loading Model Example ---
    print("\n=== Model Loading Example ===")
    loaded_model = PeftModel.from_pretrained(
        GPT2LMHeadModel.from_pretrained("gpt2").to(device),
        save_path
    )
    print("Model loaded successfully!")

    # Test with loaded model
    loaded_model.eval()
    test_prompt = "Artificial intelligence in healthcare can"
    generated_text = generate_text(test_prompt)
    print(f"\nPrompt: '{test_prompt}'\nGenerated (loaded model): '{generated_text}'")

if __name__ == '__main__':
    main()