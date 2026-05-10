import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


def main():

    # -------------------------
    # Model
    # -------------------------

    model_name = "microsoft/Phi-3-mini-4k-instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True          # FIX: required for Phi-3
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True          # FIX: required for Phi-3
    )
    tokenizer.pad_token = tokenizer.eos_token

    # FIX: set use_cache=False BEFORE prepare + get_peft_model
    model.config.use_cache = False

    # IMPORTANT QLoRA FIX
    model = prepare_model_for_kbit_training(model)

    # FIX: enable gradient checkpointing BEFORE get_peft_model
    model.gradient_checkpointing_enable()

    # -------------------------
    # LoRA Config
    # -------------------------

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # FIX: Phi-3-mini uses fused projections — correct module names
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    print("LoRA adapters added")
    model.print_trainable_parameters()  # helpful sanity check

    # -------------------------
    # Dataset
    # -------------------------

    dataset = load_dataset(
        "json",
        data_files={
            "train": "../data/processed/train.json",
            "validation": "../data/processed/validation.json"
        }
    )

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    dataset = dataset.map(tokenize, batched=True)

    collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False
    )

    # -------------------------
    # Training Arguments
    # -------------------------

    training_args = TrainingArguments(

        output_dir="../models/phi3-medical",

        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,

        num_train_epochs=2,

        learning_rate=2e-4,

        fp16=True,

        logging_steps=50,

        # FIX: evaluation_strategy is deprecated → use eval_strategy
        eval_strategy="steps",
        eval_steps=200,

        save_steps=200,

        dataloader_num_workers=0,       # Windows fix

        save_total_limit=2,

        report_to="none",

        # FIX: handle gradient checkpointing via TrainingArguments (cleaner)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # -------------------------
    # Trainer
    # -------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=dataset["train"],

        eval_dataset=dataset["validation"],

        data_collator=collator
    )

    print("Starting training...")

    trainer.train()

    print("Training completed!")

    trainer.save_model("../models/phi3-medical")


if __name__ == "__main__":
    main()