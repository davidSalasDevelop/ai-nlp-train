# main.py
"""
Main pipeline orchestrator using the Hugging Face Trainer API.
"""
import os
import json
from datetime import datetime
import mlflow
# <-- MODIFIED: Import DataCollatorWithPadding
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

# Import from our modules
import config
from data_loader import load_and_prepare_data
from model import TinyModel
from callbacks import SystemMetricsCallback

def main_pipeline():
    """Orchestrates the training pipeline using the Hugging Face Trainer."""
    start_time = datetime.now()
    print("="*70)
    print("🚀 STARTING HUGGING FACE TRAINER PIPELINE (WITH SYSTEM MONITORING) 🚀")
    print("="*70)

    # --- 1. Setup MLflow ---
    os.environ['MLFLOW_TRACKING_URI'] = config.MLFLOW_TRACKING_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.MLFLOW_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.MLFLOW_PASSWORD
    experiment_name = f"Intent-TrainerAPI-"
    mlflow.set_experiment(experiment_name)
    print(f"🔧 MLflow configured for experiment: '{experiment_name}'")

    # --- 2. Load Tokenizer and Data ---
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenized_datasets, id_to_intent = load_and_prepare_data(tokenizer, config.DATASET_PATH)
    num_labels = len(id_to_intent)

    # <-- MODIFIED: Create a data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 3. Load Model ---
    print(f"\n🧠 Initializing model '{config.MODEL_NAME}' for {num_labels} labels...")
    model = TinyModel(model_name=config.MODEL_NAME, num_labels=num_labels)

    # --- 4. Define Training Arguments ---
    training_args = TrainingArguments(
        output_dir="./results",
        run_name=f"run-{start_time.strftime('%H%M')}",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_dir='./logs',
        logging_strategy="epoch",
        report_to="mlflow",
    )

    # --- 5. Instantiate the Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,  # <-- MODIFIED: Use the data_collator instead of tokenizer
        callbacks=[SystemMetricsCallback()],
    )

    # --- 6. Train the Model ---
    print("\n🔥 Starting training...")
    trainer.train()
    print("✅ Training complete!")

    # --- 7. Save the Best Model ---
    print(f"\n💾 Saving the best model to '{config.FINAL_MODEL_OUTPUT_DIR}'...")
    trainer.save_model(config.FINAL_MODEL_OUTPUT_DIR)
    
    with open(os.path.join(config.FINAL_MODEL_OUTPUT_DIR, 'id_to_intent.json'), 'w') as f:
        json.dump(id_to_intent, f)

    print("="*70)
    print("🎉 PIPELINE COMPLETE 🎉")
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"⏱️ Total time: {total_time/60:.2f} minutes")
    print(f"📊 Check MLflow for detailed metrics: {config.MLFLOW_TRACKING_URI}")
    print("="*70)

if __name__ == "__main__":
    main_pipeline()