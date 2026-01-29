# main.py
"""
Main pipeline orchestrator using the Hugging Face Trainer API.
Saves ONLY the final .pt file of the best performing model.
"""
import os
import json
import torch
import copy
from datetime import datetime
import mlflow
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding,
    TrainerCallback,
    TrainerState,
    TrainerControl
)

# Import from our modules
import config
from data_loader import load_and_prepare_data
from model import TinyModel
from callbacks import SystemMetricsCallback

# --- NEW: Custom callback to find and restore the best model ---
class BestModelCallback(TrainerCallback):
    """
    A custom callback that monitors the evaluation loss and saves the best
    model's state dictionary in memory. At the end of training, it loads
    this best state back into the model.
    """
    def __init__(self):
        super().__init__()
        self.best_eval_loss = float('inf')
        self.best_model_state_dict = None

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """Event called after every evaluation."""
        current_eval_loss = metrics.get("eval_loss")
        if current_eval_loss is not None and current_eval_loss < self.best_eval_loss:
            self.best_eval_loss = current_eval_loss
            # Use deepcopy to ensure we're getting a snapshot of the weights
            self.best_model_state_dict = copy.deepcopy(kwargs['model'].state_dict())
            print(f"\n[BestModelCallback] New best model found! Eval loss: {self.best_eval_loss:.4f}. State saved in memory.")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Event called at the end of training."""
        if self.best_model_state_dict:
            print("\n[BestModelCallback] Training finished. Loading best model state for final save.")
            # Load the best weights back into the model
            kwargs['model'].load_state_dict(self.best_model_state_dict)

def main_pipeline():
    """Orchestrates the training pipeline using the Hugging Face Trainer."""
    start_time = datetime.now()
    print("="*70)
    print("🚀 STARTING HUGGING FACE TRAINER PIPELINE (CLEAN .PT SAVE ONLY) 🚀")
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
        # We must evaluate at each epoch to find the best model
        eval_strategy="epoch",
        # --- KEY CHANGE: Disable the Trainer's automatic saving ---
        save_strategy="no",
        # `load_best_model_at_end` must be False when save_strategy is "no"
        logging_dir='./logs',
        logging_strategy="epoch",
        report_to="mlflow",
    )

    # --- 5. Instantiate the Trainer with our custom callback ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        callbacks=[SystemMetricsCallback(), BestModelCallback()], # Add our callback here
    )

    # --- 6. Train the Model ---
    print("\n🔥 Starting training...")
    trainer.train()
    print("✅ Training complete!")

    # --- 7. Save ONLY the final .pt file ---
    print(f"\n💾 Creating self-contained .pt checkpoint file...")
    
    # The BestModelCallback has already loaded the best model weights into trainer.model
    best_model = trainer.model
    
    # Final check for tensor contiguity to prevent saving errors
    for name, param in best_model.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    checkpoint = {
        'model_state_dict': best_model.state_dict(),
        'config': best_model.config,
        'id_to_intent': id_to_intent,
        'tokenizer_name': config.MODEL_NAME
    }
    
    # Ensure the final output directory exists
    os.makedirs(config.FINAL_MODEL_OUTPUT_DIR, exist_ok=True)
    
    pt_file_path = os.path.join(config.FINAL_MODEL_OUTPUT_DIR, config.OUTPUT_MODEL_NAME)
    torch.save(checkpoint, pt_file_path)
    print(f"   ✅ Clean checkpoint saved to '{pt_file_path}'")
    
    print("="*70)
    print("🎉 PIPELINE COMPLETE 🎉")
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"⏱️ Total time: {total_time/60:.2f} minutes")
    print(f"📊 Check MLflow for detailed metrics: {config.MLFLOW_TRACKING_URI}")
    print("="*70)

if __name__ == "__main__":
    main_pipeline()