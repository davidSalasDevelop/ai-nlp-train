# ner_main.py
import logging
import os
from pathlib import Path
import torch
from datetime import datetime
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
import mlflow
from mlflow.tracking import MlflowClient

import ner_config
from ner_data_loader import load_and_prepare_ner_data
from ner_model import build_ner_model
from ner_callbacks import SystemMetricsCallback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main_ner_pipeline():
    logging.info("🚀 INICIANDO PIPELINE DE ENTRENAMIENTO DEL MODELO NER (Autocontenido) 🚀")
    
    start_time = datetime.now()
    os.environ['MLFLOW_TRACKING_URI'] = ner_config.MLFLOW_TRACKING_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = ner_config.MLFLOW_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = ner_config.MLFLOW_PASSWORD
    
    experiment_name = "Parameters-Extractor-Training"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    elif experiment.lifecycle_stage == 'deleted':
        client.restore_experiment(experiment.experiment_id)
    mlflow.set_experiment(experiment_name)
    logging.info(f"🔧 MLflow configurado para el experimento: '{experiment_name}'")

    labels = ner_config.ENTITY_LABELS
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(ner_config.MODEL_NAME)
    tokenized_datasets = load_and_prepare_ner_data(ner_config.DATASET_PATH, tokenizer, label2id)
    
    model = build_ner_model(ner_config.MODEL_NAME, id2label, label2id)

    training_args = TrainingArguments(
        output_dir="./ner_results",
        run_name=f"ner-extractor-run-{start_time.strftime('%H%M')}",
        num_train_epochs=ner_config.TRAIN_EPOCHS,
        per_device_train_batch_size=ner_config.BATCH_SIZE,
        per_device_eval_batch_size=ner_config.BATCH_SIZE,
        learning_rate=ner_config.LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        weight_decay=0.01,
        logging_strategy="epoch",
        report_to="mlflow",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [[id2label[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[id2label[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        return {"precision": precision_score(true_labels, true_predictions), "recall": recall_score(true_labels, true_predictions), "f1": f1_score(true_labels, true_predictions)}

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        # --- CORRECCIÓN: Se quita el tokenizer, ya está en el data_collator ---
        # tokenizer=tokenizer, 
        data_collator=data_collator, 
        compute_metrics=compute_metrics,
        callbacks=[SystemMetricsCallback()],
    )

    logging.info("🔥 Iniciando entrenamiento del modelo NER...")
    trainer.train()
    
    logging.info("💾 Creando archivo .pt autocontenido para el modelo NER...")
    
    best_model = trainer.model
    
    checkpoint = {
        'model_state_dict': best_model.state_dict(),
        'config': best_model.config.to_dict(),
        'tokenizer_name': ner_config.MODEL_NAME,
        'id2label': id2label,
        'label2id': label2id
    }
    
    output_dir = Path(ner_config.NER_MODEL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pt_file_path = output_dir / "get_news_extractor.pt"
    
    torch.save(checkpoint, pt_file_path)
    logging.info(f"   ✅ Checkpoint .pt guardado en '{pt_file_path}'")
    
    logging.info("🎉 PIPELINE NER COMPLETADO 🎉")

if __name__ == "__main__":
    main_ner_pipeline()