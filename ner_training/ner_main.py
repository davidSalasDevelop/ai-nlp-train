# ner_main.py
import logging, os
from datetime import datetime
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
import mlflow

import ner_config
from ner_data_loader import load_and_prepare_ner_data
from ner_model import build_ner_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main_ner_pipeline():
    logging.info("🚀 INICIANDO PIPELINE DE ENTRENAMIENTO DEL MODELO NER (Autocontenido) 🚀")
    
    start_time = datetime.now()
    os.environ['MLFLOW_TRACKING_URI'] = ner_config.MLFLOW_TRACKING_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = ner_config.MLFLOW_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = ner_config.MLFLOW_PASSWORD
    experiment_name = f"GetNews-Extractor-Training-{start_time.strftime('%Y%m%d_%H%M%S')}"
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
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
        model=model, args=training_args, train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"], tokenizer=tokenizer,
        data_collator=data_collator, compute_metrics=compute_metrics,
    )

    logging.info("🔥 Iniciando entrenamiento del modelo NER...")
    trainer.train()
    
    logging.info(f"💾 Guardando el mejor modelo NER en '{ner_config.NER_MODEL_OUTPUT_DIR}'...")
    trainer.save_model(ner_config.NER_MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(ner_config.NER_MODEL_OUTPUT_DIR)
    logging.info("🎉 PIPELINE NER COMPLETADO 🎉")

if __name__ == "__main__":
    main_ner_pipeline()