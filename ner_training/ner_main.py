import logging
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import mlflow
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForTokenClassification,
    EarlyStoppingCallback # <--- Nuevo import para parada temprana
)
from seqeval.metrics import f1_score, precision_score, recall_score

# Importaciones locales
import ner_config
from ner_data_loader import load_and_prepare_ner_data
from ner_callbacks import SystemMetricsCallback

# Configuración de entorno temporal
temp_dir = Path("ner_training/tmp")
temp_dir.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(temp_dir.resolve())
os.environ['TEMP'] = str(temp_dir.resolve())
os.environ['TMP'] = str(temp_dir.resolve())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions)
    }

def main_ner_pipeline():
    logging.info("🚀 INICIANDO PIPELINE NER AVANZADO 🚀")
    
    # 1. Configurar MLflow
    os.environ['MLFLOW_TRACKING_URI'] = ner_config.MLFLOW_TRACKING_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = ner_config.MLFLOW_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = ner_config.MLFLOW_PASSWORD
    
    # Establecer experimento
    mlflow.set_experiment(ner_config.MLFLOW_EXPERIMENT_NAME)

    # 2. Etiquetas
    labels = ner_config.ENTITY_LABELS
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    # 3. Cargar Tokenizer
    logging.info(f"📚 Cargando Tokenizer: {ner_config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(ner_config.MODEL_NAME, cache_dir=ner_config.CACHE_DIR)
    
    final_model = None

    if ner_config.DO_TRAINING:
        logging.info("🏋️  MODO ENTRENAMIENTO ACTIVADO")
        
        # Cargar Datos (Pasando parámetros extra de config)
        tokenized_datasets = load_and_prepare_ner_data(
            ner_config.DATASET_PATH, 
            tokenizer, 
            label_list=labels,
            max_length=ner_config.MAX_LENGTH # Usamos longitud de config
        )
        
        # Cargar Modelo Base
        model = AutoModelForTokenClassification.from_pretrained(
            ner_config.MODEL_NAME, 
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            cache_dir=ner_config.CACHE_DIR
        )
        
        # --- CONFIGURACIÓN AVANZADA DEL TRAINER ---
        training_args = TrainingArguments(
            output_dir="./ner_results",
            run_name=f"ner-{datetime.now().strftime('%Y%m%d-%H%M')}",
            
            # Hiperparámetros Básicos
            num_train_epochs=ner_config.TRAIN_EPOCHS,
            per_device_train_batch_size=ner_config.BATCH_SIZE,
            per_device_eval_batch_size=ner_config.BATCH_SIZE,
            learning_rate=ner_config.LEARNING_RATE,
            weight_decay=ner_config.WEIGHT_DECAY,
            
            # Optimización (FP16 / Gradientes)
            fp16=ner_config.FP16 and torch.cuda.is_available(), # Solo activa si hay GPU
            gradient_accumulation_steps=ner_config.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=ner_config.WARMUP_STEPS,
            warmup_ratio=ner_config.WARMUP_RATIO,
            
            # Estrategia de Evaluación/Guardado
            eval_strategy=ner_config.EVAL_STRATEGY,
            save_strategy=ner_config.SAVE_STRATEGY,
            save_total_limit=ner_config.SAVE_TOTAL_LIMIT,
            load_best_model_at_end=ner_config.LOAD_BEST_MODEL_AT_END,
            metric_for_best_model=f"eval_{ner_config.METRIC_FOR_BEST_MODEL}", # ej: eval_f1
            
            # Sistema y Logging
            seed=ner_config.SEED,
            data_seed=ner_config.SEED,
            logging_steps=ner_config.LOGGING_STEPS,
            report_to=ner_config.REPORT_TO,
            dataloader_num_workers=ner_config.NUM_WORKERS
        )
        
        # Callbacks (System Metrics + Early Stopping opcional)
        trainer_callbacks = [SystemMetricsCallback()]
        
        if ner_config.USE_EARLY_STOPPING:
            logging.info(f"⏱️  Early Stopping Activado (Paciencia: {ner_config.EARLY_STOPPING_PATIENCE})")
            trainer_callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=ner_config.EARLY_STOPPING_PATIENCE,
                    early_stopping_threshold=ner_config.EARLY_STOPPING_THRESHOLD
                )
            )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
            compute_metrics=lambda p: compute_metrics(p, id2label),
            callbacks=trainer_callbacks
        )
        
        logging.info("🔥 INICIANDO ENTRENAMIENTO...")
        trainer.train()
        final_model = trainer.model

    else:
        logging.info("🛑 MODO SOLO DESCARGA (Bypass Training)")
        final_model = AutoModelForTokenClassification.from_pretrained(
            ner_config.MODEL_NAME,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            cache_dir=ner_config.CACHE_DIR
        )

    # Guardar .PT Final
    logging.info("💾 Guardando checkpoint final autocontenido...")
    output_dir = Path(ner_config.NER_MODEL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_file_path = output_dir / "get_news_extractor.pt"
    
    checkpoint = {
        'model_state_dict': final_model.state_dict(),
        'config': final_model.config.to_dict(),
        'tokenizer_name': ner_config.MODEL_NAME,
        'id2label': id2label,
        'label2id': label2id
    }
    
    torch.save(checkpoint, pt_file_path)
    logging.info(f"🎉 COMPLETADO: {pt_file_path}")

if __name__ == "__main__":
    main_ner_pipeline()