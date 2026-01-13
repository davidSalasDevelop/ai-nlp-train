import logging
import os
from pathlib import Path

# --- CONFIGURACIÓN DEL DIRECTORIO TEMPORAL ---
# Este bloque se ejecuta ANTES de importar torch o transformers para asegurar
# que las librerías usen un directorio con permisos de escritura.
temp_dir = Path("ner_training/tmp")
temp_dir.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(temp_dir.resolve())
os.environ['TEMP'] = str(temp_dir.resolve()) # Compatibilidad con Windows
os.environ['TMP'] = str(temp_dir.resolve())  # Compatibilidad con Windows

# Ahora se importan las librerías pesadas
import torch
from datetime import datetime
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
import mlflow
from mlflow.tracking import MlflowClient

# Importación de los módulos personalizados del proyecto
import ner_config
from ner_data_loader import load_and_prepare_ner_data
from ner_model import build_ner_model
from ner_callbacks import SystemMetricsCallback

# Configuración del logging para monitorear el proceso
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main_ner_pipeline():
    """
    Función principal que orquesta todo el pipeline de entrenamiento del modelo NER:
    1. Configura MLflow para el seguimiento de experimentos.
    2. Carga y procesa el dataset.
    3. Construye el modelo y el tokenizer.
    4. Define los argumentos de entrenamiento.
    5. Inicia el entrenamiento con el Trainer de Hugging Face.
    6. Guarda el mejor modelo en un archivo .pt autocontenido.
    """
    logging.info("🚀 INICIANDO PIPELINE DE ENTRENAMIENTO DEL MODELO NER 🚀")
    
    start_time = datetime.now()
    
    # --- 1. Configuración de MLflow ---
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

    # --- 2. Preparación de Etiquetas y Tokenizer ---
    labels = ner_config.ENTITY_LABELS
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(ner_config.MODEL_NAME)
    
    # --- 3. Carga y Preparación de Datos ---
    # Llama a la función del data loader para obtener los datasets tokenizados y alineados
    tokenized_datasets = load_and_prepare_ner_data(
        ner_config.DATASET_PATH, 
        tokenizer, 
        label_list=labels
    )
    
    # --- 4. Construcción del Modelo ---
    model = build_ner_model(ner_config.MODEL_NAME, id2label, label2id)

    # --- 5. Configuración de Argumentos de Entrenamiento ---
    training_args = TrainingArguments(
        output_dir="./ner_results",
        run_name=f"ner-extractor-run-{start_time.strftime('%Y%m%d-%H%M%S')}",
        num_train_epochs=ner_config.TRAIN_EPOCHS,
        per_device_train_batch_size=ner_config.BATCH_SIZE,
        per_device_eval_batch_size=ner_config.BATCH_SIZE,
        learning_rate=ner_config.LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1", # Usamos F1-score para seleccionar el mejor modelo
        weight_decay=0.01,
        logging_strategy="epoch",
        report_to="mlflow", # Habilita la integración automática con MLflow
    )

    # Data Collator se encarga de rellenar (padding) los lotes de datos dinámicamente
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # --- 6. Definición de Métricas de Evaluación ---
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Elimina los tokens especiales (con label -100) para la evaluación
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Calcula las métricas usando seqeval
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions)
        }

    # --- 7. Creación e Inicio del Trainer ---
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator, 
        compute_metrics=compute_metrics,
        callbacks=[SystemMetricsCallback()], # Añade nuestro callback para métricas de sistema
    )

    logging.info("🔥 Iniciando entrenamiento del modelo NER...")
    trainer.train()
    
    # --- 8. Guardado del Modelo Autocontenido ---
    logging.info("💾 Creando archivo .pt autocontenido para el mejor modelo NER...")
    
    # El trainer.model es el mejor modelo gracias a `load_best_model_at_end=True`
    best_model = trainer.model
    
    # Creamos un diccionario (checkpoint) con todo lo necesario para cargar el modelo
    checkpoint = {
        'model_state_dict': best_model.state_dict(),
        'config': best_model.config.to_dict(),
        'tokenizer_name': ner_config.MODEL_NAME, # Guardamos el nombre para recrear el tokenizer
        'id2label': id2label,
        'label2id': label2id
    }
    
    # Creamos el directorio de salida si no existe
    output_dir = Path(ner_config.NER_MODEL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pt_file_path = output_dir / "get_news_extractor.pt"
    
    torch.save(checkpoint, pt_file_path)
    logging.info(f"   ✅ Checkpoint .pt guardado exitosamente en '{pt_file_path}'")
    
    logging.info("🎉 PIPELINE NER COMPLETADO 🎉")

if __name__ == "__main__":
    main_ner_pipeline()