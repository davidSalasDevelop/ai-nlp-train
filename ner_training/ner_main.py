import logging
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import mlflow
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
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

# --- CLASE PARA FOCAL LOSS (OPCIONAL) ---
class FocalLoss(nn.Module):
    # ... (esta clase no cambia)
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        if self.reduction == 'mean': return torch.mean(F_loss)
        elif self.reduction == 'sum': return torch.sum(F_loss)
        else: return F_loss

# --- TRAINER PERSONALIZADO PARA MANEJAR LA FUNCIÓN DE PÉRDIDA ---
class CustomLossTrainer(Trainer):
    # --- ¡CORRECCIÓN IMPORTANTE AQUÍ! ---
    # Añadimos **kwargs para aceptar cualquier argumento extra que el Trainer le pase.
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    # -----------------------------------
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = None
        if labels is not None:
            if ner_config.USE_FOCAL_LOSS:
                loss_fct = FocalLoss(alpha=ner_config.FOCAL_LOSS_ALPHA, gamma=ner_config.FOCAL_LOSS_GAMMA)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p, id2label):
    # ... (esta función no cambia)
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[id2label[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[id2label[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    return {"precision": precision_score(true_labels, true_predictions), "recall": recall_score(true_labels, true_predictions), "f1": f1_score(true_labels, true_predictions)}

def build_custom_head(model_config, num_labels, head_layers_config, activation_fn_name):
    # ... (esta función no cambia)
    head_layers = []
    input_size = model_config.hidden_size
    ActivationClass = getattr(nn, activation_fn_name)
    for layer_size, dropout_rate in head_layers_config:
        head_layers.append(nn.Dropout(dropout_rate))
        head_layers.append(nn.Linear(input_size, layer_size))
        head_layers.append(ActivationClass())
        input_size = layer_size
    head_layers.append(nn.Linear(input_size, num_labels))
    return nn.Sequential(*head_layers)

def main_ner_pipeline():
    # ... (El resto del main no cambia, es idéntico al que ya tenías)
    logging.info("🚀 INICIANDO PIPELINE NER (MODO EXPERTO) 🚀")
    
    os.environ['MLFLOW_TRACKING_URI'] = ner_config.MLFLOW_TRACKING_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = ner_config.MLFLOW_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = ner_config.MLFLOW_PASSWORD
    mlflow.set_experiment(ner_config.MLFLOW_EXPERIMENT_NAME)

    labels = ner_config.ENTITY_LABELS
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(ner_config.MODEL_NAME, cache_dir=ner_config.CACHE_DIR)
    
    final_model = None

    if ner_config.DO_TRAINING:
        logging.info("🏋️  MODO ENTRENAMIENTO ACTIVADO")
        tokenized_datasets = load_and_prepare_ner_data(
            ner_config.DATASET_PATH, tokenizer, label_list=labels, max_length=ner_config.MAX_LENGTH
        )
        
        model = AutoModelForTokenClassification.from_pretrained(
            ner_config.MODEL_NAME, num_labels=len(labels), id2label=id2label,
            label2id=label2id, ignore_mismatched_sizes=True, cache_dir=ner_config.CACHE_DIR
        )
        
        if ner_config.USE_CUSTOM_HEAD:
            logging.info("🧠 CONSTRUYENDO CABEZA PERSONALIZADA...")
            model.classifier = build_custom_head(
                model.config, len(labels), ner_config.CUSTOM_HEAD_LAYERS, ner_config.CUSTOM_HEAD_ACTIVATION
            )
        
        training_args = TrainingArguments(
            output_dir="./ner_results",
            run_name=f"ner-{datetime.now().strftime('%Y%m%d-%H%M')}",
            num_train_epochs=ner_config.TRAIN_EPOCHS,
            per_device_train_batch_size=ner_config.BATCH_SIZE,
            learning_rate=ner_config.LEARNING_RATE,
            adam_beta1=ner_config.ADAM_BETA1,
            adam_beta2=ner_config.ADAM_BETA2,
            adam_epsilon=ner_config.ADAM_EPSILON,
            lr_scheduler_type=ner_config.LR_SCHEDULER_TYPE,
            weight_decay=ner_config.WEIGHT_DECAY,
            max_grad_norm=ner_config.MAX_GRAD_NORM,
            label_smoothing_factor=ner_config.LABEL_SMOOTHING_FACTOR,
            fp16=ner_config.FP16 and torch.cuda.is_available(),
            gradient_accumulation_steps=ner_config.GRADIENT_ACCUMULATION_STEPS,
            dataloader_num_workers=ner_config.NUM_WORKERS,
            seed=ner_config.SEED,
            data_seed=ner_config.SEED,
            eval_strategy=ner_config.EVAL_STRATEGY,
            save_strategy=ner_config.SAVE_STRATEGY,
            logging_strategy=ner_config.LOGGING_STRATEGY,
            save_total_limit=ner_config.SAVE_TOTAL_LIMIT,
            load_best_model_at_end=ner_config.LOAD_BEST_MODEL_AT_END,
            metric_for_best_model=f"eval_{ner_config.METRIC_FOR_BEST_MODEL}",
            logging_steps=ner_config.LOGGING_STEPS,
            report_to=ner_config.REPORT_TO,
        )
        
        callbacks = [SystemMetricsCallback()]
        if ner_config.USE_EARLY_STOPPING:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=ner_config.EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=ner_config.EARLY_STOPPING_THRESHOLD
            ))
        
        trainer = CustomLossTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
            compute_metrics=lambda p: compute_metrics(p, id2label),
            callbacks=callbacks
        )
        
        logging.info("🔥 INICIANDO ENTRENAMIENTO CON CONFIGURACIÓN COMPLETA...")
        trainer.train()
        final_model = trainer.model
    else:
        logging.info("🛑 MODO SOLO DESCARGA")
        final_model = AutoModelForTokenClassification.from_pretrained(
            ner_config.MODEL_NAME, num_labels=len(labels), id2label=id2label,
            label2id=label2id, ignore_mismatched_sizes=True, cache_dir=ner_config.CACHE_DIR
        )

    logging.info("💾 Guardando checkpoint final...")
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