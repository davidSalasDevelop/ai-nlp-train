# ner_main_finetune.py (CORREGIDO v3)
import logging
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import mlflow
import copy
import traceback 

from huggingface_hub import hf_hub_download
from transformers import (
    AlbertTokenizer, 
    AutoModelForTokenClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from seqeval.metrics import f1_score, precision_score, recall_score

# Importaciones locales
import ner_config
from ner_data_loader import load_and_prepare_ner_data
from ner_callbacks import SystemMetricsCallback

# --- CONFIGURACIÓN DE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- FUNCIONES DE ARQUITECTURA (Reutilizadas de ner_main.py) ---

def build_custom_head(model_config, num_labels, head_layers_config, activation_fn_name):
    """Reconstruye el cabezal personalizado exactamente igual al original."""
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

# --- CLASES AUXILIARES (FocalLoss, Trainer, Metrics...) ---

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        active_loss = targets.view(-1) != -100
        active_logits = inputs.view(-1, inputs.size(-1))
        active_labels = torch.where(active_loss, targets.view(-1), torch.tensor(0).to(targets.device))
        BCE_loss = F.cross_entropy(active_logits, active_labels, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        F_loss = F_loss[active_loss.view(-1)]
        if self.reduction == 'mean': return torch.mean(F_loss)
        return F_loss

class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = None
        if labels is not None:
            loss_fct = FocalLoss(alpha=ner_config.FOCAL_LOSS_ALPHA, gamma=ner_config.FOCAL_LOSS_GAMMA) if ner_config.USE_FOCAL_LOSS else nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class BestModelCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.metric_name = f"eval_{ner_config.METRIC_FOR_BEST_MODEL}"
        self.greater_is_better = ner_config.METRIC_FOR_BEST_MODEL != 'loss'
        self.best_metric = -float('inf') if self.greater_is_better else float('inf')
        self.best_model_state_dict = None
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        current_metric = metrics.get(self.metric_name)
        if current_metric is None: return
        if (self.greater_is_better and current_metric > self.best_metric) or (not self.greater_is_better and current_metric < self.best_metric):
            self.best_metric = current_metric
            self.best_model_state_dict = copy.deepcopy(kwargs['model'].state_dict())
            logger.info(f"✨ ¡Nuevo mejor modelo encontrado! {self.metric_name}: {self.best_metric:.4f}")
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.best_model_state_dict:
            logger.info("🏁 Cargando los pesos del mejor modelo para el guardado final.")
            kwargs['model'].load_state_dict(self.best_model_state_dict)

def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[id2label[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[id2label[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    return {"precision": precision_score(true_labels, true_predictions), "recall": recall_score(true_labels, true_predictions), "f1": f1_score(true_labels, true_predictions)}

# --- PIPELINE PRINCIPAL ---

def main_finetune_pipeline():
    print("="*60)
    logger.info("🚀 INICIANDO PIPELINE DE FINE-TUNING NER")
    print("="*60)
    
    os.environ['MLFLOW_TRACKING_URI'] = ner_config.MLFLOW_TRACKING_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = ner_config.MLFLOW_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = ner_config.MLFLOW_PASSWORD
    mlflow.set_experiment(ner_config.MLFLOW_EXPERIMENT_NAME)
    
    # 1. Cargar Checkpoint
    logger.info(f"💾 Cargando checkpoint: {ner_config.BASE_MODEL_FOR_FINETUNE}")
    checkpoint = torch.load(ner_config.BASE_MODEL_FOR_FINETUNE, map_location='cpu')
    
    model_config_dict = checkpoint['config']
    model_state_dict = checkpoint['model_state_dict']
    id2label = {int(k): v for k, v in checkpoint['id2label'].items()}
    label2id = checkpoint['label2id']
    tokenizer_name = checkpoint.get('tokenizer_name', ner_config.MODEL_NAME)

    # 2. Tokenizador
    logger.info(f"🛠️  Preparando Tokenizador...")
    model_file = hf_hub_download(repo_id=tokenizer_name, filename="spiece.model", cache_dir=ner_config.CACHE_DIR)
    tokenizer = AlbertTokenizer(vocab_file=model_file, do_lower_case=False)

    # 3. Datos
    logger.info("📂 Cargando Datos...")
    tokenized_datasets = load_and_prepare_ner_data(
        ner_config.CUSTOM_DATASET_FILES, tokenizer, label_list=list(label2id.keys()), max_length=ner_config.MAX_LENGTH
    )

    # 4. Reconstruir Modelo
    logger.info("🧠 Reconstruyendo arquitectura...")
    model_type = model_config_dict['model_type']
    config_class = CONFIG_MAPPING[model_type]
    model_config = config_class.from_dict(model_config_dict)
    
    model = AutoModelForTokenClassification.from_config(model_config)

    # --- CRUCIAL: Reconstruir el cabezal antes de cargar pesos ---
    if ner_config.USE_CUSTOM_HEAD:
        logger.info("   -> Aplicando cabezal personalizado para coincidir con el checkpoint.")
        model.classifier = build_custom_head(
            model.config, len(id2label), ner_config.CUSTOM_HEAD_LAYERS, ner_config.CUSTOM_HEAD_ACTIVATION
        )

    # Ahora sí, cargar los pesos
    model.load_state_dict(model_state_dict)
    logger.info("✅ Pesos cargados correctamente.")

    # 5. Entrenamiento
    training_args = TrainingArguments(
        output_dir="./ner_results_finetune",
        run_name=f"ner-finetune-{datetime.now().strftime('%Y%m%d-%H%M')}",
        num_train_epochs=ner_config.TRAIN_EPOCHS,
        per_device_train_batch_size=ner_config.BATCH_SIZE,
        learning_rate=ner_config.LEARNING_RATE,
        fp16=ner_config.FP16 and torch.cuda.is_available(),
        eval_strategy=ner_config.EVAL_STRATEGY,
        save_strategy="no",
        logging_steps=ner_config.LOGGING_STEPS,
        report_to=ner_config.REPORT_TO,
        load_best_model_at_end=False,
    )
    
    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=lambda p: compute_metrics(p, id2label),
        callbacks=[SystemMetricsCallback(), BestModelCallback()]
    )
    
    logger.info("🔥 INICIANDO FINE-TUNING...")
    trainer.train()
    
    # 6. Guardado
    output_dir = Path(ner_config.NER_MODEL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_file_path = output_dir / ner_config.FINETUNED_OUTPUT_MODEL_NAME
    
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': trainer.model.config.to_dict(),
        'tokenizer_name': tokenizer_name,
        'id2label': id2label,
        'label2id': label2id
    }, pt_file_path)
    
    logger.info(f"🎉 COMPLETADO. Guardado en: {pt_file_path}")

if __name__ == "__main__":
    main_finetune_pipeline()