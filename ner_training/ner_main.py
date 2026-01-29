# ner_main.py
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

# Importaciones de Hugging Face
from huggingface_hub import hf_hub_download
from transformers import (
    AlbertTokenizer, 
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from seqeval.metrics import f1_score, precision_score, recall_score

# Importaciones locales
import ner_config
from ner_data_loader import load_and_prepare_ner_data
from ner_callbacks import SystemMetricsCallback

# --- CONFIGURACIÓN DE LOGGING ORDENADA ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s', 
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuración de entorno temporal
temp_dir = Path("ner_training/tmp")
temp_dir.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(temp_dir.resolve())

# --- CLASE FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        active_loss = targets.view(-1) != -100
        active_logits = inputs.view(-1, inputs.size(-1))
        active_labels = torch.where(
            active_loss, targets.view(-1), torch.tensor(0).to(targets.device)
        )
        BCE_loss = F.cross_entropy(active_logits, active_labels, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        F_loss = F_loss[active_loss.view(-1)]
        if self.reduction == 'mean': return torch.mean(F_loss)
        elif self.reduction == 'sum': return torch.sum(F_loss)
        else: return F_loss

# --- TRAINER PERSONALIZADO ---
class CustomLossTrainer(Trainer):
    # Fix: **kwargs captura argumentos extra para evitar errores de versión
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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

# --- CALLBACK MEJOR MODELO ---
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

        is_better = (self.greater_is_better and current_metric > self.best_metric) or \
                    (not self.greater_is_better and current_metric < self.best_metric)

        if is_better:
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

def build_custom_head(model_config, num_labels, head_layers_config, activation_fn_name):
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
    print("="*60)
    logger.info("🚀 INICIANDO PIPELINE NER")
    print("="*60)
    
    os.environ['MLFLOW_TRACKING_URI'] = ner_config.MLFLOW_TRACKING_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = ner_config.MLFLOW_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = ner_config.MLFLOW_PASSWORD
    mlflow.set_experiment(ner_config.MLFLOW_EXPERIMENT_NAME)

    labels = ner_config.ENTITY_LABELS
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    # 1. CARGA DEL TOKENIZADOR (FIX MANUAL)
    print("\n" + "-"*30)
    logger.info("🛠️  Preparando Tokenizador...")
    try:
        model_file = hf_hub_download(
            repo_id=ner_config.MODEL_NAME, 
            filename="spiece.model", 
            cache_dir=ner_config.CACHE_DIR
        )
        tokenizer = AlbertTokenizer(vocab_file=model_file, do_lower_case=False)
        logger.info("✅ Tokenizador (AlbertTokenizer) cargado correctamente.")
    except Exception as e:
        logger.error(f"❌ Error crítico cargando tokenizador: {e}")
        return

    final_model = None

    if ner_config.DO_TRAINING:
        # 2. CARGA DE DATOS
        print("\n" + "-"*30)
        logger.info("📂 Cargando Datasets...")
        try:
            tokenized_datasets = load_and_prepare_ner_data(
                ner_config.CUSTOM_DATASET_FILES, tokenizer, label_list=labels, max_length=ner_config.MAX_LENGTH
            )
        except RuntimeError as e:
            logger.error(f"❌ {e}")
            return

        # 3. CARGA DEL MODELO
        print("\n" + "-"*30)
        logger.info("🧠 Inicializando Modelo Base...")
        model = AutoModelForTokenClassification.from_pretrained(
            ner_config.MODEL_NAME, num_labels=len(labels), id2label=id2label,
            label2id=label2id, ignore_mismatched_sizes=True, cache_dir=ner_config.CACHE_DIR
        )
        
        if ner_config.USE_CUSTOM_HEAD:
            logger.info("   -> Añadiendo cabezal personalizado.")
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
            save_strategy="no", # Importante: No guardar checkpoints intermedios en disco
            logging_strategy=ner_config.LOGGING_STRATEGY,
            load_best_model_at_end=False,
            metric_for_best_model=f"eval_{ner_config.METRIC_FOR_BEST_MODEL}",
            greater_is_better=(ner_config.METRIC_FOR_BEST_MODEL != 'loss'),
            logging_steps=ner_config.LOGGING_STEPS,
            report_to=ner_config.REPORT_TO,
        )
        
        callbacks = [SystemMetricsCallback(), BestModelCallback()]
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
        
        # 4. ENTRENAMIENTO
        print("\n" + "-"*30)
        logger.info("🔥 INICIANDO ENTRENAMIENTO...")
        trainer.train()
        final_model = trainer.model
    else:
        logger.info("🛑 MODO SOLO DESCARGA (Sin entrenamiento)")
        final_model = AutoModelForTokenClassification.from_pretrained(
            ner_config.MODEL_NAME, num_labels=len(labels), id2label=id2label,
            label2id=label2id, ignore_mismatched_sizes=True, cache_dir=ner_config.CACHE_DIR
        )

    # 5. GUARDADO
    print("\n" + "-"*30)
    logger.info("💾 Procesando guardado final...")
    output_dir = Path(ner_config.NER_MODEL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_file_path = output_dir / config.OUTPUT_MODEL_NAME
    
    # Asegurar contigüidad
    for param in final_model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    checkpoint = {
        'model_state_dict': final_model.state_dict(),
        'config': final_model.config.to_dict(),
        'tokenizer_name': ner_config.MODEL_NAME,
        'id2label': id2label,
        'label2id': label2id
    }
    torch.save(checkpoint, pt_file_path)
    logger.info(f"🎉 COMPLETADO. Archivo guardado en: {pt_file_path}")
    print("="*60)

if __name__ == "__main__":
    main_ner_pipeline()