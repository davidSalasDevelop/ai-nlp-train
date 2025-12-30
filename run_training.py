# run_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import json
import mlflow
import os

# --------------------------------------------------------------------------
# --- 1. CONFIGURACIÓN: ¡¡MODIFICA ESTAS LÍNEAS!! ---
# --------------------------------------------------------------------------

# Apunta a la URL de tu servidor MLflow que ya está corriendo
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  # ¡¡CAMBIA ESTO por la IP y puerto de tu servidor!!

# Nombre del experimento en la interfaz de MLflow
MLFLOW_EXPERIMENT_NAME = "Entrenamiento NLU Bilingüe"

# Nombre con el que se registrará el modelo final en el "Model Registry" de MLflow
MLFLOW_MODEL_NAME = "mi-modelo-nlu"

# Ruta local al archivo con tus datos de entrenamiento
DATASET_PATH = "dataset.json"


# --------------------------------------------------------------------------
# --- 2. DEFINICIÓN DEL MODELO Y PRE-PROCESAMIENTO (Sin cambios) ---
# --------------------------------------------------------------------------

TOKENIZER = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

class IntentEntityModel(nn.Module):
    # ... (Copia y pega la clase IntentEntityModel completa aquí, sin cambios)
    def __init__(self, vocab_size, num_intents, num_entity_tags, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.intent_head = nn.Linear(d_model, num_intents)
        self.entity_head = nn.Linear(d_model, num_entity_tags)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        encoded_text = self.transformer_encoder(embedded)
        intent_logits = self.intent_head(encoded_text[:, 0, :])
        entity_logits = self.entity_head(encoded_text)
        return intent_logits, entity_logits

def load_and_preprocess_data(filepath, tokenizer, intent_to_id, entity_to_id):
    # ... (Copia y pega la función load_and_preprocess_data completa aquí, sin cambios)
    with open(filepath, 'r') as f:
        data = json.load(f)
    texts, intent_labels, entity_labels_list = [], [], []
    for item in data:
        texts.append(item['text'])
        intent_labels.append(intent_to_id[item['intent']])
        encoding = tokenizer(item['text'], return_offsets_mapping=True, truncation=True, padding=False)
        token_offsets = encoding['offset_mapping']
        entity_tags = [entity_to_id.get('O')] * len(encoding['input_ids'])
        for entity in item['entities']:
            label, start_char, end_char = entity['label'], entity['start'], entity['end']
            is_first_token = True
            for i, (start, end) in enumerate(token_offsets):
                if start >= start_char and end <= end_char and start < end:
                    if is_first_token:
                        entity_tags[i] = entity_to_id.get(f'B-{label}')
                        is_first_token = False
                    else:
                        entity_tags[i] = entity_to_id.get(f'I-{label}')
        entity_labels_list.append(entity_tags)
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    max_len = tokenized_inputs['input_ids'].shape[1]
    padded_entity_labels = []
    for labels in entity_labels_list:
        padded_labels = labels + [entity_to_id.get('O')] * (max_len - len(labels))
        padded_entity_labels.append(padded_labels[:max_len])
    return (tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'],
            torch.tensor(intent_labels, dtype=torch.long), torch.tensor(padded_entity_labels, dtype=torch.long))


# --------------------------------------------------------------------------
# --- 3. EL SCRIPT PRINCIPAL QUE SE CONECTA Y ENTRENA ---
# --------------------------------------------------------------------------

def main():
    print(f"--- Conectando a MLflow en {MLFLOW_TRACKING_URI} ---")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        print(f"--- Iniciando nuevo Run en MLflow: {run.info.run_id} ---")
        
        # --- PARÁMETROS DEL MODELO (se registrarán en MLflow) ---
        params = {
            "num_epochs": 50,
            "batch_size": 2,
            "learning_rate": 5e-5,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2
        }
        mlflow.log_params(params)

        # --- PREPARACIÓN DE DATOS ---
        print("--- Preparando datos ---")
        intents = ["get_news", "check_weather", "get_user_info"]
        entities = ["TOPIC", "LOCATION", "DATE"]
        intent_to_id = {intent: i for i, intent in enumerate(intents)}
        entity_to_id = {'O': 0}
        for entity in entities:
            entity_to_id[f'B-{entity}'] = len(entity_to_id)
            entity_to_id[f'I-{entity}'] = len(entity_to_id)

        input_ids, _, intent_labels, entity_labels = load_and_preprocess_data(
            DATASET_PATH, TOKENIZER, intent_to_id, entity_to_id
        )
        dataset = TensorDataset(input_ids, intent_labels, entity_labels)
        dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

        # --- INICIALIZACIÓN DEL MODELO ---
        device = torch.device("cpu") # Forzamos CPU para que funcione en tu servidor
        model = IntentEntityModel(
            vocab_size=TOKENIZER.vocab_size,
            num_intents=len(intent_to_id),
            num_entity_tags=len(entity_to_id),
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"]
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"])
        loss_fn_intent = nn.CrossEntropyLoss()
        loss_fn_entity = nn.CrossEntropyLoss()

        # --- BUCLE DE ENTRENAMIENTO ---
        print("--- Iniciando entrenamiento ---")
        for epoch in range(params["num_epochs"]):
            total_loss = 0
            for b_input_ids, b_intent_labels, b_entity_labels in dataloader:
                b_input_ids, b_intent_labels, b_entity_labels = b_input_ids.to(device), b_intent_labels.to(device), b_entity_labels.to(device)
                optimizer.zero_grad()
                intent_logits, entity_logits = model(b_input_ids)
                loss_intent = loss_fn_intent(intent_logits, b_intent_labels)
                loss_entity = loss_fn_entity(entity_logits.view(-1, len(entity_to_id)), b_entity_labels.view(-1))
                total_batch_loss = loss_intent + loss_entity
                total_batch_loss.backward()
                optimizer.step()
                total_loss += total_batch_loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{params['num_epochs']}, Loss: {avg_loss:.4f}")
            # Registramos la métrica de pérdida en MLflow para ver gráficos bonitos
            mlflow.log_metric("avg_loss", avg_loss, step=epoch)

        print("--- Entrenamiento completado ---")

        # --- REGISTRO DEL MODELO EN EL SERVIDOR MLFLOW ---
        print(f"--- Registrando el modelo como '{MLFLOW_MODEL_NAME}' en el servidor MLflow ---")
        
        # Creamos la "firma" que le dice a MLflow qué tipo de datos espera el modelo
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        input_schema = Schema([ColSpec("string", "text")])
        output_schema = Schema([ColSpec("string")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Creamos un diccionario con todo lo que el modelo necesita para funcionar
        model_info = {
            "model_state_dict": model.state_dict(),
            "id_to_intent": {i: intent for intent, i in intent_to_id.items()},
            "id_to_entity": {i: entity for entity, i in entity_to_id.items()}
        }

        # ¡La magia ocurre aquí! MLflow guarda el modelo y lo registra en el servidor.
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model", # Subcarpeta dentro del run de MLflow
            registered_model_name=MLFLOW_MODEL_NAME, # ¡El nombre clave!
            signature=signature,
            code_paths=[__file__] # Incluye este mismo script para reproducibilidad
        )
        print("--- ¡Modelo registrado con éxito! ---")

if __name__ == "__main__":
    # Asegúrate de tener un archivo 'dataset.json' en la misma carpeta
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: No se encuentra el archivo '{DATASET_PATH}'. Por favor, créalo.")
    else:
        main()
