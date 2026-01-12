# ner_callbacks.py
"""
Callback personalizado para el Trainer de Hugging Face.
Registra métricas detalladas del sistema (CPU, RAM, GPU) en MLflow.
"""
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import mlflow

# Importar psutil de forma segura
try:
    import psutil
except ImportError:
    psutil = None

# Importar la librería de NVIDIA de forma segura
try:
    import nvidia_smi as pynvml
except ImportError:
    pynvml = None

class SystemMetricsCallback(TrainerCallback):
    """
    Registra el uso de CPU, RAM y GPU en cada evento de log del Trainer.
    """
    def __init__(self):
        super().__init__()
        self.process = None
        self.gpu_monitoring_available = False
        self.pynvml = None

        if psutil:
            self.process = psutil.Process()
            logging.info("✅ SystemMetricsCallback: Psutil inicializado. CPU y RAM serán monitoreados.")
        else:
            logging.warning("⚠️ SystemMetricsCallback: psutil no encontrado. CPU/RAM no serán monitoreados.")

        if pynvml:
            try:
                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.gpu_monitoring_available = True
                logging.info("✅ SystemMetricsCallback: NVML inicializado. NVIDIA GPU será monitoreada.")
            except pynvml.NVMLError:
                logging.info("ℹ️ SystemMetricsCallback: No se detectó GPU NVIDIA o hay un problema con los drivers. El monitoreo de GPU será omitido.")
                self.gpu_monitoring_available = False
        else:
            logging.info("ℹ️ SystemMetricsCallback: 'nvidia-ml-py' no está instalado. El monitoreo de GPU será omitido.")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return

        metrics_to_log = {}

        if self.process:
            metrics_to_log.update({
                "system/cpu_usage_percent": psutil.cpu_percent(),
                "system/ram_usage_percent": psutil.virtual_memory().percent,
                "process/cpu_usage_percent": self.process.cpu_percent() / psutil.cpu_count(),
                "process/ram_usage_mb": self.process.memory_info().rss / (1024 * 1024)
            })

        if self.gpu_monitoring_available:
            try:
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                metrics_to_log.update({
                    "gpu/utilization_percent": util.gpu,
                    "gpu/vram_usage_percent": (mem_info.used / mem_info.total) * 100,
                    "gpu/vram_used_mb": mem_info.used / (1024**2),
                    "gpu/temperature_celsius": temp
                })
            except self.pynvml.NVMLError as e:
                logging.warning(f"⚠️ No se pudieron registrar las métricas de GPU. Deshabilitando monitoreo. Error: {e}")
                self.gpu_monitoring_available = False
        
        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log, step=state.global_step)

    def __del__(self):
        if self.gpu_monitoring_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass