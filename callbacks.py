# callbacks.py
"""
Custom Hugging Face Trainer callbacks for system and GPU monitoring.
This module will log GPU metrics ONLY IF an NVIDIA GPU and drivers are found.
"""
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import mlflow

# Import psutil at the top level for global scope
try:
    import psutil
except ImportError:
    psutil = None # If not found, set the global variable to None

class SystemMetricsCallback(TrainerCallback):
    """
    A custom TrainerCallback that logs CPU, RAM, and NVIDIA GPU usage to MLflow.
    GPU monitoring is only active if an NVIDIA GPU and drivers are detected.
    """
    def __init__(self):
        super().__init__()
        self.process = None
        self.gpu_monitoring_available = False
        self.pynvml = None # Initialize as None to be safe

        # --- Initialize psutil for CPU/RAM ---
        if psutil:
            self.process = psutil.Process()
            print("✅ SystemMetricsCallback: Psutil initialized. CPU and RAM will be monitored.")
        else:
            print("⚠️ SystemMetricsCallback: psutil not found. CPU/RAM metrics will not be logged.")

        # --- Initialize nvidia_smi for GPU (Robust Logic) ---
        try:
            # Step 1: Try to import the library. This fails if not installed.
            import nvidia_smi as pynvml
            self.pynvml = pynvml # Store the successfully imported module on `self`

            # Step 2: If import succeeds, then try to initialize the driver connection.
            try:
                self.pynvml.nvmlInit()
                self.gpu_monitoring_available = True
                print("✅ SystemMetricsCallback: NVML initialized successfully. NVIDIA GPU will be monitored.")
            except self.pynvml.NVMLError:
                # This catches driver issues (e.g., no GPU, bad driver install)
                print("ℹ️ SystemMetricsCallback: No NVIDIA GPU detected or driver issue. GPU metrics will not be logged.")
                self.gpu_monitoring_available = False

        except ImportError:
            # This only catches the case where `pip install nvidia-ml-py` was not run
            print("ℹ️ SystemMetricsCallback: The 'nvidia-ml-py' package is not installed. GPU metrics will not be logged.")
            self.gpu_monitoring_available = False


    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return

        metrics_to_log = {}

        if self.process:
            metrics_to_log.update({
                "system/cpu_usage_percent": psutil.cpu_percent(),
                "system/ram_usage_percent": psutil.virtual_memory().percent,
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
                print(f"⚠️ Warning: Could not log GPU metrics mid-training. Disabling GPU monitoring. Error: {e}")
                self.gpu_monitoring_available = False
        
        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log, step=state.global_step)

    def __del__(self):
        if self.gpu_monitoring_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass