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
        self.pynvml = None

        # --- Initialize psutil for CPU/RAM ---
        if psutil:
            self.process = psutil.Process()
            print("✅ SystemMetricsCallback: Psutil initialized. CPU and RAM will be monitored.")
        else:
            print("⚠️ SystemMetricsCallback: psutil not found. CPU/RAM metrics will not be logged.")

        # --- Initialize nvidia_smi for GPU (Robust Logic) ---
        try:
            import nvidia_smi as pynvml
            self.pynvml = pynvml
            try:
                self.pynvml.nvmlInit()
                self.gpu_monitoring_available = True
                print("✅ SystemMetricsCallback: NVML initialized successfully. NVIDIA GPU will be monitored.")
            except self.pynvml.NVMLError:
                print("ℹ️ SystemMetricsCallback: No NVIDIA GPU detected or driver issue. GPU metrics will not be logged.")
                self.gpu_monitoring_available = False
        except ImportError:
            print("ℹ️ SystemMetricsCallback: The 'nvidia-ml-py' package is not installed. GPU metrics will not be logged.")
            self.gpu_monitoring_available = False

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called by the Trainer every time it logs metrics.
        """
        if not state.is_world_process_zero:
            return

        metrics_to_log = {}

        # --- MODIFIED: Ensure all CPU/RAM metrics are logged correctly ---
        if self.process:
            # Get system-wide metrics
            system_cpu_percent = psutil.cpu_percent()
            system_ram_percent = psutil.virtual_memory().percent
            
            # Get metrics specific to this Python process
            process_cpu_percent = self.process.cpu_percent() / psutil.cpu_count()
            process_ram_mb = self.process.memory_info().rss / (1024 * 1024)

            metrics_to_log.update({
                "system/cpu_usage_percent": system_cpu_percent,
                "system/ram_usage_percent": system_ram_percent,
                "process/cpu_usage_percent": process_cpu_percent,
                "process/ram_usage_mb": process_ram_mb
            })

        # --- GPU Metrics Block (unchanged) ---
        if self.gpu_monitoring_available:
            try:
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)

                metrics_to_log.update({
                    "gpu/utilization_percent": util.gpu,
                    "gpu/vram_usage_percent": (mem_info.used / mem_info.total) * 100,
                    "gpu/vram_used_mb": mem_info.used / (1024**2),
                })
            except self.pynvml.NVMLError as e:
                print(f"⚠️ Warning: Could not log GPU metrics mid-training. Disabling GPU monitoring. Error: {e}")
                self.gpu_monitoring_available = False
        
        # Log whatever metrics were collected
        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log, step=state.global_step)

    def __del__(self):
        if self.gpu_monitoring_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass