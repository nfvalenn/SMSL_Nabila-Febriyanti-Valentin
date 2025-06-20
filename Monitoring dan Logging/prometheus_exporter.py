from prometheus_client import start_http_server, Gauge
import time
import mlflow
import random

# Port untuk Prometheus scrape
PROMETHEUS_EXPORTER_PORT = 8000

# Tracking URI MLflow (local)
mlflow.set_tracking_uri("file:///D:/SMSL_Nabila-Febriyanti-Valentin/Membangun_model/SMSL_Nabila-Febriyanti-Valentin/Membangun_model/mlruns")

# Inisialisasi metriks Prometheus
loss_metric = Gauge('model_loss', 'Current Loss of ML Model')
accuracy_metric = Gauge('model_accuracy', 'Current Accuracy of ML Model')
f1_score_metric = Gauge('model_f1_score', 'Current F1 Score of ML Model')

# Tambahan 7 metrik baru
precision_metric = Gauge('model_precision', 'Current Precision of ML Model')
recall_metric = Gauge('model_recall', 'Current Recall of ML Model')
auc_metric = Gauge('model_auc', 'Current AUC of ML Model')
latency_metric = Gauge('model_latency_ms', 'Response Latency of ML Model (ms)')
cpu_usage_metric = Gauge('model_cpu_usage', 'CPU Usage (%) of ML Model')
memory_usage_metric = Gauge('model_memory_usage', 'Memory Usage (MB) of ML Model')
inference_requests_metric = Gauge('model_inference_requests_total', 'Number of Inference Requests')

# Fungsi untuk ambil data metriks (masih simulasi/random)
def get_metrics():
    loss = random.uniform(0.1, 1.0)
    accuracy = random.uniform(0.5, 1.0)
    f1_score = random.uniform(0.5, 1.0)
    precision = random.uniform(0.5, 1.0)
    recall = random.uniform(0.5, 1.0)
    auc = random.uniform(0.5, 1.0)
    latency = random.uniform(10, 100)  # ms
    cpu_usage = random.uniform(0, 100)  # percent
    memory_usage = random.uniform(100, 1000)  # MB
    inference_requests = random.randint(0, 5000)
    return loss, accuracy, f1_score, precision, recall, auc, latency, cpu_usage, memory_usage, inference_requests

if __name__ == "__main__":
    print(f"Starting Prometheus Exporter on port {PROMETHEUS_EXPORTER_PORT}...")
    start_http_server(PROMETHEUS_EXPORTER_PORT)

    while True:
        (
            loss, accuracy, f1_score,
            precision, recall, auc,
            latency, cpu_usage, memory_usage, inference_requests
        ) = get_metrics()

        # Update metriks Prometheus
        loss_metric.set(loss)
        accuracy_metric.set(accuracy)
        f1_score_metric.set(f1_score)
        precision_metric.set(precision)
        recall_metric.set(recall)
        auc_metric.set(auc)
        latency_metric.set(latency)
        cpu_usage_metric.set(cpu_usage)
        memory_usage_metric.set(memory_usage)
        inference_requests_metric.set(inference_requests)

        print(f"[Updated] Loss={loss:.4f} | Accuracy={accuracy:.4f} | F1={f1_score:.4f} | "
              f"Precision={precision:.4f} | Recall={recall:.4f} | AUC={auc:.4f} | "
              f"Latency={latency:.2f}ms | CPU={cpu_usage:.2f}% | Mem={memory_usage:.2f}MB | Requests={inference_requests}")

        time.sleep(5)
