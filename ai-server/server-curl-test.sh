curl -X POST "http://vscode:8000/predict"      -H "Content-Type: application/json"      -d '{"text": "quiero reservar un vuel"}' &
curl -X POST "http://vscode:8001/predict"      -H "Content-Type: application/json"      -d '{"text": "quiero reservar un vuel"}' &
curl -X POST "http://vscode:8003/predict"      -H "Content-Type: application/json"      -d '{"text": "quiero reservar un vuel"}'