#!/bin/bash

echo "=== Enviando peticiones a los servicios ==="
echo ""

# Ejecutar uno tras otro (sin &)
echo "📤 Puerto 8000:"
curl -X POST "http://vscode:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Guatemala y Venezuela"}'
echo ""

echo ""

echo "📤 Puerto 8001:"
curl -X POST "http://vscode:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "quiero reservar un vuelo"}'
echo ""

echo ""

echo "📤 Puerto 8002:"
curl -X POST "http://vscode:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "quiero reservar un vuelo"}'
echo ""

echo ""

echo "✅ Todas las peticiones completadas"