#!/bin/bash
# Contoh pengujian endpoint /predict (Deployment Opsi A - REST API)
# Ganti URL dengan URL deployment publik Anda.
URL="http://localhost:8000"

echo "--- Health check ---"
curl -s $URL/health

echo -e "\n\n--- High-risk machine ---"
curl -s -X POST $URL/predict \
  -H "Content-Type: application/json" \
  -d '{"Type":"L","Air temperature [K]":298.0,"Process temperature [K]":308.0,"Rotational speed [rpm]":1320,"Torque [Nm]":65.0,"Tool wear [min]":220}'

echo -e "\n\n--- Normal machine ---"
curl -s -X POST $URL/predict \
  -H "Content-Type: application/json" \
  -d '{"Type":"H","Air temperature [K]":300.0,"Process temperature [K]":310.0,"Rotational speed [rpm]":1500,"Torque [Nm]":40.0,"Tool wear [min]":50}'
echo ""
