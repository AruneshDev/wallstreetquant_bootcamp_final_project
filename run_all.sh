#!/usr/bin/env bash
set -euo pipefail

# cd into project
cd semiconductor_quant_research

# Activate venv
source .venv/bin/activate

echo "=== 1. Data download (all universes) ==="
python -c "from src.data_loader import download_all_universes; download_all_universes()"

echo "=== 2. Base features (semi_core) ==="
python src/features.py

echo "=== 3. Alt-data features (semi_core) ==="
python src/features_alt.py

echo "=== 4. NLP features (semi_core) ==="
python src/nlp_signal.py

echo "=== 5. Backtests (CS momentum + pairs) ==="
python src/backtest.py

echo "=== 6. Alpha decomposition ==="
python src/alpha.py

echo "=== 7. ML baselines (RF/GBM) ==="
python src/model_baseline.py

echo "=== 8. Transformer model ==="
python src/model_transformer.py

echo "=== 9. GNN model ==="
python src/model_gnn.py

echo "=== 10. Signal combiner (GBM meta-model) ==="
python src/model_signal_combiner.py
echo "=== 11. UI Dashboard ==="
python -m streamlit run app.py
echo "=== DONE ==="