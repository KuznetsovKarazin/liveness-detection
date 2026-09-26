"""Avvio dell'app: python run_app.py
Variabili d'ambiente: HOST (default 127.0.0.1), PORT (8000), FORWARDED_ALLOW_IPS (127.0.0.1), LIVENESSLAB_SKIP_UNTRAINED (0/1),
LIVENESSLAB_ALLOW_EVAL (1), LIVENESSLAB_ALLOW_FORCE (1; 0 sul server pubblico), LIVENESSLAB_MAX_PENDING (4), LIVENESSLAB_MAX_CONN_PER_IP (4).
In alternativa: uvicorn tesi_app.server:app --app-dir src"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: TensorFlow e PyTorch portano ciascuno una copia di OpenMP

# Su Windows importare PyTorch DOPO TensorFlow/h5py fa morire il processo senza errori (conflitto di DLL):
# lo importiamo per primo, prima di qualunque altra libreria numerica (tesi_app.server lo ripete per chi usa uvicorn diretto).
try:
    import torch  # noqa: E402,F401
except Exception:  # noqa: BLE001
    pass

import uvicorn  # noqa: E402

if __name__ == "__main__":
    # HOST di default solo su loopback: per raggiungere l'app da un telefono in rete locale usare HOST=0.0.0.0
    uvicorn.run("tesi_app.server:app", host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)),
                reload=False,
                # dietro IIS/ARR sulla stessa macchina: si accettano gli header X-Forwarded-* solo dal proxy locale
                proxy_headers=True, forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
                # frame WebSocket fino a 17 MB: un'immagine di 12 MB (limite di decode_data_url) in base64 ne occupa 16;
                # dal browser le immagini arrivano comunque già ridotte (< 1 MB)
                ws_max_size=17 * 1024 * 1024,
                # IIS/ARR non gestisce l'estensione permessage-deflate dei WebSocket: senza questa riga il browser
                # apre la connessione e la vede chiudersi subito (codice 1006) quando l'app sta dietro IIS.
                ws_per_message_deflate=False)
