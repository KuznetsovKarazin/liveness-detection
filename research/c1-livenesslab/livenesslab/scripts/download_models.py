"""
Scarica in anticipo tutti i modelli di terze parti usati da LivenessLab, così il primo avvio sul server
non deve aspettare ~700 MB di download. Uso:  python scripts/download_models.py

1. MediaPipe Face Landmarker (3,7 MB) -> models/third_party/face_landmarker.task
2. CLIP ViT-B/32 (~600 MB) e Depth Anything V2 Small (~100 MB) nella cache di Hugging Face dell'utente
3. verifica che MiniFASNet e RetinaFace siano nel submodule Silent-Face-Anti-Spoofing
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD = ROOT / "models" / "third_party"
THIRD.mkdir(parents=True, exist_ok=True)

# 1. MediaPipe Face Landmarker
task = THIRD / "face_landmarker.task"
if not task.exists():
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    print("Scarico", url)
    urllib.request.urlretrieve(url, task)
print("OK face_landmarker.task", task.stat().st_size // 1024, "KB")

# 2. Modelli Hugging Face (solo i file necessari all'inferenza)
from huggingface_hub import snapshot_download  # noqa: E402

REVISIONS = {   # CLIP (zero-shot e linear probe), DINOv2 (linear probe), Depth Anything (profondità)"openai/clip-vit-base-patch32": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
             "facebook/dinov2-small": "ed25f3a31f01632728cabb09d1542f84ab7b0056",   # DINOv2 linear probe (~90 MB)
             "depth-anything/Depth-Anything-V2-Small-hf": "5426e4f0f36572d16453bbda7a8389317b1bef99"}   # le stesse revisioni usate dagli analizzatori
for repo, rev in REVISIONS.items():
    p = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.txt", "*.safetensors", "vocab*", "merges*", "preprocessor*", "tokenizer*"])
    print("OK", repo, "->", p)

# 3. Pesi MiniFASNet e RetinaFace (arrivano con il submodule: git submodule update --init)
sf = ROOT / "src" / "third_party" / "Silent-Face-Anti-Spoofing"
for rel in ["resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth", "resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth",
            "resources/detection_model/Widerface-RetinaFace.caffemodel"]:
    ok = (sf / rel).exists()
    print("OK" if ok else "MANCA", sf / rel)
    if not ok:
        sys.exit("Manca il submodule Silent-Face-Anti-Spoofing: git submodule update --init --recursive")
print("\nTutti i modelli sono pronti.")
