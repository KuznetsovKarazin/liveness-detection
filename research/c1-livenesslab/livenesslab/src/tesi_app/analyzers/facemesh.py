"""
Geometria 3D del volto con MediaPipe Face Landmarker (Kartynnik et al., 2019; Grishchenko et al., 2020):
478 punti 3D, posa della testa, blendshape (occhi, bocca). Analizzatore DESCRITTIVO: su una singola immagine non
dà un verdetto (la z è predetta da un modello canonico del viso e "funziona" anche su una foto stampata); è la base
per l'analisi video (battito di ciglia, coerenza del movimento).
"""
from __future__ import annotations

import math
import threading

import cv2
import numpy as np

from ..core import Analyzer, NodeSpec, Result, RunContext, registry
from ..paths import ROOT
from .common import COMMON_EDGES, COMMON_NODES, no_face, step_crop, step_face, step_input

MODEL_PATH = ROOT / "models" / "third_party" / "face_landmarker.task"
_lock = threading.Lock()
_lm = {}


def _stub_sounddevice():
    """mediapipe.tasks importa anche i task audio, che caricano `sounddevice`/PortAudio: su un server senza
    scheda audio (es. Windows Server) l'import fallisce con 'Error initializing PortAudio'. Non ci serve
    l'audio: se il modulo non si carica, ne registriamo uno vuoto al suo posto."""
    import sys
    if "sounddevice" in sys.modules:
        return
    try:
        import sounddevice  # noqa: F401
    except Exception:  # noqa: BLE001
        import types
        sys.modules["sounddevice"] = types.ModuleType("sounddevice")


_stub_sounddevice()   # eseguito al caricamento del modulo, prima di ogni `import mediapipe`


def _load():
    """Crea il Face Landmarker (una volta sola) con blendshape e matrice di posa attivi."""
    with _lock:
        if "lm" not in _lm:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
            _lm["lm"] = vision.FaceLandmarker.create_from_options(opts)
            _lm["conn"] = vision.FaceLandmarksConnections
        return _lm["lm"], _lm["conn"]


def _euler_deg(R: np.ndarray):
    """Angoli di Eulero (pitch, yaw, roll) in gradi da una matrice di rotazione 3×3 (convenzione ZYX)."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], sy))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


class FaceMeshAnalyzer(Analyzer):
    id = "facemesh"; name = "Face Mesh 3D (MediaPipe)"; family = "classico"; order = 33; color = "#b8f26b"
    short = "478 punti 3D del volto, posa della testa e 52 espressioni (blendshape): descrive la geometria del viso e prepara l'analisi video (battito di ciglia, movimento)."
    reference = "Kartynnik et al., Real-time Facial Surface Geometry from Monocular Video, CVPRW 2019 · Grishchenko et al., Attention Mesh, 2020"
    reference_url = "https://arxiv.org/abs/1907.06724"

    def reliability(self):
        return "descriptive"

    def graph(self):
        nodes = COMMON_NODES + [NodeSpec("mesh", "Face Landmarker", "model"), NodeSpec("landmarks", "478 punti 3D"),
                                NodeSpec("pose", "Posa della testa"), NodeSpec("zrelief", "Rilievo (z)"),
                                NodeSpec("blend", "Blendshape"), NodeSpec("score", "Sintesi geometrica", "decision"),
                                NodeSpec("verdict", "Verdetto", "output")]
        edges = COMMON_EDGES + [["crop", "mesh"], ["mesh", "landmarks"], ["mesh", "pose"], ["landmarks", "zrelief"], ["mesh", "blend"],
                                ["pose", "score"], ["zrelief", "score"], ["blend", "score"], ["score", "verdict"]]
        return nodes, edges

    def _no_mesh(self, aid: str, face) -> Result:
        """Risultato descrittivo quando il Landmarker non trova un volto nel ritaglio (non è un errore)."""
        return Result(aid, "unknown", None, "Solo descrittivo (mesh non trovata)", "descriptive", {"mesh": False, "no_face": no_face(face)},
                      explanation="Il Face Landmarker non ha trovato un volto nel ritaglio: nessuna geometria da descrivere. Non pesa nel consenso.")

    def run(self, ctx: RunContext) -> Result:
        import mediapipe as mp
        aid = self.id
        step_input(ctx, aid); face = step_face(ctx, aid)
        crop = step_crop(ctx, aid, face, margin=0.35, square=True)
        lm, conn = _load()
        with ctx.step(aid, "mesh", "Face Landmarker (BlazeFace + Attention Mesh)",
                      "Due reti leggere in cascata: un rilevatore trova il volto, una rete di regressione predice 468 punti "
                      "del viso + 10 dell'iride con coordinate x, y e una profondità relativa z, più la matrice di posa 4×4.") as s:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            found = bool(res.face_landmarks)
            if not found:
                s.note("Nessun volto trovato dal Landmarker nel ritaglio.")
            pts = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], np.float32) if found else np.zeros((0, 3), np.float32)
            s.metric("punti", int(len(pts))); s.metric("blendshape", len(res.face_blendshapes[0]) if found and res.face_blendshapes else 0)
        if not found:
            return self._no_mesh(aid, face)
        H, W = crop.shape[:2]
        xy = (pts[:, :2] * [W, H]).astype(int)
        with ctx.step(aid, "landmarks", "Mesh dei 478 punti",
                      "I punti formano una maglia triangolare (tessellazione) che segue la superficie del viso. "
                      "Il colore dei punti codifica z: chiaro = più vicino alla camera.") as s:
            vis = crop.copy()
            z = pts[:, 2]; zn = (z - z.min()) / (z.max() - z.min() + 1e-9)
            for c in conn.FACE_LANDMARKS_TESSELATION:
                cv2.line(vis, tuple(xy[c.start]), tuple(xy[c.end]), (70, 70, 70), 1)
            for (x, y), v in zip(xy, zn):
                cv2.circle(vis, (int(x), int(y)), 1, (int(255 * (1 - v)), int(255 * (1 - v) * 0.6 + 80), int(255 * v + 60)), -1)
            s.image(vis, max_side=520)
            s.metric("larghezza volto (px)", int(xy[:, 0].max() - xy[:, 0].min())); s.metric("altezza volto (px)", int(xy[:, 1].max() - xy[:, 1].min()))
        with ctx.step(aid, "pose", "Posa della testa",
                      "Dalla matrice di trasformazione 4×4 si ricavano gli angoli di Eulero: yaw (rotazione sinistra/destra), "
                      "pitch (su/giù), roll (inclinazione). Una posa plausibile e coerente con la mesh è un requisito di liveness; "
                      "nei video la variazione di posa nel tempo è un indizio forte.") as s:
            yaw = pitch = roll = 0.0
            if res.facial_transformation_matrixes:
                M = np.array(res.facial_transformation_matrixes[0])
                pitch, yaw, roll = _euler_deg(M[:3, :3])
            vis = crop.copy()
            nose = tuple(xy[1]); L = W // 4             # punto 1 della mesh = punta del naso
            cv2.arrowedLine(vis, nose, (int(nose[0] + L * math.sin(math.radians(yaw))), int(nose[1] - L * math.sin(math.radians(pitch)))), (80, 220, 120), 3)
            cv2.putText(vis, f"yaw {yaw:.0f}  pitch {pitch:.0f}  roll {roll:.0f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            s.image(vis, max_side=520); s.metric("yaw (°)", yaw); s.metric("pitch (°)", pitch); s.metric("roll (°)", roll)
        with ctx.step(aid, "zrelief", "Rilievo della mesh (coordinata z)",
                      "Deviazione standard di z normalizzata sulla larghezza del volto e differenza naso-guance. "
                      "ATTENZIONE: la rete predice z da un modello canonico del viso, quindi anche una foto stampata ottiene un "
                      "rilievo 'da volto': l'indizio è debole su singola immagine, utile nel video (coerenza tra frame).") as s:
            wf = float(pts[:, 0].max() - pts[:, 0].min()) + 1e-9
            z_std = float(pts[:, 2].std() / wf)
            nose_z = float(pts[1, 2]); cheek_z = float((pts[234, 2] + pts[454, 2]) / 2)   # 234 e 454 = zigomi sinistro/destro
            relief = float((cheek_z - nose_z) / wf)
            s.metric("z dev. std / larghezza", z_std); s.metric("rilievo naso-guance", relief)
            zimg = np.zeros((H, W), np.float32)
            for (x, y), v in zip(xy, zn):
                cv2.circle(zimg, (int(x), int(y)), max(2, W // 90), float(1 - v), -1)
            s.image(cv2.applyColorMap((zimg * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), max_side=520)
        with ctx.step(aid, "blend", "Blendshape (espressioni)",
                      "52 coefficienti in [0,1] che descrivono l'espressione: apertura occhi, sorriso, sopracciglia. "
                      "Su una foto singola descrivono lo stato; su un video permettono di misurare il battito di ciglia "
                      "(un volto stampato non sbatte mai le palpebre).") as s:
            bs = {c.category_name: float(c.score) for c in (res.face_blendshapes[0] if res.face_blendshapes else [])}
            keys = ["eyeBlinkLeft", "eyeBlinkRight", "mouthSmileLeft", "mouthSmileRight", "jawOpen", "browInnerUp", "eyeLookOutLeft", "mouthPucker"]
            for k in keys:
                if k in bs:
                    s.metric(k, bs[k])
            top = sorted(bs.items(), key=lambda kv: -kv[1])[:8]
            s.image(_bars(top))
        with ctx.step(aid, "score", "Sintesi geometrica",
                      "Su una singola immagine la geometria stimata NON permette un verdetto onesto: la rete predice la z da "
                      "un modello canonico del viso, quindi anche una foto stampata ottiene un rilievo 'da volto' (verificato sui "
                      "campioni). Qui riassumiamo posa, rilievo e stato degli occhi; il verdetto arriverà dall'analisi video "
                      "(battito di ciglia, coerenza 3D del movimento).") as s:
            pose_ok = max(abs(yaw), abs(pitch)) < 35
            eyes_open = bs.get("eyeBlinkLeft", 0) < 0.5 and bs.get("eyeBlinkRight", 0) < 0.5
            s.metric("posa frontale plausibile", "sì" if pose_ok else "no")
            s.metric("occhi aperti", "sì" if eyes_open else "no")
            s.metric("rilievo naso-guance", relief)
        with ctx.step(aid, "verdict", "Nessun verdetto (analisi descrittiva)",
                      "Questo analizzatore descrive la geometria e non partecipa al consenso. Diventa un rilevatore vero "
                      "nel video: un volto stampato non sbatte le palpebre e non ruota in modo coerente con la mesh 3D.") as s:
            s.metric("decisione", "NON DETERMINABILE SU SINGOLA IMMAGINE")
        expl = (f"Trovati {len(pts)} punti 3D. Posa della testa: yaw {yaw:.0f}°, pitch {pitch:.0f}°, roll {roll:.0f}° "
                + ("(frontale, plausibile). " if pose_ok else "(posa molto ruotata). ")
                + ("Occhi aperti" if eyes_open else "Occhi chiusi o socchiusi") + f" (eyeBlink {bs.get('eyeBlinkLeft', 0):.2f}/{bs.get('eyeBlinkRight', 0):.2f}); "
                f"rilievo naso-guance {relief:.2f}. Su una foto singola questa geometria non distingue un volto vero da una stampa: "
                "nessun verdetto, non pesa nel consenso. Nel video misurerà battito di ciglia e coerenza del movimento.")
        return Result(aid, "unknown", None, "Solo descrittivo (serve il video)", "descriptive",
                      {"yaw": yaw, "pitch": pitch, "roll": roll, "relief": relief, "blink": bs.get("eyeBlinkLeft", 0), "mesh": True, "no_face": no_face(face)}, explanation=expl)


def _bars(items, w=640, h=260) -> np.ndarray:
    """Barre orizzontali dei blendshape più attivi."""
    img = np.full((h, w, 3), 24, np.uint8)
    rowh = h // max(1, len(items))
    for i, (k, v) in enumerate(items):
        y = i * rowh + 5
        cv2.rectangle(img, (220, y), (220 + int(v * (w - 280)), y + rowh - 10), (107, 242, 184), -1)
        cv2.putText(img, k[:26], (8, y + rowh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.putText(img, f"{v:.2f}", (226 + int(v * (w - 280)), y + rowh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return img


registry.register(FaceMeshAnalyzer())
