"""
Rilevamento del volto, condiviso da tutti gli analizzatori.

Rilevatore principale: RetinaFace (WIDER FACE) in formato Caffe, distribuito con Silent-Face-Anti-Spoofing
(Apache-2.0) ed eseguito con il modulo DNN di OpenCV. Fallback: Haar cascade (Viola-Jones) incluso in OpenCV.
Tutte le coordinate sono in pixel dell'immagine originale, con (x, y) angolo in alto a sinistra.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .paths import SILENT_FACE


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    detector: str

    @property
    def bbox(self):
        return [self.x, self.y, self.w, self.h]

    def crop_box(self, img_shape, margin: float = 0.0, square: bool = False) -> Tuple[int, int, int, int]:
        """Riquadro di ritaglio (x0, y0, x1, y1) allargato del `margin` relativo per lato (0.2 = +20 %) e, se
        richiesto, reso quadrato sul lato maggiore. Il riquadro viene tagliato ai bordi dell'immagine: chi lo usa
        deve fare i conti con questo (ad esempio depth.py, che deve sapere dove sta il volto dentro il ritaglio)."""
        H, W = img_shape[:2]
        cx, cy = self.x + self.w / 2, self.y + self.h / 2
        w, h = self.w * (1 + 2 * margin), self.h * (1 + 2 * margin)
        if square:
            w = h = max(w, h)
        x0, y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
        x1, y1 = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
        return x0, y0, x1, y1

    def crop(self, img: np.ndarray, margin: float = 0.0, square: bool = False) -> np.ndarray:
        """Ritaglio del volto secondo `crop_box`."""
        x0, y0, x1, y1 = self.crop_box(img.shape, margin, square)
        return img[y0:y1, x0:x1]


class FaceDetector:
    """RetinaFace via OpenCV DNN, con Haar cascade di riserva se il modello Caffe non è disponibile
    o non trova nulla."""

    def __init__(self):
        proto = SILENT_FACE / "resources" / "detection_model" / "deploy.prototxt"
        model = SILENT_FACE / "resources" / "detection_model" / "Widerface-RetinaFace.caffemodel"
        self.net = None
        if proto.exists() and model.exists():
            self.net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, img: np.ndarray, min_conf: float = 0.6) -> Optional[Face]:
        """Il volto più probabile, oppure None."""
        if self.net is not None:
            f = self._retina(img, min_conf)
            if f is not None:
                return f
        return self._haar(img)

    def _retina(self, img: np.ndarray, min_conf: float) -> Optional[Face]:
        H, W = img.shape[:2]
        ar = W / H
        small = img
        if W * H >= 192 * 192:
            # la rete lavora bene su ~192 px di lato: si riduce l'immagine mantenendo le proporzioni
            small = cv2.resize(img, (int(192 * math.sqrt(ar)), int(192 / math.sqrt(ar))), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(small, 1, mean=(104, 117, 123))    # media BGR del training di RetinaFace
        self.net.setInput(blob, "data")
        out = self.net.forward("detection_out").squeeze()
        if out.ndim == 1:
            out = out[None, :]
        i = int(np.argmax(out[:, 2]))                 # colonna 2 = confidenza; 3..6 = riquadro normalizzato (l, t, r, b)
        conf = float(out[i, 2])
        if conf < min_conf:
            return None
        l, t, r, b = out[i, 3] * W, out[i, 4] * H, out[i, 5] * W, out[i, 6] * H
        x, y = int(max(0, l)), int(max(0, t))
        w, h = int(min(W, r)) - x, int(min(H, b)) - y
        if w < 16 or h < 16:
            return None
        return Face(x, y, w, h, conf, "RetinaFace (Caffe, OpenCV DNN)")

    def _haar(self, img: np.ndarray) -> Optional[Face]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # il volto più grande
        return Face(int(x), int(y), int(w), int(h), 0.5, "Haar cascade (Viola-Jones)")


_detector: Optional[FaceDetector] = None


def get_detector() -> FaceDetector:
    """Istanza unica del rilevatore (il caricamento del modello Caffe si fa una volta sola)."""
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector


def draw_face(img: np.ndarray, face: Face, color=(80, 220, 120)) -> np.ndarray:
    """Copia dell'immagine con il riquadro del volto disegnato sopra."""
    out = img.copy()
    t = max(2, int(min(out.shape[:2]) / 200))
    cv2.rectangle(out, (face.x, face.y), (face.x + face.w, face.y + face.h), color, t)
    return out
