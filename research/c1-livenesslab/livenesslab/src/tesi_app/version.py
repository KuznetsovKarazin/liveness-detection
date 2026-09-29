"""Versione dell'applicazione, in un unico punto.

Formato: data della revisione (anno.mese.giorno) e, solo se nello stesso giorno ci sono più revisioni pubblicate,
un numero progressivo in coda (es. 2026.09.26.2). Compare accanto al titolo nell'interfaccia, nel manifest della
PWA e nel nome della cache del service worker: cambiare la versione fa scaricare la nuova interfaccia a tutti.
"""
VERSION = "2026.09.26"
