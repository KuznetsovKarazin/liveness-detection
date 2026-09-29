/* Service worker di LivenessLab (PWA).
 * Mette in cache solo la "shell" dell'interfaccia (pagina, CSS, JS, icone, manifest) così l'app si apre subito e
 * mostra la pagina anche senza rete. Le richieste all'API e al WebSocket non vengono mai messe in cache: l'analisi
 * richiede il server. Il nome della cache contiene la versione dell'app ({{VERSION}}, sostituita dal server):
 * a ogni nuova versione la cache precedente viene cancellata. */
const VERSION = "{{VERSION}}";
const CACHE = "livenesslab-" + VERSION;
const SHELL = ["/", "/manifest.webmanifest", "/static/icons/icon-192.png", "/static/icons/icon-512.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  if (url.pathname.startsWith("/api/") || url.pathname === "/ws" || url.pathname === "/sw.js") return;
  if (e.request.mode === "navigate" || url.pathname === "/") {
    // pagina: prima la rete (così gli aggiornamenti arrivano subito), la cache solo se la rete manca
    e.respondWith(fetch(e.request).then(r => { const copy = r.clone(); caches.open(CACHE).then(c => c.put("/", copy)); return r; })
      .catch(() => caches.match("/")));
    return;
  }
  if (url.pathname.startsWith("/static/")) {
    // risorse statiche (già con ?v= nel nome): prima la cache, poi la rete
    e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request).then(r => { const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); return r; })));
  }
});
