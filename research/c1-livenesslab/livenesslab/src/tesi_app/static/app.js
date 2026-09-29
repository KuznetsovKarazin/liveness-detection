/* LivenessLab – frontend. Nessuna dipendenza esterna.
 *
 * Due moduli indipendenti (IIFE):
 *  1. analisi di una singola immagine: catalogo, finestra di configurazione a quattro passi (sorgente, algoritmi,
 *     opzioni, riepilogo), barra di missione, WebSocket, coda degli eventi con ritmo, diagramma SVG della pipeline,
 *     card dei passi, riepilogo con consenso, scheda-catalogo degli algoritmi;
 *  2. valutazione su dataset: schede dei dataset (dentro la finestra di configurazione), tabella delle metriche,
 *     curve ROC, confronto tra dataset, esportazione Excel/PDF.
 *  3. PWA: registrazione del service worker, avviso "Vuoi installare LivenessLab?", installazione diretta su
 *     Android/Chrome/Edge (beforeinstallprompt), guida per iPhone e iPad, avviso di nuova versione.
 * I moduli comunicano attraverso poche variabili globali (window.__ws, window.__analyzers, window.__evalHandler,
 * window.__eval, window.__wizard, window.__showModal).
 */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const FAMILY = { docente: "CNN del docente", pretrained: "Pre-addestrato", classico: "Classico spiegabile" };
  const FAMILY_ORDER = ["docente", "pretrained", "classico"];
  // etichette dei livelli di affidabilità restituiti dal server (Result.reliability)
  const REL = { trained: "addestrato", heuristic: "euristico", untrained: "non addestrato", zeroshot: "zero-shot", descriptive: "descrittivo", error: "errore" };
  // pesi nel consenso: chi non è in tabella (untrained, descriptive, error) non ha punteggio e non entra
  const RELW = { trained: 1, zeroshot: 0.75, heuristic: 0.5 };
  const WITH_VERDICT = ["trained", "zeroshot", "heuristic"], NO_VERDICT = ["untrained", "descriptive", "error"];

  let resolveAnalyzers; window.__analyzersReady = new Promise(res => { resolveAnalyzers = res; });   // il modulo Valutazione aspetta il catalogo
  const state = { analyzers: [], selected: new Set(), image: null, mode: "image", ws: null, running: false, runId: null, ignoreRunId: null,
                  queue: [], timer: null, results: {}, panes: {}, stepCount: 0, totalSteps: 0,
                  follow: true, current: { analyzer: null, step: null } };

  // ------------------------------------------------------------------ catalogo e selezione
  async function loadCatalog() {
    try {
      state.analyzers = await (await fetch("/api/analyzers")).json(); window.__analyzers = state.analyzers;
    } catch (err) {
      console.error("catalogo non caricato", err);
      $("#sr-algos").textContent = "catalogo non caricato: server non raggiungibile, ricarica la pagina"; $("#sr-algos").classList.add("empty");
      resolveAnalyzers([]);      // chi aspetta il catalogo non deve restare sospeso
      return;
    }
    state.analyzers.forEach(a => state.selected.add(a.id));
    window.__selectedIds = () => state.analyzers.map(a => a.id).filter(id => state.selected.has(id));
    renderWizardAlgos(); renderCatalog(); refreshSelectionUI(); resolveAnalyzers(state.analyzers);
    let samples = [];
    try { samples = await (await fetch("/api/samples")).json(); } catch (err) { console.error("immagini di prova non caricate", err); }
    const sbox = $("#samples"); sbox.innerHTML = "";
    for (const s of samples) {
      const b = document.createElement("button"); b.type = "button"; b.className = "sample"; b.title = `${s.id} · ${s.label} · ${s.source}`;
      b.innerHTML = `<img src="${s.thumb}" alt=""><span class="tag">${s.label}</span>`;
      b.addEventListener("click", async () => { const r = await (await fetch(`/api/samples/${s.id}`)).json(); setImage(r.image); });
      sbox.appendChild(b);
    }
  }
  const selectedList = () => state.analyzers.filter(a => state.selected.has(a.id));
  function setSelected(id, on) { on ? state.selected.add(id) : state.selected.delete(id); refreshSelectionUI(); }
  // preset di selezione (passo 2 della finestra)
  function applyPreset(name) {
    const isPooled = a => (a.id || "").endsWith("-pooled");
    for (const a of state.analyzers) {
      let on = true;
      if (name === "none") on = false;
      else if (name === "verdict") on = WITH_VERDICT.includes(a.reliability);
      else if (name === "official") on = a.family !== "docente" || !isPooled(a);
      else if (name === "fast") on = a.family !== "docente";
      on ? state.selected.add(a.id) : state.selected.delete(a.id);
    }
    refreshSelectionUI();
  }
  // valore valido del campo "immagini per classe" (vuoto o fuori range → massimo del dataset)
  function limitValue() { const inp = $("#eval-limit"); const mx = +inp.dataset.max || +inp.max || 150; const v = +inp.value; return v >= 1 ? Math.min(v, mx) : mx; }
  // conteggi usati in più punti: quanti con verdetto, quanti senza, per famiglia
  function selectionStats() {
    const sel = selectedList(); const count = {}; sel.forEach(a => { count[a.reliability] = (count[a.reliability] || 0) + 1; });
    const nv = WITH_VERDICT.reduce((n, k) => n + (count[k] || 0), 0), nn = NO_VERDICT.reduce((n, k) => n + (count[k] || 0), 0);
    const part = keys => keys.filter(k => count[k]).map(k => `${count[k]} ${REL[k]}`).join(", ");
    const fam = {}; sel.forEach(a => { fam[a.family] = (fam[a.family] || 0) + 1; });
    return { sel, count, nv, nn, part, fam, total: state.analyzers.length };
  }
  // aggiorna tutte le viste che dipendono da selezione, immagine, modalità: card di partenza, finestra, catalogo
  function refreshSelectionUI() {
    const st = selectionStats();
    // card di partenza
    const src = $("#sr-source");
    if (state.mode === "dataset") { const d = window.__eval?.current(); src.textContent = d ? `dataset ${d.name}` : "dataset: nessuno scelto"; }
    else src.textContent = state.image ? "immagine caricata" : "nessuna immagine";
    src.classList.toggle("empty", state.mode === "dataset" ? !window.__eval?.current() : !state.image);
    $("#sr-algos").textContent = `${st.sel.length} su ${st.total}` + (st.sel.length ? ` · ${st.nv} con verdetto` : "");
    $("#sr-algos").classList.toggle("empty", !st.sel.length);
    $("#sr-opts").textContent = `ritmo ${$("#pace").selectedOptions[0].textContent.toLowerCase()}` + (state.mode === "dataset" ? ` · ${limitValue()} immagini per classe` : "");
    if (!state.running && $("#mission").hidden) $("#setup-title").textContent = state.mode === "dataset" ? "Nuova valutazione" : "Nuova analisi";
    // finestra: chip, contatori, riepilogo, piè di pagina
    $$("#wz-algos .chip input").forEach(i => { i.checked = state.selected.has(i.dataset.id); });
    $$("#wz-algos details").forEach(d => { const boxes = $$(".chip input", d); const c = $(".acc-cnt", d); if (c) c.textContent = `${boxes.filter(b => b.checked).length} / ${boxes.length}`; });
    $("#wz-algo-count").innerHTML = st.sel.length ? `<b>${st.sel.length}</b> selezionati su ${st.total} · <span class="ok">${st.nv} con verdetto</span>${st.nn ? ` · <span class="warn">${st.nn} senza verdetto</span>` : ""}` : `<span class="warn">Nessun algoritmo selezionato</span>`;
    $$("#catalog .cat-check input").forEach(i => { i.checked = state.selected.has(i.dataset.id); });
    if (wz.dlg.open) { renderWizardSummary(); updateWizardFooter(); }
  }

  // ------------------------------------------------------------------ input immagine
  function setImage(dataUrl) {
    state.image = dataUrl;
    const dz = $("#dropzone"), img = $("#drop-preview");
    img.src = dataUrl; img.hidden = false; dz.classList.add("has-image");
    $("#btn-clear").disabled = false; refreshSelectionUI();
  }
  function clearImage() {
    state.image = null; const dz = $("#dropzone"), img = $("#drop-preview");
    img.hidden = true; img.src = ""; dz.classList.remove("has-image"); $("#btn-clear").disabled = true; refreshSelectionUI();
  }
  function fileToImage(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const fr = new FileReader(); fr.onload = () => downscale(fr.result, 1600).then(setImage); fr.readAsDataURL(file);
  }
  // le foto dei telefoni sono enormi: si riducono a 1600 px lato lungo prima di inviarle al server
  function downscale(dataUrl, max) {
    return new Promise(res => { const im = new Image(); im.onload = () => {
      const k = Math.min(1, max / Math.max(im.width, im.height));
      if (k === 1) return res(dataUrl);
      const c = document.createElement("canvas"); c.width = im.width * k; c.height = im.height * k;
      c.getContext("2d").drawImage(im, 0, 0, c.width, c.height); res(c.toDataURL("image/jpeg", 0.92)); }; im.src = dataUrl; });
  }
  const dz = $("#dropzone");
  dz.addEventListener("click", () => $("#file").click());
  dz.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file").click(); } });
  $("#file").addEventListener("change", e => fileToImage(e.target.files[0]));
  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", e => fileToImage(e.dataTransfer.files[0]));
  document.addEventListener("paste", e => { const f = [...(e.clipboardData?.files || [])][0]; if (f) { fileToImage(f); if (!wz.dlg.open) wz.open(1, "image"); } });
  $("#btn-clear").addEventListener("click", clearImage);

  // webcam: anteprima nel <video>, scatto su canvas
  let stream = null;
  $("#btn-webcam").addEventListener("click", async () => {
    try { stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 1280 } });
      $("#video").srcObject = stream; $("#webcam-box").hidden = false; } catch (err) { alert("Webcam non disponibile: " + err.message); }
  });
  function stopCam() { stream?.getTracks().forEach(t => t.stop()); stream = null; $("#webcam-box").hidden = true; }
  $("#btn-webcam-stop").addEventListener("click", stopCam);
  $("#btn-shot").addEventListener("click", () => { const v = $("#video"); const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0); setImage(c.toDataURL("image/jpeg", 0.92)); stopCam(); });

  // ritmo: il menu della barra superiore e quello del passo 3 sono sincronizzati
  $("#pace").addEventListener("change", () => { $("#wz-pace").value = $("#pace").value; refreshSelectionUI(); });
  $("#wz-pace").addEventListener("change", () => { $("#pace").value = $("#wz-pace").value; refreshSelectionUI(); });
  $("#eval-limit").addEventListener("input", e => { const mx = +e.target.dataset.max || +e.target.max || 5000; if (+e.target.value > mx) e.target.value = mx; refreshSelectionUI(); });
  $("#eval-limit").addEventListener("change", e => { e.target.value = limitValue(); refreshSelectionUI(); });

  // ------------------------------------------------------------------ finestra di configurazione (dialog nativo)
  const wz = { dlg: $("#wizard"), step: 1, pushed: false };
  wz.open = (step = 1, mode = null) => {
    setMode(mode || state.mode);      // sempre: nasconde le opzioni dei dataset quando si analizza un'immagine
    if (!wz.dlg.open) {
      wz.dlg.showModal();
      // sui telefoni il tasto "indietro" chiude la finestra invece di lasciare la pagina (una sola voce nella cronologia)
      if (!history.state?.wizard) { history.pushState({ wizard: true }, ""); wz.pushed = true; }
    }
    wz.go(step);
  };
  wz.close = () => { if (wz.dlg.open) wz.dlg.close(); };
  wz.dlg.addEventListener("close", () => { stopCam(); closeWizardDoc(); if (wz.pushed) { wz.pushed = false; history.back(); } });
  // il tasto indietro chiude la finestra solo se la voce {wizard} è stata effettivamente tolta dalla cronologia
  window.addEventListener("popstate", e => { if (wz.dlg.open && !e.state?.wizard) { wz.pushed = false; wz.dlg.close(); } });
  // Esc: se è aperta la scheda "i" chiude prima quella, poi la finestra
  wz.dlg.addEventListener("cancel", e => { if (!$("#wz-doc").hidden) { e.preventDefault(); closeWizardDoc(); } });
  wz.go = step => {
    wz.step = Math.min(4, Math.max(1, step));
    $$(".wz-pane").forEach(p => p.classList.toggle("active", +p.dataset.step === wz.step));
    $$(".wz-step").forEach(b => { const on = +b.dataset.step === wz.step; b.classList.toggle("active", on); b.classList.toggle("done", +b.dataset.step < wz.step); if (on) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current"); });
    if (wz.step === 4) renderWizardSummary();
    updateWizardFooter();
    $(".wz-main").scrollTop = 0;
  };
  // cosa manca per poter procedere dal passo corrente
  function stepProblem(step) {
    if (step === 1) {
      if (state.mode === "image" && !state.image) return "Carica un'immagine, scattala con la webcam o scegline una di prova.";
      if (state.mode === "dataset" && !window.__eval?.current()) return "Scegli un dataset.";
    }
    if (step === 2 && !state.selected.size) return "Seleziona almeno un algoritmo.";
    return "";
  }
  function updateWizardFooter() {
    const prob = stepProblem(wz.step);
    const anyProb = [1, 2].map(stepProblem).find(Boolean);
    $("#wz-back").disabled = wz.step === 1;
    const next = $("#wz-next");
    const busyMsg = state.running ? "Un'analisi è già in corso: fermala o attendi la fine." : (window.__eval?.running() ? "Una valutazione è già in corso: fermala o attendi la fine." : "");
    if (wz.step < 4) { next.textContent = "Avanti ›"; next.disabled = !!prob; $("#wz-hint").textContent = prob; }
    else { next.textContent = state.mode === "dataset" ? "📊 Avvia valutazione" : "▶ Avvia analisi"; next.disabled = !!anyProb || !!busyMsg; $("#wz-hint").textContent = anyProb || busyMsg; }
  }
  $("#wz-back").addEventListener("click", () => wz.go(wz.step - 1));
  $("#wz-next").addEventListener("click", () => { if (wz.step < 4) wz.go(wz.step + 1); else startFromWizard(); });
  $$(".wz-step").forEach(b => b.addEventListener("click", () => wz.go(+b.dataset.step)));
  $("#wz-close").addEventListener("click", wz.close);
  $("#btn-configure").addEventListener("click", () => wz.open(state.mode === "image" && state.image ? 2 : 1));
  $$(".setup-row").forEach(b => b.addEventListener("click", () => wz.open(+b.dataset.step)));
  // modalità: immagine singola oppure dataset
  function setMode(mode) {
    state.mode = mode;
    $$("#wz-mode button").forEach(b => b.classList.toggle("on", b.dataset.mode === mode));
    $("#wz-image").hidden = mode !== "image"; $("#wz-dataset").hidden = mode !== "dataset";
    $("#opt-limit").hidden = mode !== "dataset"; $("#opt-force").hidden = mode !== "dataset";
    if (mode === "dataset" && !window.__eval?.loaded()) window.__eval?.load();
    refreshSelectionUI();
  }
  $$("#wz-mode button").forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));
  window.__wizard = { open: wz.open, close: wz.close, refresh: refreshSelectionUI, mode: () => state.mode };

  // passo 2: elenco "a pile" (accordion): famiglia → (per le CNN) dataset di addestramento o architettura → chip
  function renderWizardAlgos() {
    const byArch = $("#wz-byarch").checked;
    const box = $("#wz-algos"); box.innerHTML = "";
    const archOf = a => a.name.split(" · ")[0];
    const chip = (a, label) => {
      const lab = document.createElement("label"); lab.className = "chip"; lab.dataset.id = a.id; lab.title = a.short;
      lab.innerHTML = `<input type="checkbox" data-id="${a.id}"><span class="dot" style="background:${a.color}"></span><span class="name">${label}</span><span class="rel rel-${a.reliability}">${REL[a.reliability]}</span><button type="button" class="ibtn" title="Scheda di ${a.name}" aria-label="Scheda di ${a.name}">i</button>`;
      lab.querySelector("input").checked = state.selected.has(a.id);
      lab.querySelector("input").addEventListener("change", e => setSelected(a.id, e.target.checked));
      lab.querySelector(".ibtn").addEventListener("click", e => { e.preventDefault(); e.stopPropagation(); openDoc(a); });
      return lab;
    };
    const summaryEl = (title, ids, cls) => {
      const s = document.createElement("summary"); s.className = cls;
      s.innerHTML = `<span class="acc-title">${title}</span><span class="acc-cnt"></span><span class="acc-btns"><button type="button" class="acc-all">tutte</button><button type="button" class="acc-none">nessuna</button></span><span class="acc-arrow">›</span>`;
      s.querySelector(".acc-all").addEventListener("click", e => { e.preventDefault(); e.stopPropagation(); ids.forEach(id => state.selected.add(id)); refreshSelectionUI(); });
      s.querySelector(".acc-none").addEventListener("click", e => { e.preventDefault(); e.stopPropagation(); ids.forEach(id => state.selected.delete(id)); refreshSelectionUI(); });
      return s;
    };
    const oneOpen = (details, siblingsSel) => details.addEventListener("toggle", () => { if (details.open) $$(siblingsSel, details.parentElement).forEach(d => { if (d !== details) d.open = false; }); });
    FAMILY_ORDER.forEach((fam, fi) => {
      const items = state.analyzers.filter(a => a.family === fam); if (!items.length) return;
      const g = document.createElement("details"); g.className = "acc-g"; g.open = fi === 0;
      g.appendChild(summaryEl(FAMILY[fam], items.map(a => a.id), "acc-sum"));
      const body = document.createElement("div"); body.className = "acc-body";
      if (fam === "docente") {
        const keyOf = a => byArch ? archOf(a) : (a.subgroup || "senza pesi addestrati");
        const keys = [...new Set(items.map(keyOf))];
        keys.forEach((k, ki) => {
          const sub = items.filter(a => keyOf(a) === k);
          const d = document.createElement("details"); d.className = "acc-sub"; d.open = ki === 0;
          d.appendChild(summaryEl(byArch || k === "senza pesi addestrati" ? k : `addestrate su ${k}`, sub.map(a => a.id), "acc-sum sub"));
          const grid = document.createElement("div"); grid.className = "chips2";
          sub.forEach(a => grid.appendChild(chip(a, byArch ? a.name.replace(archOf(a) + " · ", "") : a.name.replace(" · " + a.subgroup, ""))));
          d.appendChild(grid); body.appendChild(d); oneOpen(d, ".acc-sub");
        });
      } else {
        const grid = document.createElement("div"); grid.className = "chips2";
        items.forEach(a => grid.appendChild(chip(a, a.name))); body.appendChild(grid);
      }
      g.appendChild(body); box.appendChild(g); oneOpen(g, ".acc-g");
    });
    refreshSelectionUI();
  }
  $("#wz-byarch").addEventListener("change", renderWizardAlgos);
  $$(".presets button").forEach(b => b.addEventListener("click", () => applyPreset(b.dataset.preset)));

  // passo 4: riepilogo
  function renderWizardSummary() {
    const st = selectionStats(); const box = $("#wz-summary");
    const famRows = FAMILY_ORDER.filter(f => st.fam[f]).map(f => `<div class="sum-fam"><b>${FAMILY[f]}</b> <span class="muted">(${st.fam[f]})</span><div class="sum-names">${st.sel.filter(a => a.family === f).map(a => `<span class="sum-name"><i style="background:${a.color}"></i>${a.name}</span>`).join("")}</div></div>`).join("");
    let source;
    if (state.mode === "dataset") { const d = window.__eval?.current(); source = d ? `<b>Dataset ${d.name}</b><div class="muted">${d.n_real} reali · ${d.n_attack} attacchi${d.cached ? " · risultati precedenti in cache" : ""}</div>` : `<span class="warn">Nessun dataset scelto</span>`; }
    else source = state.image ? `<img class="sum-thumb" src="${state.image}" alt=""><div><b>Immagine caricata</b><div class="muted">verrà ridotta e inviata al server solo per questa analisi</div></div>` : `<span class="warn">Nessuna immagine</span>`;
    const opts = `ritmo <b>${$("#pace").selectedOptions[0].textContent.toLowerCase()}</b>` + (state.mode === "dataset" ? ` · <b>${limitValue()}</b> immagini per classe${$("#eval-force").checked ? " · <b>ricalcolo forzato</b>" : " · punteggi in cache riusati"}` : "");
    box.innerHTML = `
      <div class="sum-row"><div class="sum-k">Sorgente</div><div class="sum-v sum-source">${source}</div></div>
      <div class="sum-row"><div class="sum-k">Algoritmi</div><div class="sum-v"><b>${st.sel.length}</b> su ${st.total} · <span class="ok">${st.nv} con verdetto</span>${st.nv ? ` (${st.part(WITH_VERDICT)})` : ""}${st.nn ? ` · <span class="warn">${st.nn} senza verdetto</span> (${st.part(NO_VERDICT)}: producono misure o card informative, non entrano nel consenso)` : ""}${famRows}</div></div>
      <div class="sum-row"><div class="sum-k">Opzioni</div><div class="sum-v">${opts}</div></div>`;
  }
  function startFromWizard() {
    if (state.mode === "dataset") { window.__eval?.start(limitValue(), $("#eval-force").checked); wz.close(); return; }
    if (!state.ws || state.ws.readyState !== 1) return alert("Connessione al server non attiva.");
    state.ws.send(JSON.stringify({ type: "analyze", image: state.image, analyzers: window.__selectedIds() }));
    wz.close();
  }

  // ------------------------------------------------------------------ barra di missione (card 1 durante e dopo l'analisi)
  function showMission(on) { $("#setup-idle").hidden = on; $("#mission").hidden = !on; $("#setup-title").textContent = on ? "Analisi in corso" : (state.mode === "dataset" ? "Nuova valutazione" : "Nuova analisi"); }
  // esito della corsa: done | cancelled | error | offline (per i testi di missione, barra di controllo e pannello finale)
  const OUTCOME = { done: "completata", cancelled: "interrotta", error: "interrotta per un errore", offline: "interrotta: connessione persa" };
  const verdictCount = () => Object.values(state.results).filter(r => r.score_real != null).length;
  function updateMission(final, reason = "done") {
    const ids = state.runIds || [];
    const sel = state.analyzers.filter(a => ids.includes(a.id));
    const nv = sel.filter(a => WITH_VERDICT.includes(a.reliability)).length;
    const n = verdictCount();
    $("#mission-title").textContent = final ? `Analisi ${OUTCOME[reason] || OUTCOME.done} · ${n} verdett${n === 1 ? "o" : "i"}${reason !== "done" ? " (parziali)" : ""}` : `Analisi in corso · ${sel.length} algoritm${sel.length === 1 ? "o" : "i"}`;
    $("#setup-title").textContent = final ? `Analisi ${reason === "done" ? "completata" : "interrotta"}` : "Analisi in corso";
    $("#mission-sub").textContent = `${nv} con verdetto · ${sel.length - nv} senza verdetto · ritmo ${$("#pace").selectedOptions[0].textContent.toLowerCase()}`;
    const fam = {}; sel.forEach(a => { fam[a.family] = (fam[a.family] || 0) + 1; });
    $("#mission-groups").innerHTML = FAMILY_ORDER.filter(f => fam[f]).map(f => `<span class="mg"><b>${fam[f]}</b> ${FAMILY[f]}</span>`).join("");
    $("#btn-stop").hidden = final;
  }
  $("#btn-stop").addEventListener("click", resetRun);
  $("#btn-new").addEventListener("click", () => { resetRun(); wz.open(1); });

  // ------------------------------------------------------------------ catalogo (scheda "Algoritmi")
  const cat = { family: "", q: "" };
  function renderCatalog() {
    const box = $("#catalog"); box.innerHTML = "";
    const q = cat.q.trim().toLowerCase();
    const items = state.analyzers.filter(a => (!cat.family || a.family === cat.family) && (!q || [a.name, a.short, a.reference, a.subgroup].join(" ").toLowerCase().includes(q)));
    $("#cat-count").textContent = `${items.length} algoritm${items.length === 1 ? "o" : "i"}`;
    for (const a of items) {
      const el = document.createElement("article"); el.className = "cat-card"; el.style.borderTopColor = a.color;
      el.innerHTML = `<header><b>${a.name}</b><label class="cat-check" title="Includi nella prossima analisi"><input type="checkbox" data-id="${a.id}" aria-label="Includi ${a.name} nella prossima analisi"> nell'analisi</label></header>
        <div class="badges"><span class="badge-f">${FAMILY[a.family]}</span><span class="badge-f rel-${a.reliability}">${REL[a.reliability]}</span>${a.subgroup ? `<span class="badge-f">addestrata su ${a.subgroup}</span>` : ""}</div>
        <p>${a.short}</p>
        <div class="ref muted">📄 ${a.reference}${a.reference_url ? ` · <a href="${a.reference_url}" target="_blank" rel="noopener">fonte</a>` : ""}</div>
        <div class="cat-foot"><button type="button" class="btn tiny">Scheda completa →</button><span class="muted">${a.graph.nodes.length} passi</span></div>`;
      el.querySelector("input").checked = state.selected.has(a.id);
      el.querySelector("input").addEventListener("change", e => setSelected(a.id, e.target.checked));
      el.querySelector(".btn").addEventListener("click", () => openDoc(a));
      box.appendChild(el);
    }
  }
  $$("#cat-family button").forEach(b => b.addEventListener("click", () => { cat.family = b.dataset.f; $$("#cat-family button").forEach(x => x.classList.toggle("on", x === b)); renderCatalog(); }));
  $("#cat-search").addEventListener("input", e => { cat.q = e.target.value; renderCatalog(); });

  // ------------------------------------------------------------------ websocket
  function connect() {
    const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    ws.onopen = () => { $("#conn").textContent = "connesso"; $("#conn").className = "pill pill-on"; };
    ws.onclose = () => {
      $("#conn").textContent = "offline"; $("#conn").className = "pill pill-off";
      // connessione persa a metà analisi: il server non manderà più run_end, si chiude qui la corsa
      if (state.running) { state.queue = []; endRun("offline"); }
      window.__evalHandler?.({ type: "eval_error", message: "connessione persa" });
      setTimeout(connect, 1500);
    };
    ws.onmessage = e => { try { enqueue(JSON.parse(e.data)); } catch (err) { console.error("Messaggio non valido", err); } };
    state.ws = ws; window.__ws = ws;
  }

  // coda con ritmo: gli eventi arrivano più veloci di quanto l'occhio segua, vengono mostrati con un ritardo scelto dall'utente
  function enqueue(ev) { state.queue.push(ev); if (!state.timer) drain(); }
  function drain() {
    state.timer = null;
    const pace = +$("#pace").value;
    // pagina nascosta o ritmo istantaneo: i timer del browser sono rallentati, si svuota subito la coda
    if (pace === 0 || document.hidden) { while (state.queue.length) safeHandle(state.queue.shift()); return; }
    const ev = state.queue.shift();
    if (!ev) return;
    safeHandle(ev);
    const delay = ev.type === "step" ? (ev.status === "running" ? pace : pace * 0.6) : 0;
    if (ev.type && ev.type.startsWith("eval_")) { state.timer = setTimeout(drain, 0); return; }
    state.timer = setTimeout(drain, delay);
  }
  function safeHandle(ev) { try { handle(ev); } catch (err) { console.error("Evento non gestito", ev, err); } }
  document.addEventListener("visibilitychange", () => { if (!state.timer && state.queue.length) drain(); });

  // ------------------------------------------------------------------ gestione eventi
  function handle(ev) {
    if (ev.type && ev.type.startsWith("eval_")) { window.__evalHandler?.(ev); return; }
    // eventi di una corsa annullata o resettata (arrivati dopo il Reset) vengono scartati
    if (ev.run_id && ev.run_id === state.ignoreRunId) return;
    if (ev.type !== "run_start" && ev.run_id && state.runId && ev.run_id !== state.runId) return;
    switch (ev.type) {
      case "run_start": startRun(ev); break;
      case "analyzer_start": setTabStatus(ev.analyzer, "running"); state.current.analyzer = ev.analyzer; if (state.follow) activateTab(ev.analyzer); updateCtrl(); break;
      case "step": onStep(ev); break;
      case "result": onResult(ev); break;
      case "run_end": if (ev.run_id === state.runId) endRun(ev.cancelled ? "cancelled" : "done"); break;
      case "error": alert(ev.message); if (state.running) endRun("error"); break;
    }
  }

  function startRun(ev) {
    state.running = true; state.runId = ev.run_id; state.ignoreRunId = null; state.results = {}; state.stepCount = 0; state.follow = true;
    state.current = { analyzer: null, step: null }; state.runIds = ev.analyzers;
    $("#ctrl").hidden = false; updateCtrl();
    $("#mission-thumb").src = ev.image; $("#mission-thumb").hidden = false; showMission(true); updateMission(false);
    $("#summary-empty").hidden = true; $("#summary").hidden = false; $("#results-body").innerHTML = "";
    $("#consensus-title").textContent = "Analisi in corso…"; $("#consensus-desc").textContent = ""; drawGauge(null);
    // una tab e un pannello per ogni analizzatore della corsa
    $$(".tab[data-tab]:not([data-tab=summary]):not([data-tab=eval]):not([data-tab=catalog])").forEach(t => t.remove());
    $$(".pane[data-pane]:not([data-pane=summary]):not([data-pane=eval]):not([data-pane=catalog])").forEach(p => p.remove());
    state.panes = {}; state.totalSteps = 0;
    for (const id of ev.analyzers) { const a = state.analyzers.find(x => x.id === id); if (!a) continue; buildPane(a); state.totalSteps += a.graph.nodes.length; }
    hideFinal(); showView(ev.image, "Immagine caricata", "In attesa del primo passo…", "");
    $("#progress-bar").style.width = "0%";
    activateTab("summary");
    if (wz.dlg.open) updateWizardFooter();
  }
  function endRun(reason = "done") {
    state.running = false; state.lastReason = reason;
    if (reason === "done") $("#progress-bar").style.width = "100%";
    updateConsensus(true); updateCtrl(); updateMission(true, reason); showFinal(reason);
    // a fine analisi si torna sempre al Riepilogo con l'elenco dei verdetti, anche in navigazione libera
    state.follow = true;
    activateTab("summary"); const sp = $(".pane[data-pane=summary]"); if (sp) sp.scrollTop = 0; window.scrollTo({ top: 0, behavior: "smooth" });
    if (wz.dlg.open) updateWizardFooter();
  }

  // ---------------------------------------------------------------- navigazione libera / riprendi / reset
  function setFollow(on) {
    if (state.follow === on) return;
    state.follow = on; updateCtrl();
    if (on) { if (state.current.analyzer) activateTab(state.current.analyzer); scrollToCurrent(); }
  }
  function scrollToCurrent() {
    const p = state.panes[state.current.analyzer]; const c = p?.cards[state.current.step];
    if (c) c.parentElement.scrollTo({ left: c.offsetLeft - 8, behavior: "smooth" });
    if (p) scrollGraphTo(p, state.current.step);
  }
  function scrollGraphTo(p, stepId) {
    const g = p.graph; const pos = g.pos?.[stepId]; if (!pos || !g.wrap) return;
    if (g.wrap.scrollWidth <= g.wrap.clientWidth) return;
    g.wrap.scrollTo({ left: Math.max(0, pos.x + g.NW / 2 - g.wrap.clientWidth / 2), behavior: "smooth" });
  }
  function updateCtrl() {
    const dot = $("#ctrl-dot"), st = $("#ctrl-status"), bf = $("#btn-follow");
    const a = state.analyzers.find(x => x.id === state.current.analyzer);
    const p = state.panes[state.current.analyzer];
    const where = a ? `<b>${a.name}</b>${p ? ` · passo ${p.done + (p.pane.querySelector(".card.running") ? 1 : 0)}/${a.graph.nodes.length}` : ""}` : "";
    if (!state.running) { const r = state.lastReason || "done"; const n = verdictCount(); dot.className = "ctrl-dot " + (r === "done" ? "done" : "free"); st.innerHTML = `Analisi ${OUTCOME[r] || OUTCOME.done} · ${n} verdett${n === 1 ? "o" : "i"}${r !== "done" ? (n === 1 ? " parziale" : " parziali") : ""}`; bf.hidden = true; return; }
    if (state.follow) { dot.className = "ctrl-dot"; st.innerHTML = "Segue l'analisi in corso · " + where; bf.hidden = true; }
    else { dot.className = "ctrl-dot free"; st.innerHTML = "Navigazione libera: l'analisi continua in background · " + where; bf.hidden = false; }
  }
  function resetRun() {
    if (state.ws?.readyState === 1 && state.running) state.ws.send(JSON.stringify({ type: "cancel" }));
    state.ignoreRunId = state.runId;      // gli eventi ancora in viaggio di questa corsa vanno ignorati
    state.queue = []; if (state.timer) { clearTimeout(state.timer); state.timer = null; }
    state.running = false; state.runId = null; state.results = {}; state.panes = {}; state.follow = true; state.current = { analyzer: null, step: null };
    $$(".tab[data-tab]:not([data-tab=summary]):not([data-tab=eval]):not([data-tab=catalog])").forEach(t => t.remove());
    $$(".pane[data-pane]:not([data-pane=summary]):not([data-pane=eval]):not([data-pane=catalog])").forEach(p => p.remove());
    $("#summary").hidden = true; $("#summary-empty").hidden = false; $("#results-body").innerHTML = "";
    $("#ctrl").hidden = true; activateTab("summary");
    $("#progress-bar").style.width = "0%";
    $("#view-a").classList.remove("show"); $("#view-b").classList.remove("show"); $("#view-empty").hidden = false;
    $("#vc-analyzer").textContent = "—"; $("#vc-step").textContent = ""; $("#vc-desc").textContent = "";
    hideFinal(); showMission(false); refreshSelectionUI();
  }
  $("#btn-follow").addEventListener("click", () => setFollow(true));
  $("#btn-reset").addEventListener("click", resetRun);
  // qualunque interazione dell'utente nella parte destra (tab, card, nodi, scroll) blocca la navigazione automatica
  $(".right").addEventListener("pointerdown", e => { if (state.running && !e.target.closest("#ctrl")) setFollow(false); }, true);
  $(".right").addEventListener("wheel", e => { if (state.running && (e.target.closest(".cards") || e.target.closest(".flow-wrap"))) setFollow(false); }, { capture: true, passive: true });

  // ------------------------------------------------------------------ pannello analizzatore
  function buildPane(a) {
    const tab = document.createElement("button"); tab.className = "tab"; tab.dataset.tab = a.id; tab.setAttribute("role", "tab"); tab.setAttribute("aria-selected", "false");
    tab.innerHTML = `<span class="st"></span>${a.name}<span class="badge" hidden></span>`;
    tab.addEventListener("click", () => activateTab(a.id));
    $("#tabs").appendChild(tab);

    const pane = document.createElement("section"); pane.className = "pane"; pane.dataset.pane = a.id;
    pane.innerHTML = `
      <div class="a-head" style="border-left-color:${a.color}">
        <div>
          <div><h3>${a.name}</h3><button type="button" class="ibtn lg" title="Come funziona, esempi e riferimenti">i</button> <span class="badges"><span class="badge-f">${FAMILY[a.family]}</span><span class="badge-f rel-${a.reliability}">${REL[a.reliability]}</span></span></div>
          <p>${a.short}</p>
          <div class="ref muted">📄 ${a.reference}${a.reference_url ? ` · <a href="${a.reference_url}" target="_blank" rel="noopener">fonte</a>` : ""}</div>
        </div>
        <div class="verdict-big"><div class="v unknown">in attesa</div><small>verdetto</small></div>
      </div>
      <div class="flow-wrap"><div class="flow-title"><span>Diagramma della pipeline</span><span class="flow-progress">0 / ${a.graph.nodes.length}</span></div><div class="flow-svg"></div></div>
      <div class="cards-wrap"><button class="gal-btn prev" title="Card precedenti">‹</button><div class="cards"></div><button class="gal-btn next" title="Card successive">›</button></div>`;
    $("#tab-panes").appendChild(pane);
    pane.querySelector(".a-head .ibtn").addEventListener("click", () => openDoc(a));
    const cardsEl = pane.querySelector(".cards");
    pane.querySelector(".gal-btn.prev").addEventListener("click", () => cardsEl.scrollBy({ left: -cardsEl.clientWidth * 0.8, behavior: "smooth" }));
    pane.querySelector(".gal-btn.next").addEventListener("click", () => cardsEl.scrollBy({ left: cardsEl.clientWidth * 0.8, behavior: "smooth" }));
    // la rotella verticale scorre la galleria in orizzontale
    cardsEl.addEventListener("wheel", e => { if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) { e.preventDefault(); cardsEl.scrollLeft += e.deltaY; } }, { passive: false });
    const graph = buildGraph(a.graph, pane.querySelector(".flow-svg"), a);
    state.panes[a.id] = { pane, tab, graph, cards: {}, done: 0, a };
  }

  function activateTab(id) {
    $$(".tab").forEach(t => { const on = t.dataset.tab === id; t.classList.toggle("active", on); t.setAttribute("aria-selected", on ? "true" : "false"); });
    $$(".pane").forEach(p => p.classList.toggle("active", p.dataset.pane === id));
    const t = $(`.tab[data-tab="${id}"]`); t?.scrollIntoView({ inline: "nearest", block: "nearest", behavior: "smooth" });
  }
  window.__activateTab = activateTab;
  function setTabStatus(id, st, text) {
    const p = state.panes[id]; if (!p) return;
    p.tab.querySelector(".st").className = "st " + st;
    const b = p.tab.querySelector(".badge"); if (text) { b.textContent = text; b.hidden = false; }
  }

  // ------------------------------------------------------------------ grafo SVG (layout a livelli)
  // Ogni nodo va nel livello = lunghezza del cammino più lungo dalle sorgenti (grafo aciclico); i livelli sono colonne.
  function buildGraph(g, container, a) {
    const NW = 148, NH = 42, GX = 46, GY = 16;
    const ids = g.nodes.map(n => n.id), level = {}, indeg = {};
    ids.forEach(id => { level[id] = 0; indeg[id] = 0; });
    g.edges.forEach(([s, t]) => indeg[t]++);
    const order = []; const q = ids.filter(id => indeg[id] === 0); const ind = { ...indeg };
    while (q.length) { const n = q.shift(); order.push(n); g.edges.filter(e => e[0] === n).forEach(([s, t]) => { level[t] = Math.max(level[t], level[s] + 1); if (--ind[t] === 0) q.push(t); }); }
    const cols = {}; ids.forEach(id => (cols[level[id]] ||= []).push(id));
    const nLevels = Object.keys(cols).length, maxRows = Math.max(...Object.values(cols).map(c => c.length));
    const W = nLevels * NW + (nLevels - 1) * GX + 16, H = maxRows * NH + (maxRows - 1) * GY + 16;
    const pos = {};
    Object.entries(cols).forEach(([l, arr]) => { const colH = arr.length * NH + (arr.length - 1) * GY; arr.forEach((id, i) => { pos[id] = { x: 8 + l * (NW + GX), y: 8 + (H - 16 - colH) / 2 + i * (NH + GY) }; }); });
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H); svg.classList.add("flow");
    svg.style.width = W + "px"; svg.style.height = H + "px"; svg.style.maxWidth = "none";
    const edgeEls = {};
    for (const [s, t] of g.edges) {   // archi come curve di Bézier tra il bordo destro del nodo sorgente e il sinistro del nodo destinazione
      const p = document.createElementNS(ns, "path"); const a1 = pos[s], b1 = pos[t];
      const x1 = a1.x + NW, y1 = a1.y + NH / 2, x2 = b1.x, y2 = b1.y + NH / 2, dx = (x2 - x1) / 2;
      p.setAttribute("d", `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`); p.classList.add("edge"); svg.appendChild(p);
      (edgeEls[t] ||= []).push(p);
    }
    const nodeEls = {};
    for (const n of g.nodes) {
      const grp = document.createElementNS(ns, "g"); grp.classList.add("node", n.kind || "process"); grp.setAttribute("transform", `translate(${pos[n.id].x},${pos[n.id].y})`);
      const r = document.createElementNS(ns, "rect"); r.setAttribute("width", NW); r.setAttribute("height", NH); r.setAttribute("rx", 11); grp.appendChild(r);
      if (n.kind === "model") { const ic = document.createElementNS(ns, "circle"); ic.setAttribute("cx", 14); ic.setAttribute("cy", NH / 2); ic.setAttribute("r", 4); ic.classList.add("icon"); grp.appendChild(ic); }
      const t = document.createElementNS(ns, "text"); t.setAttribute("x", n.kind === "model" ? 24 : 12); t.setAttribute("y", NH / 2 + 4); t.textContent = n.label.length > 19 ? n.label.slice(0, 18) + "…" : n.label; grp.appendChild(t);
      const ck = document.createElementNS(ns, "text"); ck.setAttribute("x", NW - 18); ck.setAttribute("y", NH / 2 + 5); ck.textContent = "✓"; ck.classList.add("check"); ck.style.fill = "var(--accent2)"; grp.appendChild(ck);
      const title = document.createElementNS(ns, "title"); title.textContent = n.label; grp.appendChild(title);
      // click su un nodo: porta in vista la card corrispondente
      grp.addEventListener("click", () => { const c = state.panes[a.id]?.cards[n.id]; if (c) { c.parentElement.scrollTo({ left: c.offsetLeft - 8, behavior: "smooth" }); c.classList.add("highlight"); setTimeout(() => c.classList.remove("highlight"), 1200); } });
      svg.appendChild(grp); nodeEls[n.id] = grp;
    }
    container.innerHTML = ""; container.appendChild(svg);
    const wrap = container.closest(".flow-wrap");
    wrap.addEventListener("wheel", e => { if (Math.abs(e.deltaY) > Math.abs(e.deltaX) && wrap.scrollWidth > wrap.clientWidth) { e.preventDefault(); wrap.scrollLeft += e.deltaY; } }, { passive: false });
    return { nodeEls, edgeEls, svg, pos, NW, wrap };
  }

  // ------------------------------------------------------------------ passi
  // Un evento "running" crea la card (con spinner) e accende il nodo; il successivo "done"/"error" la completa.
  function onStep(ev) {
    const p = state.panes[ev.analyzer]; if (!p) return;
    const node = p.graph.nodeEls[ev.step];
    if (node) { node.classList.remove("running", "done", "error"); node.classList.add(ev.status === "running" ? "running" : ev.status === "error" ? "error" : "done"); }
    (p.graph.edgeEls[ev.step] || []).forEach(e => { e.classList.toggle("flow", ev.status === "running"); e.classList.toggle("done", ev.status !== "running"); });

    let card = p.cards[ev.step];
    if (!card) {
      card = document.createElement("article"); card.className = "card running"; card.dataset.step = ev.step;
      const idx = Object.keys(p.cards).length + 1;
      card.innerHTML = `<div class="card-img"><div class="spinner"></div></div><div class="card-body"><div class="card-top"><h4><span class="idx">${String(idx).padStart(2, "0")}</span> ${ev.title}</h4><span class="card-time"></span></div><p>${ev.description}</p><div class="metrics"></div><div class="notes" hidden></div><div class="card-foot"><span class="more"></span><button type="button" class="btn tiny">Dettagli →</button></div></div>`;
      card.addEventListener("click", () => openDetail(p.a, card._data || ev));
      p.pane.querySelector(".cards").appendChild(card); p.cards[ev.step] = card;
    }
    if (ev.status === "running") {
      showView(null, p.a.name, ev.title, ev.description);
      state.current.step = ev.step; updateCtrl();
      if (state.follow) { const gal = p.pane.querySelector(".cards"); gal.scrollTo({ left: card.offsetLeft - 8, behavior: "smooth" }); scrollGraphTo(p, ev.step); }
    } else {
      card.classList.remove("running"); card.classList.toggle("error", ev.status === "error"); card._data = ev;
      const imgBox = card.querySelector(".card-img"); imgBox.innerHTML = "";
      if (ev.image) { const im = document.createElement("img"); im.src = ev.image; im.alt = ev.title; imgBox.appendChild(im); showView(ev.image, p.a.name, ev.title, ev.description); }
      else { imgBox.style.display = "none"; }
      card.querySelector(".card-time").textContent = ev.elapsed_ms >= 1000 ? (ev.elapsed_ms / 1000).toFixed(2) + " s" : ev.elapsed_ms.toFixed(1) + " ms";
      const entries = Object.entries(ev.metrics || {});
      const m = card.querySelector(".metrics"); m.innerHTML = entries.slice(0, 3).map(([k, v]) => `<span class="metric"><b>${k}</b> ${fmt(v)}</span>`).join("");
      card.querySelector(".more").textContent = entries.length > 3 ? `+${entries.length - 3} misure nel dettaglio` : "";
      if (ev.notes?.length) { const n = card.querySelector(".notes"); n.hidden = false; n.textContent = "⚠ " + ev.notes.join(" "); }
      p.done++; p.pane.querySelector(".flow-progress").textContent = `${p.done} / ${p.a.graph.nodes.length}`;
      state.stepCount++; $("#progress-bar").style.width = Math.min(100, 100 * state.stepCount / state.totalSteps) + "%";
    }
  }
  function fmt(v) { if (v == null) return "—"; if (Array.isArray(v)) return "[" + v.map(fmt).join(", ") + "]"; if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(Math.abs(v) < 1 ? 3 : 2); return String(v); }

  function onResult(ev) {
    state.results[ev.analyzer] = ev;
    const p = state.panes[ev.analyzer]; if (!p) return;
    const v = p.pane.querySelector(".verdict-big .v"); v.className = "v " + ev.label; v.textContent = ev.verdict;
    setTabStatus(ev.analyzer, ev.label, ev.label === "real" ? "reale" : ev.label === "attack" ? "attacco" : "n/d");
    const tr = document.createElement("tr"); const a = p.a; const s = ev.score_real;
    tr.innerHTML = `<td><span class="name-i"><span class="dot" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${a.color};flex:0 0 auto"></span><span>${a.name}</span><button type="button" class="ibtn" title="Scheda di ${a.name}" aria-label="Scheda di ${a.name}">i</button></span></td><td>${FAMILY[a.family]}</td><td><span class="rel rel-${ev.reliability}" style="font-size:11px;padding:2px 7px;border-radius:999px;border:1px solid var(--line)">${REL[ev.reliability] || ev.reliability}</span></td><td class="v-${ev.label}">${ev.verdict}</td><td>${s == null ? "—" : `<span class="bar"><i style="width:${(s * 100).toFixed(0)}%"></i></span>${(s * 100).toFixed(0)}%`}</td><td class="mono">${(ev.elapsed_ms / 1000).toFixed(2)} s</td>`;
    tr.querySelector(".ibtn").addEventListener("click", () => openDoc(a));
    $("#results-body").appendChild(tr);
    if (ev.explanation) { const tx = document.createElement("tr"); tx.className = "expl"; tx.innerHTML = `<td colspan="6"><span class="expl-dot" style="background:${a.color}"></span>${ev.explanation}</td>`; $("#results-body").appendChild(tx); }
    updateConsensus(false);
  }

  // consenso: media pesata dei P(reale) disponibili; è uno strumento di lettura, non una metrica
  function updateConsensus(final) {
    const rs = Object.values(state.results).filter(r => r.score_real != null);
    const w = r => RELW[r.reliability] ?? 0.5;
    if (!rs.length) { drawGauge(null); if (final) { $("#consensus-title").textContent = "Nessun verdetto affidabile"; $("#consensus-desc").textContent = "Gli algoritmi eseguiti non hanno prodotto un punteggio: la pipeline è stata dimostrata ma il verdetto non è determinabile."; } return; }
    const score = rs.reduce((s, r) => s + w(r) * r.score_real, 0) / rs.reduce((s, r) => s + w(r), 0);
    drawGauge(score);
    const real = score >= 0.5, nTr = rs.filter(r => r.reliability === "trained").length, nHe = rs.length - nTr;
    $("#consensus-title").textContent = (final ? "" : "Parziale: ") + (real ? "volto reale" : "attacco di presentazione") + ` (${(score * 100).toFixed(0)}%)`;
    const nZs = rs.filter(r => r.reliability === "zeroshot").length;
    $("#consensus-desc").textContent = `Media pesata di ${rs.length} verdetti (${nTr} da modelli addestrati, peso 1; ${nZs} zero-shot, peso 0,75; ${nHe - nZs} euristici, peso 0,5). I modelli non addestrati sono esclusi. Il consenso è uno strumento di lettura, non una metrica scientifica: in tesi contano APCER/BPCER misurate sui dataset.`;
  }
  function drawGauge(v, svgSel = "#gauge", valueSel = "#gauge-value") {
    const svg = $(svgSel); const arc = (a0, a1, r) => { const p = a => [100 + r * Math.cos(Math.PI * (1 - a)), 110 - r * Math.sin(Math.PI * (1 - a))]; const [x0, y0] = p(a0), [x1, y1] = p(a1); return `M${x0},${y0} A${r},${r} 0 ${a1 - a0 > .5 ? 1 : 0} 1 ${x1},${y1}`; };
    const gid = "gg-" + svgSel.slice(1);   // un gradiente per ogni svg: due id uguali nella pagina risolverebbero sempre al primo
    svg.innerHTML = `<defs><linearGradient id="${gid}" x1="0" x2="1"><stop offset="0" stop-color="#ff5c7a"/><stop offset=".5" stop-color="#ffb648"/><stop offset="1" stop-color="#3ddc97"/></linearGradient></defs>
      <path d="${arc(0, 1, 84)}" stroke="var(--line)" stroke-width="16" fill="none" stroke-linecap="round"/>
      ${v == null ? "" : `<path d="${arc(0, Math.max(0.02, v), 84)}" stroke="url(#${gid})" stroke-width="16" fill="none" stroke-linecap="round"/>`}`;
    $(valueSel).textContent = v == null ? "—" : (v * 100).toFixed(0) + "%";
  }
  // a fine analisi il pannello "Cosa sta guardando adesso" mostra consenso, verdetto e il collegamento ai risultati
  function showFinal(reason = "done") {
    const rs = Object.values(state.results).filter(r => r.score_real != null);
    const w = r => RELW[r.reliability] ?? 0.5;
    const score = rs.length ? rs.reduce((s, r) => s + w(r) * r.score_real, 0) / rs.reduce((s, r) => s + w(r), 0) : null;
    drawGauge(score, "#gauge-final", "#gauge-final-value");
    const t = $("#final-title");
    if (score == null) { t.textContent = "Nessun verdetto affidabile"; t.className = "unknown"; $("#final-desc").textContent = "Gli algoritmi eseguiti non hanno prodotto un punteggio."; }
    else {
      const real = score >= 0.5; t.textContent = real ? "Volto reale" : "Attacco di presentazione"; t.className = real ? "real" : "attack";
      const nReal = rs.filter(r => r.score_real >= 0.5).length;
      $("#final-desc").textContent = `${rs.length} verdetti pesati: ${nReal} per "reale", ${rs.length - nReal} per "attacco" · ${Object.keys(state.results).length - rs.length} senza punteggio. Il consenso è uno strumento di lettura, non una metrica.`;
    }
    $("#viewer-live").hidden = true; $("#final").hidden = false; $("#viewer-title").textContent = reason === "done" ? "Risultato" : "Risultato parziale";
  }
  function hideFinal() { $("#final").hidden = true; $("#viewer-live").hidden = false; $("#viewer-title").textContent = "Cosa sta guardando adesso"; }
  $("#btn-goto-results").addEventListener("click", () => { activateTab("summary"); const sp = $(".pane[data-pane=summary]"); if (sp) sp.scrollTop = 0; $(".right").scrollIntoView({ behavior: "smooth", block: "start" }); });

  // ------------------------------------------------------------------ viewer ("Cosa sta guardando adesso")
  // due <img> alternate con dissolvenza, così il cambio di immagine non lampeggia
  let viewToggle = false;
  function showView(img, analyzer, step, desc) {
    $("#view-empty").hidden = true;
    if (img) { const a = $("#view-a"), b = $("#view-b"); const nxt = viewToggle ? a : b, cur = viewToggle ? b : a; nxt.src = img; nxt.classList.add("show"); cur.classList.remove("show"); viewToggle = !viewToggle; }
    $("#vc-analyzer").textContent = analyzer; $("#vc-step").textContent = step; $("#vc-desc").textContent = desc;
  }

  // ------------------------------------------------------------------ finestre modali (scheda "i", glossario, dettaglio card)
  function showModal(innerHtml) {
    const opener = document.activeElement;     // per riportare il focus a chi ha aperto la finestra
    const m = document.createElement("div"); m.className = "modal"; m.setAttribute("role", "dialog"); m.setAttribute("aria-modal", "true"); m.innerHTML = innerHtml;
    const h = m.querySelector("h3"); if (h) m.setAttribute("aria-label", h.textContent);
    const close = () => { m.remove(); document.removeEventListener("keydown", onKey); opener?.focus?.(); };
    const onKey = e => { if (e.key === "Escape") close(); };
    m.addEventListener("click", e => { if (e.target === m || e.target.closest(".modal-close")) close(); });
    document.addEventListener("keydown", onKey);
    document.body.appendChild(m);
    m.querySelector(".modal-close")?.focus();
  }
  function docHtml(a) {
    const d = a.doc || {}; const passi = d.passi || {};
    const steps = a.graph.nodes.map((n, i) => { const s = passi[n.id]; if (!s) return ""; return `<div class="stp"><span class="n">${String(i + 1).padStart(2, "0")}</span><div><b>${n.label}</b><p>${s.spiegazione}</p>${s.esempio && s.esempio !== "—" ? `<div class="es">${s.esempio}</div>` : ""}</div></div>`; }).join("");
    const refs = (d.riferimenti || []).map(r => `<div class="ref">📄 ${r.testo}${r.url ? ` · <a href="${r.url}" target="_blank" rel="noopener">link</a>` : ""}</div>`).join("");
    return `<div class="kicker">${FAMILY[a.family]} · ${REL[a.reliability]}${a.subgroup ? ` · addestrata su ${a.subgroup}` : ""}</div><h3>${a.name}</h3>
      <p>${d.cosa_fa || a.short}</p>
      ${d.quando_funziona ? `<h4>Cosa fa bene</h4><ul>${d.quando_funziona.map(x => `<li>${x}</li>`).join("")}</ul>` : ""}
      ${d.limiti ? `<h4>Limiti</h4><ul>${d.limiti.map(x => `<li>${x}</li>`).join("")}</ul>` : ""}
      ${d.esempi?.length ? `<h4>Esempi</h4>${d.esempi.map(e => `<div class="ex"><b>${e.titolo}</b>${e.testo}</div>`).join("")}` : ""}
      ${steps ? `<h4>I passi della pipeline</h4><div class="steps">${steps}</div>` : ""}
      <h4>Riferimento</h4><div class="ref">📄 ${a.reference}${a.reference_url ? ` · <a href="${a.reference_url}" target="_blank" rel="noopener">fonte</a>` : ""}</div>${refs}`;
  }
  // la scheda "i": dentro la finestra di configurazione è un pannello a fianco, altrove una finestra modale
  function openDoc(a) {
    if (wz.dlg.open) { $("#wz-doc-body").innerHTML = docHtml(a); $("#wz-doc").hidden = false; $("#wz-doc").scrollTop = 0; return; }
    showModal(`<div class="modal-box doc-box"><div class="modal-body doc">${docHtml(a)}</div></div><button class="modal-close" title="Chiudi">✕</button>`);
  }
  function closeWizardDoc() { $("#wz-doc").hidden = true; }
  $("#wz-doc-close").addEventListener("click", closeWizardDoc);
  async function openGlossary() {
    const g = await (await fetch("/api/glossary")).json();
    showModal(`<div class="modal-box doc-box"><div class="modal-body doc"><div class="kicker">LivenessLab</div><h3>Glossario</h3><dl class="glossary">${g.map(x => `<dt>${x.termine}</dt><dd>${x.definizione}</dd>`).join("")}</dl></div></div><button class="modal-close" title="Chiudi">✕</button>`);
  }
  $("#btn-glossary").addEventListener("click", openGlossary);
  window.__showModal = showModal;
  function openDetail(a, ev) {
    const metrics = Object.entries(ev.metrics || {}).map(([k, v]) => `<span class="metric"><b>${k}</b> ${fmt(v)}</span>`).join("");
    showModal(`<div class="modal-box">
        <div class="modal-img">${ev.image ? `<img src="${ev.image}" alt="">` : `<span class="muted">nessuna immagine per questo passo</span>`}</div>
        <div class="modal-body"><div class="kicker">${a.name}</div><h3>${ev.title}</h3><p>${ev.description}</p>
          <div class="metrics">${metrics}</div>${ev.notes?.length ? `<div class="notes">⚠ ${ev.notes.join(" ")}</div>` : ""}
          <p class="muted" style="margin-top:14px;font-size:12px">Tempo di calcolo: ${ev.elapsed_ms >= 1000 ? (ev.elapsed_ms / 1000).toFixed(2) + " s" : (ev.elapsed_ms || 0).toFixed(1) + " ms"} · Esc o click fuori per chiudere</p></div></div>
      <button class="modal-close" title="Chiudi">✕</button>`);
  }

  $(".tab[data-tab=summary]").addEventListener("click", () => activateTab("summary"));
  $(".tab-catalog").addEventListener("click", () => activateTab("catalog"));
  $(".tab-eval").addEventListener("click", () => activateTab("eval"));
  loadCatalog(); connect();
})();

/* ===================================================================== Valutazione su dataset */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const ev = { datasets: [], results: {}, selected: null, running: false, loaded: false, limitFor: null, partial: null, loading: null };
  // colonne della tabella: chiave nel riepilogo del server -> intestazione (HTER non c'è: coinciderebbe con ACER finché la soglia non è scelta su un set di sviluppo)
  const METRICS = [["apcer", "APCER"], ["bpcer", "BPCER"], ["acer", "ACER"], ["eer", "EER"], ["bpcer_at_apcer10", "BPCER@APCER10%"], ["auc", "AUC"], ["accuracy", "Accuracy"]];
  const NAMES = () => Object.fromEntries((window.__analyzers || []).map(a => [a.id, a]));
  const pct = v => v == null ? "—" : (v * 100).toFixed(1) + "%";
  // classi di colore: per gli errori più basso è meglio, per AUC/accuracy più alto è meglio
  const cls = (k, v) => { if (v == null) return "na"; if (k === "auc" || k === "accuracy") return v >= 0.9 ? "good" : v >= 0.7 ? "mid" : "bad"; return v <= 0.1 ? "good" : v <= 0.25 ? "mid" : "bad"; };
  const current = () => ev.datasets.find(d => d.id === ev.selected);

  function load() {
    if (ev.loading) return ev.loading;         // una sola richiesta in volo, anche se la scheda e la finestra la chiedono insieme
    ev.loading = _load().finally(() => { ev.loading = null; });
    return ev.loading;
  }
  async function _load() {
    let d;
    try {
      let res = await fetch("/api/datasets", { cache: "no-store" });
      if (res.status === 503) { await new Promise(r => setTimeout(r, 3000)); res = await fetch("/api/datasets", { cache: "no-store" }); }   // server occupato: un solo nuovo tentativo
      d = await res.json();
    } catch (err) { console.error("dataset non caricati", err); $("#eval-results").innerHTML = `<div class="eval-block"><div class="note">Dataset non caricati: server non raggiungibile.</div></div>`; return; }
    if (!d || !d.datasets) { $("#eval-results").innerHTML = `<div class="eval-block"><div class="note">${d?.error || "Risposta non valida dal server."}</div></div>`; return; }
    await window.__analyzersReady;      // i nomi degli analizzatori servono per le tabelle: senza, le righe sarebbero vuote
    ev.datasets = d.datasets; ev.results = d.results || {}; ev.loaded = true;
    if (!ev.selected && ev.datasets.length) ev.selected = ev.datasets[0].id;
    renderDatasets(); renderCompare(); renderSelected(); syncLimit();
    if (ev.running) return;             // valutazione in corso: la tabella parziale non va coperta con quella in cache
    const r = ev.results[ev.selected];
    $("#eval-results").innerHTML = r ? "" : `<div class="eval-block"><div class="note">Nessun risultato per questo dataset: premi "Nuova valutazione".</div></div>`; if (r) renderResults(r);
    window.__wizard?.refresh();
  }
  // il campo "immagini per classe" segue il dataset scelto: massimo = la classe meno numerosa, valore di default = tutto
  function syncLimit() {
    const d = current(); const inp = $("#eval-limit"); if (!d || !inp) return;
    const max = Math.max(1, Math.min(d.n_real, d.n_attack));
    inp.max = max; inp.dataset.max = max;
    if (ev.limitFor !== d.id || +inp.value > max || !(+inp.value >= 1)) inp.value = max;   // il valore scelto resta finché il dataset non cambia
    ev.limitFor = d.id;
    const hint = $("#opt-limit .opt-h"); if (hint) hint.textContent = `${d.name}: ${d.n_real} reali e ${d.n_attack} attacchi disponibili. ${max} = tutto il sottoinsieme; un numero più basso per una prova rapida (campione distribuito su tutto l'elenco).`;
    window.__wizard?.refresh();
  }
  // pulsanti-dataset nella scheda: cambiano il dataset mostrato senza aprire la finestra; la spunta segna i risultati in cache
  function renderSelected() {
    const box = $("#ds-chips"); box.innerHTML = "";
    for (const d of ev.datasets) {
      const b = document.createElement("button"); b.type = "button"; b.className = "ds-chip" + (d.id === ev.selected ? " on" : "") + (ev.results[d.id] ? " has" : "");
      b.innerHTML = `${ev.results[d.id] ? "✓ " : ""}${d.name}`; b.title = ev.results[d.id] ? `${d.name}: risultati in cache` : `${d.name}: nessun risultato ancora (avvia una valutazione)`;
      b.addEventListener("click", () => selectDataset(d.id));
      box.appendChild(b);
    }
  }
  function selectDataset(id) {
    ev.selected = id; renderDatasets(); renderSelected(); syncLimit();
    if (ev.running && ev.partial && ev.partial.dataset === id) { renderResults(ev.partial); window.__wizard?.refresh(); return; }   // valutazione in corso: tabella parziale
    const r = ev.results[id]; $("#eval-results").innerHTML = r ? "" : `<div class="eval-block"><div class="note">Nessun risultato per questo dataset: premi "Nuova valutazione".</div></div>`; if (r) renderResults(r);
    window.__wizard?.refresh();
  }
  function renderDatasets() {
    const box = $("#ds-list"); box.innerHTML = "";
    for (const d of ev.datasets) {
      const el = document.createElement("button"); el.type = "button"; el.className = "ds" + (d.id === ev.selected ? " sel" : ""); el.title = d.note || ""; el.setAttribute("aria-pressed", d.id === ev.selected ? "true" : "false");
      el.innerHTML = `<b>${d.name}</b><span class="cnt">${d.n_real} reali · ${d.n_attack} attacchi${d.license ? ` · ${d.license}` : ""}</span><span class="src">${d.source || ""}</span>${d.note ? `<span class="nt">${d.note}</span>` : ""}${d.cached ? `<span class="cached">✓ risultati in cache</span>` : ""}`;
      el.addEventListener("click", () => selectDataset(d.id));
      box.appendChild(el);
    }
  }
  function renderResults(sum) { $("#eval-results").innerHTML = resultsHtml(sum); }
  function resultsHtml(sum) {
    const names = NAMES(); const ds = ev.datasets.find(d => d.id === sum.dataset);
    // righe con punteggio prima, poi quelle con la sola nota; dentro ogni gruppo l'ordine del catalogo
    const rows = Object.entries(sum.analyzers).filter(([aid]) => names[aid]).sort((a, b) => ((a[1].note ? 1 : 0) - (b[1].note ? 1 : 0)) || ((names[a[0]]?.order ?? 99) - (names[b[0]]?.order ?? 99)));
    let html = `<div class="eval-block"><h4>Risultati · ${ds?.name || sum.dataset}${sum.updated ? ` <span class="note">(aggiornato ${sum.updated})</span>` : ""}</h4>`;
    html += `<p class="explain">Una riga per algoritmo: quante immagini ha valutato (n), le metriche ISO/IEC 30107-3 e il tempo medio per immagine. APCER = attacchi accettati come reali, BPCER = persone vere rifiutate, ACER = la loro media; EER, BPCER@APCER10% e AUC non dipendono dalla soglia. Le righe senza numeri spiegano perché l'algoritmo non ha un punteggio (non addestrato, descrittivo, escluso dal proprio dataset).</p>`;
    html += `<div class="note">${ds?.note || ""}</div><div style="overflow-x:auto"><table class="mtab"><thead><tr><th>Algoritmo</th><th>n</th>${METRICS.map(m => `<th>${m[1]}</th>`).join("")}<th>ms/img</th></tr></thead><tbody>`;
    for (const [aid, m] of rows) {
      const a = names[aid] || { name: aid, color: "#888" };
      const dot = `<span class="dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${a.color};margin-right:6px"></span>`;
      if (m.note) { html += `<tr><td>${dot}${a.name}</td><td>${m.n}</td><td colspan="${METRICS.length + 1}" class="na" style="text-align:left;font-family:var(--font);white-space:normal">${m.note}</td></tr>`; continue; }
      const extra = (m.n_errors ? ` <span class="na" title="immagini su cui l'analizzatore ha dato errore">(${m.n_errors} err.)</span>` : "") + (m.n_noface ? ` <span class="na" title="immagini senza volto rilevato">(${m.n_noface} senza volto)</span>` : "");
      html += `<tr><td>${dot}${a.name}${extra}</td><td>${m.n}</td>${METRICS.map(([k]) => `<td class="${cls(k, m[k])}">${k === "auc" ? (m[k] == null ? "—" : m[k].toFixed(3)) : pct(m[k])}</td>`).join("")}<td>${m.elapsed_ms_mean ? m.elapsed_ms_mean.toFixed(0) : "—"}</td></tr>`;
    }
    html += `</tbody></table></div><div class="legend">Soglia fissa 0,5 (punteggio = probabilità di attacco) per APCER/BPCER/ACER/Accuracy; EER = punto in cui APCER e BPCER coincidono; BPCER@APCER10% = bona fide rifiutati alla soglia che accetta il 10% degli attacchi. Verde ≤ 10% di errore (AUC ≥ 0,9), giallo ≤ 25%, rosso oltre. "—" = non calcolabile. Gli analizzatori senza pesi e il Face Mesh non producono punteggi; le CNN con lo split 80/20 del docente non vengono valutate sul proprio dataset.</div>`;
    const rocs = rows.filter(([, m]) => m.roc);
    if (rocs.length) {
      html += `<h4>Curve ROC</h4><p class="explain">Una curva per algoritmo: al variare della soglia, sull'asse verticale la quota di attacchi rilevati (1 − APCER) e su quello orizzontale la quota di persone vere rifiutate (BPCER). Più la curva sta in alto a sinistra, meglio l'algoritmo separa le due classi; la diagonale tratteggiata corrisponde a una scelta casuale (AUC 0,5). L'AUC riassume la curva in un numero: 1 = separazione perfetta, 0,5 = caso.</p><div class="roc-wrap">` + rocs.map(([aid, m]) => rocSvg(names[aid] || { name: aid, color: "#888" }, m)).join("") + `</div>`;
    }
    html += `</div>`;
    return html;
  }
  // curva ROC: x = BPCER (bona fide rifiutati), y = 1 − APCER (attacchi rilevati), al variare della soglia
  function rocSvg(a, m) {
    const W = 180, H = 180, pad = 22;
    const pts = m.roc.map(([x, y]) => `${(pad + x * (W - 2 * pad)).toFixed(1)},${(H - pad - y * (H - 2 * pad)).toFixed(1)}`).join(" ");
    // la didascalia è HTML sotto il grafico (va a capo), non testo SVG (che verrebbe tagliato con i nomi lunghi)
    return `<div class="roc"><svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect x="${pad}" y="${pad}" width="${W - 2 * pad}" height="${H - 2 * pad}" fill="none" stroke="var(--line)"/><line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${pad}" stroke="var(--line)" stroke-dasharray="3 3"/><polyline points="${pts}" fill="none" stroke="${a.color}" stroke-width="2"/><text x="${W / 2}" y="${H - 6}" text-anchor="middle" fill="var(--muted)" font-size="8">BPCER →</text><text x="8" y="${H / 2}" fill="var(--muted)" font-size="8" transform="rotate(-90 8 ${H / 2})">1 − APCER →</text></svg><div class="roc-cap"><span class="dot" style="background:${a.color}"></span>${a.name} · AUC ${m.auc == null ? "—" : m.auc.toFixed(2)}</div></div>`;
  }
  function renderCompare() { $("#eval-compare").innerHTML = compareHtml(); }
  function compareHtml() {
    const ids = Object.keys(ev.results).sort((x, y) => (x === "samples") - (y === "samples")); const names = NAMES();
    if (ids.length < 2) return ids.length === 1 ? `<div class="eval-block"><div class="note">Valuta almeno un secondo dataset per vedere il confronto tra dataset.</div></div>` : "";
    const aids = [...new Set(ids.flatMap(id => Object.keys(ev.results[id].analyzers)))].filter(aid => names[aid] && ids.some(id => ev.results[id].analyzers[aid]?.acer != null)).sort((a, b) => ((names[a]?.order ?? 99) - (names[b]?.order ?? 99)));
    let html = `<div class="eval-block"><h4>Confronto tra dataset · ACER (soglia 0,5) e, tra parentesi, EER</h4><p class="explain">Una riga per algoritmo, una colonna per ogni dataset già valutato. Letta per riga, mostra quanto un metodo regge quando cambia il dataset: è la prova cross-dataset. Per le CNN e i classificatori addestrati, la cella del dataset su cui sono stati addestrati è il risultato intra-dataset, le altre sono cross-dataset. ACER e EER sono percentuali di errore: più basse, meglio.</p><div style="overflow-x:auto"><table class="mtab"><thead><tr><th>Algoritmo</th>${ids.map(id => `<th>${ev.datasets.find(d => d.id === id)?.name || id}</th>`).join("")}</tr></thead><tbody>`;
    for (const aid of aids) {
      const a = names[aid] || { name: aid, color: "#888" };
      html += `<tr><td><span class="dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${a.color};margin-right:6px"></span>${a.name}</td>` + ids.map(id => { const m = ev.results[id].analyzers[aid]; if (!m || m.acer == null) return `<td class="na">—</td>`; return `<td><span class="cmp-cell ${cls("acer", m.acer)}">${pct(m.acer)}</span> <span class="na">(${pct(m.eer)})</span> <span class="na cmp-n" title="immagini valutate">n=${m.n}</span></td>`; }).join("") + `</tr>`;
    }
    html += `</tbody></table></div><div class="legend">Un algoritmo che va bene su un dataset e male su un altro non generalizza: è il problema del cross-dataset descritto dal docente. Per le CNN, le righe "addestrata su A" lette lungo le colonne "testata su B" sono la matrice cross-dataset; la diagonale (stesso dataset) è l'intra-dataset.</div></div>`;
    return html;
  }
  // eventi eval_* dal websocket (inoltrati dal modulo principale)
  window.__evalHandler = m => {
    // la scheda mostra sempre il dataset che il server sta valutando: i pulsanti-dataset seguono la valutazione
    if (m.type === "eval_start") { ev.running = true; ev.partial = { dataset: m.dataset, analyzers: {} }; if (m.dataset && m.dataset !== ev.selected) { ev.selected = m.dataset; renderDatasets(); renderSelected(); syncLimit(); } renderResults(ev.partial); $("#eval-progress").hidden = false; $("#btn-eval-new").disabled = true; $("#btn-eval-stop").hidden = false; $("#eval-bar").style.width = "0%"; $("#eval-status").textContent = `${m.name}: ${m.todo} immagini da analizzare (${m.total - m.todo} già in cache) con ${m.analyzers.length} algoritmi…`; window.__activateTab?.("eval"); window.__wizard?.refresh(); }
    if (m.type === "eval_progress") { $("#eval-bar").style.width = (100 * m.done / Math.max(1, m.todo)) + "%"; const rate = m.done / Math.max(1, m.elapsed_s); const eta = (m.todo - m.done) / Math.max(rate, 0.01); $("#eval-status").textContent = `${m.done}/${m.todo} · ${m.current} · ${rate.toFixed(1)} img/s · restano ~${Math.round(eta)} s`; ev.partial = { dataset: m.dataset, analyzers: m.partial }; if (m.dataset === ev.selected) renderResults(ev.partial); }
    if (m.type === "eval_result") { ev.running = false; ev.partial = null; $("#btn-eval-new").disabled = false; $("#btn-eval-stop").hidden = true; $("#eval-bar").style.width = "100%"; $("#eval-status").textContent = (m.cancelled ? "Interrotta. " : "Completata. ") + `${m.elapsed_s.toFixed(0)} s`; ev.results = m.all; ev.datasets = m.datasets; if (m.dataset) ev.selected = m.dataset; renderDatasets(); renderSelected(); renderResults(m.summary); renderCompare(); window.__wizard?.refresh(); }
    if (m.type === "eval_error") { if (m.message === "connessione persa" && !ev.running) return; ev.running = false; ev.partial = null; $("#btn-eval-new").disabled = false; $("#btn-eval-stop").hidden = true; $("#eval-status").textContent = "Errore: " + m.message; window.__wizard?.refresh(); }
  };
  function start(limit, force) {
    const ws = window.__ws; if (!ws || ws.readyState !== 1) return alert("Connessione al server non attiva.");
    ws.send(JSON.stringify({ type: "evaluate", dataset: ev.selected, analyzers: window.__selectedIds ? window.__selectedIds() : null, limit: limit || null, force: !!force }));
    window.__activateTab?.("eval");
  }
  $("#btn-eval-stop").addEventListener("click", () => { window.__ws?.send(JSON.stringify({ type: "cancel" })); });
  $("#btn-eval-new").addEventListener("click", () => window.__wizard?.open(1, "dataset"));
  // esportazione: Excel dal server (file .xlsx), PDF con la stampa del browser (foglio di stile di stampa)
  const acts = $(".eval-actions");
  const xlsx = document.createElement("a"); xlsx.id = "btn-xlsx"; xlsx.className = "btn ghost"; xlsx.href = "/api/eval/export.xlsx"; xlsx.textContent = "⬇ Risultati completi (Excel)"; xlsx.title = "File .xlsx con un foglio per dataset (tutte le metriche), il confronto tra dataset, i punti delle curve ROC e le note"; xlsx.setAttribute("download", ""); acts.appendChild(xlsx);
  const pdf = document.createElement("button"); pdf.type = "button"; pdf.className = "btn ghost"; pdf.textContent = "🖨 Risultati completi (PDF)"; pdf.title = "Tabelle, curve ROC e confronto di tutti i dataset valutati: nella finestra di stampa scegli Salva come PDF"; acts.appendChild(pdf);
  pdf.addEventListener("click", () => {
    if (!Object.keys(ev.results).length) return alert("Nessun risultato da esportare: valuta prima un dataset.");
    window.__activateTab?.("eval");
    const when = new Date().toLocaleString("it-IT");
    $("#print-all").innerHTML = `<h2 class="print-title">LivenessLab · Valutazione su dataset</h2><p class="note">Generato il ${when} · metriche ISO/IEC 30107-3, soglia 0,5 (punteggio = probabilità di attacco)</p>` +
      Object.keys(ev.results).sort((x, y) => (x === "samples") - (y === "samples")).map(id => resultsHtml(ev.results[id])).join("") + compareHtml();
    document.body.classList.add("print-eval"); setTimeout(() => window.print(), 80);
  });
  window.addEventListener("afterprint", () => { document.body.classList.remove("print-eval"); $("#print-all").innerHTML = ""; });
  window.__eval = { load, start, current, running: () => ev.running, loaded: () => ev.loaded, datasets: () => ev.datasets };
  $(".tab-eval").addEventListener("click", () => { if (!ev.loaded) load(); });
  load();
})();

// ------------------------------------------------------------------ 3. PWA: service worker, installazione, guida iOS
(() => {
  const $ = s => document.querySelector(s);
  // localStorage può mancare o lanciare (navigazione privata): ogni accesso è protetto
  const store = { get(k) { try { return localStorage.getItem(k); } catch { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignorato */ } } };
  const standalone = window.matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
  const ua = navigator.userAgent;
  const isIOS = /iPad|iPhone|iPod/.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const isIOSSafari = isIOS && /Safari/.test(ua) && !/CriOS|FxiOS|EdgiOS|OPiOS/.test(ua);
  const isAndroid = /Android/.test(ua);
  const DISMISS_DAYS = 14;                       // dopo "Non ora" l'avviso non ricompare per due settimane
  const banner = $("#install-banner"), btn = $("#btn-install"), sub = $("#install-sub"), yes = $("#install-yes");
  let deferred = null;                           // evento beforeinstallprompt trattenuto (Android, Chrome, Edge)

  function dismissed() { const t = Number(store.get("ll.install.dismissed") || 0); return Date.now() - t < DISMISS_DAYS * 864e5; }
  function showBanner(kind) {
    if (standalone || dismissed() || !banner.hidden) return;
    banner.dataset.kind = kind;
    sub.textContent = kind === "ios" ? "Su iPhone e iPad si installa da Safari in tre tocchi: ti mostro come." : "Si apre a schermo intero come un'app, con la sua icona nella schermata Home.";
    yes.textContent = kind === "ios" ? "Come si fa" : "Installa";
    banner.hidden = false;
  }
  function hideBanner() { banner.hidden = true; }

  // ----- service worker: shell in cache, avviso quando arriva una versione nuova
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js").then(reg => {
      reg.addEventListener("updatefound", () => {
        const w = reg.installing;
        w?.addEventListener("statechange", () => { if (w.state === "installed" && navigator.serviceWorker.controller) showUpdate(); });
      });
    }).catch(() => { /* senza HTTPS o senza supporto: l'app funziona comunque dal browser */ }));
  }
  function showUpdate() {
    if ($("#update-pill")) return;
    const p = document.createElement("div"); p.id = "update-pill"; p.className = "update-pill"; p.setAttribute("role", "status");
    p.innerHTML = `<span>È disponibile una nuova versione di LivenessLab.</span><button class="btn small primary" type="button">Ricarica</button>`;
    p.querySelector("button").addEventListener("click", () => location.reload());
    document.body.appendChild(p);
  }

  // ----- installazione diretta (Android, Chrome, Edge): il browser propone l'evento, noi mostriamo il nostro avviso
  window.addEventListener("beforeinstallprompt", e => { e.preventDefault(); deferred = e; btn.hidden = false; showBanner("android"); });
  window.addEventListener("appinstalled", () => { deferred = null; hideBanner(); btn.hidden = true; });
  // ----- iPhone e iPad: nessun evento, si passa dal menu Condividi di Safari: avviso con la guida
  if (!standalone) {
    btn.hidden = false;                          // il pulsante "Installa" apre l'installazione o la guida
    if (isIOSSafari) setTimeout(() => showBanner("ios"), 1500);
  }

  async function install() {
    if (deferred) {
      deferred.prompt();
      const r = await deferred.userChoice.catch(() => ({ outcome: "dismissed" }));
      deferred = null;
      if (r.outcome === "accepted") { hideBanner(); btn.hidden = true; } else store.set("ll.install.dismissed", String(Date.now()));
      return;
    }
    hideBanner(); openGuide();
  }
  yes.addEventListener("click", install);
  $("#install-no").addEventListener("click", () => { store.set("ll.install.dismissed", String(Date.now())); hideBanner(); });
  btn.addEventListener("click", install);

  // ----- guida all'installazione: la piattaforma corrente per prima
  function openGuide() {
    const ios = `<h4>iPhone e iPad (Safari)</h4><ol>
      <li>Apri <b>${location.host}</b> in <b>Safari</b> (da altri browser il passo 3 non compare).</li>
      <li>Tocca il pulsante <b>Condividi</b> <span class="kbd">⎋</span> (il quadrato con la freccia verso l'alto, in basso al centro su iPhone, in alto su iPad).</li>
      <li>Scorri l'elenco e tocca <b>Aggiungi alla schermata Home</b>.</li>
      <li>Conferma con <b>Aggiungi</b>: l'icona di LivenessLab compare nella schermata Home e l'app si apre a schermo intero.</li></ol>`;
    const android = `<h4>Android (Chrome, Edge, Samsung Internet)</h4><ol>
      <li>Apri <b>${location.host}</b> nel browser.</li>
      <li>Tocca <b>Installa</b> nell'avviso in basso, oppure apri il menu <b>⋮</b> e scegli <b>Installa app</b> (o <b>Aggiungi a schermata Home</b>).</li>
      <li>Conferma: l'icona compare nella schermata Home e nell'elenco delle app.</li></ol>`;
    const desktop = `<h4>Computer (Chrome, Edge)</h4><ol>
      <li>Apri <b>${location.host}</b> nel browser.</li>
      <li>Fai clic sull'icona di installazione <span class="kbd">⊕</span> a destra nella barra dell'indirizzo, oppure sul pulsante <b>📲 Installa</b> in alto, oppure nel menu <b>⋮</b> scegli <b>Installa LivenessLab</b>.</li>
      <li>L'app si apre in una finestra propria, senza barra dell'indirizzo, e compare tra le applicazioni.</li></ol>`;
    const order = isIOS ? [ios, android, desktop] : isAndroid ? [android, ios, desktop] : [desktop, android, ios];
    window.__showModal(`<div class="modal-box doc-box"><div class="modal-body doc guide"><div class="kicker">LivenessLab · app</div><h3>Installare LivenessLab come app</h3>
      <p>LivenessLab è una <b>progressive web app</b>: si installa dal browser, senza store, e si apre a schermo intero con la sua icona. I calcoli restano sul server: senza connessione l'app si apre ma non può analizzare.</p>
      ${order.join("")}
      <p class="muted" style="margin-top:14px;font-size:12px">Versione installata: v${document.querySelector(".version")?.textContent.replace(/^v/, "") || "?"} · Per rimuoverla si elimina l'icona come per qualsiasi app.</p>
      </div></div><button class="modal-close" title="Chiudi">✕</button>`);
  }
  window.__pwa = { openGuide, showBanner };      // usato dal manuale (schermate) e per prove
})();
