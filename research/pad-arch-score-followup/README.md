# Materiale per Valerio Cassano

**Messaggio per la scheda C2**

Ciao Valerio, grazie per le tabelle e per i log: ci hanno permesso di ricostruire l’epoca ripristinata per ciascuno dei 48 checkpoint e di capire quanto pesa il caso fold 3, seed 44. Per il prossimo passo ti chiederei di verificare le quattro liste originali NUAA e l'associazione degli identificativi tra client e imposter. Abbiamo 14 etichette presenti in entrambe le classi e due presenti in una sola classe, entrambe soltanto nella sessione 3. Mi occupo io del chiarimento con i curatori del dataset.

Leggi poi PROTOCOL.md e indicami eventuali differenze rispetto ai file che conservi. Prima di leggere nuovi risultati fissiamo la versione del protocollo e dello script in un commit datato. Inizia dai punteggi già salvati. Il programma allegato produce AUROC, EER, DET, gli errori alle soglie selezionate sul dev e le analisi di sensibilità. Per questa fase non serve riaddestrare. Restituisci i soli risultati aggregati e segnala qualunque mancata riproduzione. Le nuove stime BN con augmentation e dropout sono una seconda fase. Il confronto con le tre AttackNet prosegue nella scheda dedicata. Grazie!

## Run on the approved local machine

This is a proposal and a script for local execution. Artificial fixtures were used to check the code. No current study score file was processed. The manuscript has a separate source ZIP.

Use Python 3, NumPy, SciPy and Matplotlib. Validate the checksum manifests of the original evaluation and both training folders. Point the first three arguments at the existing restricted subdirectories; the outputs are aggregate only. The bundled units.csv is the same reader-facing comparison reference supplied with the manuscript.

~~~sh
python check_synthetic.py

python score_analysis.py \
  --test /path/to/20261121-ARCHEVAL-332fbdf/restricted \
  --dev4 /path/to/20261120-ARCHRUN-p4-7faa49f/restricted \
  --dev12 /path/to/20261120-ARCHRUN-p12-ea7fdca/restricted \
  --units units.csv \
  --out /path/to/new/score-summary
~~~

Expected outputs are score_metrics.csv (96 condition rows), paired_summary.csv (24 summary rows), det_grid.csv (1,152 aggregate coordinates), det_curves.pdf and sensitivity.json. The primary aggregate must reproduce +9.983325 percentage points. Report any missing or mismatching input to the supervisor before interpreting estimates. The mechanistic re-estimation is a separate implementation described in Section 4 of PROTOCOL.md.
