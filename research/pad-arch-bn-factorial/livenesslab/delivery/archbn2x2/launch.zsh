#!/bin/zsh
# Lancio della prova 2x2 (check-inputs o trial) dalla radice del repository, con le condizioni registrate, come
# delivery/aneteval/launch.zsh. Da una zsh con setopt nobgnice e l'alimentatore collegato:
#
#   setopt nobgnice
#   ARCHBN2X2_LOG_DIR=<cartella fuori dal repository> zsh delivery/archbn2x2/launch.zsh check-inputs
#   ARCHBN2X2_LOG_DIR=<cartella fuori dal repository> ARCHBN2X2_ATTEMPT=1 nohup zsh delivery/archbn2x2/launch.zsh trial > <log fuori dal repository> 2>&1 &
#
# Scrive in ARCHBN2X2_LOG_DIR <etichetta>-conditions.txt (sistema; powermode e lowpowermode; alimentazione; nice dello
# script; numero di variabili TF_NUM_ impostate; se TMPDIR è quello predefinito di macOS, cartelle delle viste lasciate e
# GB liberi della cartella temporanea; commit e righe di git status --porcelain; data, carico, uso della CPU, alimentazione,
# powermode e memoria all'inizio e alla fine; uscita di check-inputs; comando ed esito) e <etichetta>.console.log (l'uscita del
# comando). trial non parte se il nice non è 0, se pmset -g batt non dice «AC Power», se powermode non è 0 (sui Mac recenti
# la modalità a basso consumo è powermode 1 e la voce lowpowermode può mancare), se lowpowermode è 1, se nella cartella
# temporanea resta una cartella archbn2x2-views-*, se la cartella non è la radice di un repository git, se git status
# --porcelain fallisce o non è vuoto o se check-inputs fallisce. Un mancato avvio non occupa il nome del tentativo: il
# motivo va in <etichetta>-not-launched-<data e ora>-conditions.txt, e lo stesso tentativo si può rilanciare. Lo stesso
# vale se trial esce con errore senza aver stampato la riga «== 2x2 trial » (la stampa subito dopo aver creato la cartella
# del run e il primo status.json): file delle condizioni e console vanno sui nomi «not-launched» e l'uscita è quella di
# trial. Una prova che ha stampato la riga d'avvio usa il tentativo.
# Il file delle condizioni non contiene percorsi: solo conteggi, sì/no, numeri e l'uscita dei comandi.
setopt nobgnice
set -u
cmd=${1:-}
log_dir=${ARCHBN2X2_LOG_DIR:-}
attempt=${ARCHBN2X2_ATTEMPT:-}
py=.venv/bin/python
if [[ ! -f scripts/pad_arch_bn_factorial.py ]]; then print -u2 "run from the root of the repository"; exit 9; fi
if [[ -z $log_dir ]]; then print -u2 "ARCHBN2X2_LOG_DIR is required"; exit 9; fi
case $cmd in
  check-inputs) label="check-inputs-$(date '+%Y%m%d-%H%M%S')"; args=(check-inputs);;
  trial) if [[ -z $attempt ]]; then print -u2 "trial needs ARCHBN2X2_ATTEMPT"; exit 9; fi
         label="trial-attempt$attempt"; args=(trial --attempt $attempt --i-am-authorized);;
  *) print -u2 "usage: launch.zsh check-inputs | trial"; exit 9;;
esac
repo=${PWD:A}
log_dir=${log_dir:A}
if [[ $log_dir == $repo || $log_dir == $repo/* ]]; then print -u2 "ARCHBN2X2_LOG_DIR must be outside the repository"; exit 9; fi
mkdir -p $log_dir
C=$log_dir/$label-conditions.txt
N=$log_dir/$label-not-launched-$(date '+%Y%m%d-%H%M%S')-conditions.txt
if [[ -e $C ]]; then print -u2 "$label-conditions.txt already exists"; exit 9; fi
head_line="Conditions of $cmd${attempt:+, attempt $attempt}, written by delivery/archbn2x2/launch.zsh (not by the trial)."
# rifiuto prima del lancio: il motivo (e il valore letto) in un file «not launched», non in quello del tentativo
refuse() {
  { print -r -- $head_line; print -r -- "$1"; print "== not launched: $2"; } > $N
  print -u2 "not launched: $2 (see ${N:t})"
  exit 1
}
tmp=${TMPDIR:-/tmp}
nice_now=$(ps -o nice= -p $$ | tr -d ' ')
[[ $nice_now == 0 ]] || refuse "== nice of the launching shell"$'\n'"$nice_now" "the nice value is not 0 (launch from a zsh with setopt nobgnice)"
if [[ $cmd == trial ]]; then
  power=$(pmset -g batt | head -1)
  [[ $power == *"'AC Power'"* ]] || refuse "== power (pmset -g batt: power source)"$'\n'"$power" "trial needs the power adapter (pmset -g batt does not say 'AC Power')"
  lpm=$(pmset -g | awk '$1 == "lowpowermode" {print $2}')
  [[ ${lpm:-0} == 0 ]] || refuse "== lowpowermode"$'\n'"$lpm" "low power mode is on"
  pm=$(pmset -g | awk '$1 == "powermode" {print $2}')
  [[ ${pm:-0} == 0 ]] || refuse "== powermode"$'\n'"$pm" "powermode is not 0"
  left=$(ls $tmp | grep -c '^archbn2x2-views-')
  [[ $left == 0 ]] || refuse "== views folders left in TMPDIR"$'\n'"$left" "a views folder of an earlier trial is left in TMPDIR"
  # la prova rifiuta da sola un albero non pulito; fermarsi qui lascia libero il tentativo. Fuori da un repository git, o
  # se la radice del repository git non è questa cartella (anche se una cartella che la contiene è un repository), ci si
  # ferma comunque: l'esito non dipende da ciò che sta intorno
  top=$(git rev-parse --show-toplevel 2>/dev/null)
  [[ -n $top && ${top:A} == $repo ]] || refuse "== root of a git work tree"$'\n'"no" "the folder is not the root of a git work tree"
  # git status che fallisce (indice illeggibile…) è un rifiuto esplicito, non un albero pulito; con lo stato vuoto il
  # conteggio è 0 (print -rn non stampa nulla), altrimenti una riga per voce
  st=$(git status --porcelain 2>/dev/null) || refuse "== git status --porcelain"$'\n'"failed" "git status failed"
  dirty=$(print -rn -- "$st" | grep -c '')
  [[ $dirty == 0 ]] || refuse "== lines of git status --porcelain"$'\n'"$dirty" "the work tree is not clean"
fi
# condizioni registrate prima e dopo il comando
conditions() {
  date '+%Y-%m-%d %H:%M:%S %z'
  uptime | sed 's/.*load/load/'
  top -l 1 -n 0 | grep 'CPU usage'
  pmset -g batt | head -1
  pmset -g | awk '$1 == "powermode" || $1 == "lowpowermode" {print $1, $2}'
  memory_pressure | tail -1
}
{
  print -r -- $head_line
  print "== conditions before"
  conditions
  print "== system"
  sw_vers | tr '\t' ' '
  print "== power (pmset -g: powermode, lowpowermode; pmset -g batt: power source)"
  pmset -g | grep -E 'powermode|lowpowermode' | tr -s ' \t' ' '
  pmset -g batt | head -1
  print "== nice of the launching shell"
  print $nice_now
  print "== environment: number of TF_NUM_ variables set"
  env | grep -c '^TF_NUM_'
  print "== temporary folder (default of macOS: yes/no; views folders left; free GB)"
  [[ ${tmp:A} == ${$(getconf DARWIN_USER_TEMP_DIR 2>/dev/null):A} ]] && print yes || print no
  ls $tmp | grep -c '^archbn2x2-views-'
  df -g $tmp | tail -1 | awk '{print $4}'
  print "== git (commit, lines of git status --porcelain)"
  git rev-parse --short=7 HEAD 2>/dev/null
  git status --porcelain 2>/dev/null | wc -l | tr -d ' '
  print "== check-inputs"
  $py -I -B scripts/pad_arch_bn_factorial.py check-inputs 2>&1
  print "check-inputs exit $?"
} > $C
if ! grep -q '^check-inputs exit 0$' $C; then
  print "== not launched: check-inputs failed" >> $C
  mv $C $N
  print -u2 "check-inputs failed: see ${N:t}"
  exit 1
fi
[[ $cmd == check-inputs ]] && exit 0
print "== command" >> $C
print "caffeinate -dims .venv/bin/python -I -B scripts/pad_arch_bn_factorial.py ${args[*]}" >> $C
console=$log_dir/$label.console.log
caffeinate -dims $py -I -B scripts/pad_arch_bn_factorial.py "${args[@]}" > $console 2>&1
rc=$?
{
  print "$cmd exit $rc"
  print "== conditions after"
  conditions
  print "== views folders left in TMPDIR"
  ls $tmp | grep -c '^archbn2x2-views-'
} >> $C
# la prova stampa «== 2x2 trial <run id>» subito dopo aver creato la cartella del run e il primo status.json: se è uscita
# con errore senza quella riga, si è rifiutata prima (albero non pulito, nota nulla, cartella esistente…) e il tentativo
# resta libero; se la riga c'è, la prova è partita e il tentativo è usato
if (( rc != 0 )) && ! grep -q '^== 2x2 trial ' $console; then
  print "== not started: the trial refused before creating its run folder" >> $C
  mv $C $N; [[ -e $console ]] && mv $console ${N%-conditions.txt}.console.log
  print -u2 "not started: the trial refused before creating its run folder (see ${N:t})"
fi
exit $rc
