#!/bin/zsh
# Lancio di check-estimator o di un run della C5, dalla radice del repository, con le condizioni registrate. Da una
# shell zsh con setopt nobgnice (altrimenti un lavoro in background parte a nice 5 e lo script si ferma):
#
#   setopt nobgnice
#   ANETRUN_LOG_DIR=<cartella fuori dal repository> ANETRUN_ATTEMPT=1 nohup zsh delivery/anetrun/launch.zsh check-estimator > <log fuori dal repository> 2>&1 &
#   ANETRUN_LOG_DIR=<cartella fuori dal repository> ANETRUN_ATTEMPT=1 ANETRUN_ESTIMATOR=results/anetrun/<id> nohup zsh delivery/anetrun/launch.zsh run anet_v1_p4 > <log fuori dal repository> 2>&1 &
#
# Scrive in ANETRUN_LOG_DIR, per il comando, <etichetta>-conditions.txt (sistema, modalità di alimentazione e alimentatore
# da pmset, nice della shell, variabili TF_NUM_ impostate, commit e righe di git status --porcelain, carico e uso della
# CPU all'inizio e alla fine, uscite di check-machine e check-inputs, comando ed esito) e <etichetta>.console.log. Se
# check-machine o check-inputs falliscono, o se il nice dello script non è 0, il comando non parte. Nessun percorso
# personale nel file delle condizioni.
setopt nobgnice
set -u
cmd=${1:-}
run=${2:-}
log_dir=${ANETRUN_LOG_DIR:-}
attempt=${ANETRUN_ATTEMPT:-}
est=${ANETRUN_ESTIMATOR:-}
py=.venv/bin/python
if [[ ! -f scripts/pad_anet_run.py ]]; then print -u2 "run from the root of the repository"; exit 9; fi
if [[ -z $log_dir || -z $attempt ]]; then print -u2 "ANETRUN_LOG_DIR and ANETRUN_ATTEMPT are required"; exit 9; fi
case $cmd in
  check-estimator) label="check-estimator-attempt$attempt"; args=(check-estimator --attempt $attempt --i-am-authorized);;
  run) if [[ -z $run || -z $est ]]; then print -u2 "run needs the run name and ANETRUN_ESTIMATOR"; exit 9; fi
       label="$run-attempt$attempt"; args=(run --run $run --estimator-check $est --attempt $attempt --i-am-authorized);;
  *) print -u2 "usage: launch.zsh check-estimator | run <run name>"; exit 9;;
esac
repo=${PWD:A}
log_dir=${log_dir:A}
if [[ $log_dir == $repo || $log_dir == $repo/* ]]; then print -u2 "ANETRUN_LOG_DIR must be outside the repository"; exit 9; fi
mkdir -p $log_dir
C=$log_dir/$label-conditions.txt
if [[ -e $C ]]; then print -u2 "$label-conditions.txt already exists"; exit 9; fi
nice_now=$(ps -o nice= -p $$ | tr -d ' ')
if [[ $nice_now != 0 ]]; then
  {
    print "Conditions of $cmd${run:+ $run}, attempt $attempt, written by delivery/anetrun/launch.zsh (not by the runner)."
    print "== nice of the launching shell"
    print $nice_now
    print "== not launched: the nice value is not 0 (launch from a zsh with setopt nobgnice)"
  } > $C
  print -u2 "the nice value of the launch is $nice_now, not 0: launch from a zsh with setopt nobgnice"
  exit 1
fi
conditions() {
  date '+%Y-%m-%d %H:%M:%S %z'
  uptime
  top -l 1 -n 0 | grep 'CPU usage'
}
{
  print "Conditions of $cmd${run:+ $run}, attempt $attempt, written by delivery/anetrun/launch.zsh (not by the runner)."
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
  print "== git (commit, lines of git status --porcelain)"
  git rev-parse --short=7 HEAD
  git status --porcelain | wc -l | tr -d ' '
  print "== check-machine"
  $py -I -B scripts/pad_anet_run.py check-machine 2>&1
  print "check-machine exit $?"
  print "== check-inputs"
  ci=(check-inputs)
  [[ -n $est ]] && ci+=(--estimator-check $est)
  $py -I -B scripts/pad_anet_run.py "${ci[@]}" 2>&1
  print "check-inputs exit $?"
} > $C
if ! grep -q '^check-machine exit 0$' $C || ! grep -q '^check-inputs exit 0$' $C; then
  print "== not launched: check-machine or check-inputs failed" >> $C
  print -u2 "check-machine or check-inputs failed: see $label-conditions.txt"
  exit 1
fi
print "== command" >> $C
print "caffeinate -dims .venv/bin/python -I -B scripts/pad_anet_run.py ${args[*]}" >> $C
caffeinate -dims $py -I -B scripts/pad_anet_run.py "${args[@]}" > $log_dir/$label.console.log 2>&1
rc=$?
{
  print "$cmd exit $rc"
  print "== conditions after"
  conditions
} >> $C
exit $rc
