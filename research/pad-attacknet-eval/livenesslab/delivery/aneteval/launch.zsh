#!/bin/zsh
# Lancio della valutazione C5 (check-inputs o evaluate) dalla radice del repository, con le condizioni registrate, come
# delivery/anetrun/launch.zsh per i run. Da una shell zsh con setopt nobgnice:
#
#   setopt nobgnice
#   ANETEVAL_LOG_DIR=<cartella fuori dal repository> zsh delivery/aneteval/launch.zsh check-inputs
#   ANETEVAL_LOG_DIR=<cartella fuori dal repository> ANETEVAL_ATTEMPT=1 nohup zsh delivery/aneteval/launch.zsh evaluate > <log fuori dal repository> 2>&1 &
#
# Scrive in ANETEVAL_LOG_DIR <etichetta>-conditions.txt (sistema, modalità di alimentazione e alimentatore da pmset, nice
# dello script, variabili TF_NUM_ impostate, commit e righe di git status --porcelain, carico, uso della CPU e alimentazione
# all'inizio e alla fine, uscite di check-machine dei run e di check-inputs della valutazione, comando ed esito) e
# <etichetta>.console.log. Se il nice non è 0, se (per evaluate) pmset -g batt non dice «AC Power», o se check-machine o
# check-inputs falliscono, evaluate non parte, con il motivo nel file delle condizioni. Lo script non passa mai
# --second-reading: una seconda lettura del test si lancia solo a mano. Nessun percorso personale nel file delle condizioni.
setopt nobgnice
set -u
cmd=${1:-}
run=${2:-}
log_dir=${ANETEVAL_LOG_DIR:-}
attempt=${ANETEVAL_ATTEMPT:-}
py=.venv/bin/python
if [[ ! -f scripts/pad_anet_eval.py ]]; then print -u2 "run from the root of the repository"; exit 9; fi
if [[ -z $log_dir ]]; then print -u2 "ANETEVAL_LOG_DIR is required"; exit 9; fi
case $cmd in
  check-inputs) label="eval-check-inputs-$(date '+%Y%m%d-%H%M%S')"; args=(check-inputs);;
  evaluate) if [[ -z $attempt ]]; then print -u2 "evaluate needs ANETEVAL_ATTEMPT"; exit 9; fi
            label="evaluate-attempt$attempt"; args=(evaluate --attempt $attempt --i-am-authorized);;
  *) print -u2 "usage: launch.zsh check-inputs | evaluate"; exit 9;;
esac
repo=${PWD:A}
log_dir=${log_dir:A}
if [[ $log_dir == $repo || $log_dir == $repo/* ]]; then print -u2 "ANETEVAL_LOG_DIR must be outside the repository"; exit 9; fi
mkdir -p $log_dir
C=$log_dir/$label-conditions.txt
if [[ -e $C ]]; then print -u2 "$label-conditions.txt already exists"; exit 9; fi
nice_now=$(ps -o nice= -p $$ | tr -d ' ')
if [[ $nice_now != 0 ]]; then
  {
    print "Conditions of $cmd${attempt:+, attempt $attempt}, written by delivery/aneteval/launch.zsh (not by the evaluation)."
    print "== nice of the launching shell"
    print $nice_now
    print "== not launched: the nice value is not 0 (launch from a zsh with setopt nobgnice)"
  } > $C
  print -u2 "the nice value of the launch is $nice_now, not 0: launch from a zsh with setopt nobgnice"
  exit 1
fi
if [[ $cmd == evaluate ]]; then
  power=$(pmset -g batt | head -1)
  if [[ $power != *"'AC Power'"* ]]; then
    {
      print "Conditions of $cmd, attempt $attempt, written by delivery/aneteval/launch.zsh (not by the evaluation)."
      print "== power (pmset -g batt: power source)"
      print -r -- $power
      print "== not launched: evaluate needs the power adapter (pmset -g batt does not say 'AC Power')"
    } > $C
    print -u2 "evaluate needs the power adapter: pmset -g batt does not say 'AC Power'"
    exit 1
  fi
fi
conditions() {
  date '+%Y-%m-%d %H:%M:%S %z'
  uptime
  top -l 1 -n 0 | grep 'CPU usage'
  pmset -g batt | head -1
}
{
  print "Conditions of $cmd${attempt:+, attempt $attempt}, written by delivery/aneteval/launch.zsh (not by the evaluation)."
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
  print "== check-machine (of the runs)"
  $py -I -B scripts/pad_anet_run.py check-machine 2>&1
  print "check-machine exit $?"
  print "== check-inputs (of the evaluation)"
  $py -I -B scripts/pad_anet_eval.py check-inputs 2>&1
  print "check-inputs exit $?"
} > $C
if ! grep -q '^check-machine exit 0$' $C || ! grep -q '^check-inputs exit 0$' $C; then
  print "== not launched: check-machine or check-inputs failed" >> $C
  print -u2 "check-machine or check-inputs failed: see $label-conditions.txt"
  exit 1
fi
print "== command" >> $C
print "caffeinate -dims .venv/bin/python -I -B scripts/pad_anet_eval.py ${args[*]}" >> $C
caffeinate -dims $py -I -B scripts/pad_anet_eval.py "${args[@]}" > $log_dir/$label.console.log 2>&1
rc=$?
{
  print "$cmd exit $rc"
  print "== conditions after"
  conditions
} >> $C
exit $rc
