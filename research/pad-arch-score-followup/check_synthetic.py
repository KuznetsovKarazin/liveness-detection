"""Exercise the analysis on invented scores only. No NUAA files are used."""
from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import run
import sys
import json
from fractions import Fraction
import numpy as np
from score_analysis import NETWORKS, STATES, SEEDS, score_metrics, write, read, original_dev_threshold

def main():
    y=np.array([0,0,1,1])
    for scores,auc,eer in [([0,0,1,1],1,0),([1,1,0,0],0,1),([.5]*4,.5,.5)]:
        a,e,_=score_metrics(y,np.array(scores))
        assert a==auc and e==eer
    assert original_dev_threshold(y, np.array([.1,.4,.3,.9])) == .3
    assert original_dev_threshold(y, np.array([.5,.5,.5,.5])) == .5
    assert original_dev_threshold(y, np.array([0.,0.,0.,0.])) == 0.
    rng=np.random.default_rng(73)
    with TemporaryDirectory() as temp:
        root=Path(temp)
        for n in ('test','dev4','dev12'): (root/n).mkdir()
        references={}
        for p in (4,12):
            for f in range(1,5):
                for seed in SEEDS:
                    for net in NETWORKS:
                        base=f'{net}_fold{f}_seed{seed}'
                        per_test=[]
                        for state in STATES:
                            def make(role,n):
                                rows=[]
                                for i in range(n):
                                    label=i%2
                                    if role=='test':
                                        person=f*4+((i//2)%4)
                                    else:
                                        person=(f%4+1)*4+((i//2)%4)
                                    score=float(rng.uniform()*0.85+label*0.15)
                                    rows.append({'network':net,'fold':f,'train_seed':seed,'bn_state':state,'role':role,
                                                 'sample_id':f'{role}-{f}-{i}','subject_id':str(person),'label':label,
                                                 'score_attack':score,'prediction_05':int(score>.5),'checkpoint_sha256':'1'*64})
                                return rows
                            test,dev=make('test',24),make('dev',400)
                            per_test.extend(test)
                            write(root/f'dev{p}'/f'{base}_dev_{state}.csv',dev)
                            eer_cut=original_dev_threshold(
                                np.array([x['label'] for x in dev]),
                                np.array([x['score_attack'] for x in dev]))

                            for mode,t in [('fixed_0.5',.5),('dev_eer',eer_cut)]:
                                k=(p,f,seed,state,mode)
                                r=references.setdefault(k,dict(patience=p,fold=f,seed=seed,bn_state=state,threshold=mode,n_frames=len(test)))
                                pre='livenessnet' if net=='LivenessNet' else 'resnet18'
                                ac=Fraction(sum(x['label']==1 and x['score_attack']<=t for x in test),12)
                                bc=Fraction(sum(x['label']==0 and x['score_attack']>t for x in test),12)
                                for mk,v in [('threshold',t),('apcer',ac),('bpcer',bc),('acer',(ac+bc)/2)]:
                                    r[pre+'_'+mk]=str(v)
                        write(root/'test'/f'p{p}_{base}_test_scores.csv',per_test)
        write(root/'units.csv',list(references.values()))
        run([sys.executable,str(Path(__file__).with_name('score_analysis.py')),
             '--test',str(root/'test'),'--dev4',str(root/'dev4'),'--dev12',str(root/'dev12'),
             '--units',str(root/'units.csv'),'--out',str(root/'out')],check=True)
        stat=json.loads((root/'out/sensitivity.json').read_text())
        assert stat['pooled']['attempted']==10000 and stat['pooled']['invalid']>0
        assert stat['within_fold']['valid']==10000 and stat['valid_identifier_deletions']==16
        assert len(read(root/'out/score_metrics.csv'))==96 and len(read(root/'out/paired_summary.csv'))==24
        assert len(read(root/'out/det_grid.csv'))==96*12
        assert (root/'out/det_curves.pdf').stat().st_size>1000
        print('Synthetic end-to-end check passed.')

if __name__=='__main__':
    main()
