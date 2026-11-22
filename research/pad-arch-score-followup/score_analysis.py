"""Secondary analysis of the native NUAA score files. Export aggregates only."""
import argparse
import csv
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

NETWORKS = ('LivenessNet','ResNet-18')
STATES = ('ordinary','re-estimated')
GRID = (0.001,0.005,0.01,0.02,0.05,0.10,0.20,0.40,0.60,0.80,0.95,0.99)
SEEDS = (42,43,44)


def read(path):
    with path.open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def profile(y, scores):
    """Ascending thresholds, strict attack score > threshold. Preserve tied scores."""
    if not np.isfinite(scores).all() or not np.isin(y,[0,1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Each score set must be finite and contain both classes')
    if np.min(scores)<0 or np.max(scores)>1:
        raise ValueError('Expected attack probabilities in [0,1]')
    attacks=np.sort(scores[y==1])
    genuine=np.sort(scores[y==0])
    thr=np.r_[np.nextafter(np.min(scores),-np.inf),np.unique(scores)]
    a=np.searchsorted(attacks,thr,side='right')/len(attacks)
    b=1.0-np.searchsorted(genuine,thr,side='right')/len(genuine)
    return thr,a,b


def original_dev_threshold(labels, scores):
    """Reproduce the archived highest-threshold tie rule using integer counts."""
    attacks = np.sort(scores[labels == 1])
    genuine = np.sort(scores[labels == 0])
    candidates = np.unique(scores)
    lowest = np.nextafter(candidates[0], -np.inf)
    if lowest >= 0:
        candidates = np.r_[lowest, candidates]
    accepted_attacks = np.searchsorted(attacks, candidates, side='right')
    rejected_genuine = len(genuine) - np.searchsorted(genuine, candidates, side='right')
    imbalance = np.abs(rejected_genuine * len(attacks) - accepted_attacks * len(genuine))
    return float(candidates[np.flatnonzero(imbalance == imbalance.min())[-1]])


def score_metrics(y,s):
    _,a,b=profile(y,s)
    w=a-b
    j=int(np.searchsorted(w,0,side='left'))
    if w[j] == 0:
        eer=float(a[j])
    else:
        k=j-1
        frac=-w[k]/(w[j]-w[k])
        eer=float(a[k]+frac*(a[j]-a[k]))
    n1=int(y.sum())
    auc=(rankdata(s,method='average')[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
    return float(auc),eer,[float(b[a<=q].min()) for q in GRID]


def errors(y,s,t):
    return np.asarray([np.mean(s[y==1]<=t),np.mean(s[y==0]>t)])


def prepare(rows,net,fold,seed,state,role):
    if len(rows)==0:
        raise ValueError('Missing score rows')
    order=sorted(rows,key=lambda r:r['sample_id'])
    if any((r['network'],r['fold'],r['train_seed'],r['bn_state'],r['role']) !=
           (net,str(fold),str(seed),state,role) for r in order):
        raise ValueError('Inconsistent checkpoint metadata')
    ids=tuple(r['sample_id'] for r in order)
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate sample within a checkpoint state')
    persons=tuple(r['subject_id'] for r in order)
    labels=np.array([int(r['label']) for r in order])
    scores=np.array([float(r['score_attack']) for r in order])
    if not np.array_equal(np.array([int(r['prediction_05']) for r in order]),scores>0.5):
        raise ValueError('Stored fixed-threshold decisions differ')
    return ids,persons,labels,scores


def bootstrap(counts, numer, folds, keep):
    """The numerator is the signed four-way difference of misclassification counts."""
    def evaluate(weights, selected):
        shape=(len(weights),4,3,2)
        d=(weights@counts.reshape(len(folds),-1)).reshape(shape)[:,selected]
        n=(weights@numer.reshape(len(folds),-1)).reshape(shape)[:,selected]
        valid=(d>0).all(axis=(1,2))
        return (100*(n[valid]/d[valid]).mean(axis=(1,2))),int((~valid).sum())
    rng=np.random.default_rng(20261122)
    n=len(folds)
    all_units=np.ones((4,3),dtype=bool)
    eye=np.eye(n,dtype=int)
    full=evaluate(np.ones((1,n),dtype=int),all_units)[0][0]
    jack,_=evaluate(1-eye,all_units)
    def summary(w,selected,method,with_bca):
        draws,invalid=evaluate(w,selected)
        if len(draws)<2:
            raise ValueError('Insufficient valid draws')
        theta=evaluate(np.ones((1,n),dtype=int),selected)[0][0]
        lo,hi=np.quantile(draws,[.025,.975])
        ans={'method':method,'attempted':len(w),'valid':len(draws),'invalid':invalid,'estimate_pp':float(theta),
             'percentile_95_pp':[float(lo),float(hi)]}
        if with_bca:
            jcenter=(jack.mean()-jack) if len(jack)==n else np.array([])
            den=6*np.sum(jcenter**2)**1.5
            prop=(np.sum(draws<full)+.5*np.sum(draws==full))/len(draws)
            if den and 0<prop<1:
                acc=np.sum(jcenter**3)/den
                z=norm.ppf(prop)
                tails=z+norm.ppf([.025,.975])
                adj=norm.cdf(z+tails/(1-acc*tails))
                if np.all(np.isfinite(adj)) and 0<adj[0]<adj[1]<1:
                    ans['BCa_95_pp']=[float(v) for v in np.quantile(draws,adj)]
            ans.setdefault('BCa_95_pp',None)
        return ans
    pooled=rng.multinomial(n,np.full(n,1/n),size=10000)
    strata=np.zeros_like(pooled)
    for fold in range(1,5):
        mask=np.asarray(folds)==fold
        size=int(mask.sum())
        strata[:,mask]=rng.multinomial(size,np.full(size,1/size),size=10000)
    return {'pooled':summary(pooled,all_units,'pooled identifiers',True),
            'within_fold':summary(strata,all_units,'identifiers resampled within original fold',False),
            'omit_influential_unit':summary(pooled,keep,'pooled, omit fold 3 seed 44',False),
            'valid_identifier_deletions':len(jack),
            'identifier_deletion_range_pp':[float(jack.min()),float(jack.max())] if len(jack) else None}


def draw_det(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(8,6),sharex=True,sharey=True)
    targets=np.array(GRID)
    transform=lambda x:norm.ppf(np.clip(x,.0001,.9999))
    ticks=np.array([.001,.01,.05,.20,.50,.80,.99])
    for i,net in enumerate(NETWORKS):
        for j,p in enumerate((4,12)):
            ax=axes[i,j]
            for state,color in zip(STATES,('#204f71','#b34929')):
                values=[]
                for f in range(1,5):
                    for seed in SEEDS:
                        rr=[r for r in rows if (r['network'],r['patience'],r['bn_state'],r['fold'],r['seed'])==(net,p,state,f,seed)]
                        v=[r['minimum_attainable_test_bpcer'] for r in rr]
                        values.append(v)
                        ax.plot(transform(targets),transform(v),color=color,alpha=.14,lw=.5)
                ax.plot(transform(targets),transform(np.mean(values,axis=0)),color=color,lw=1.5,label=state)
            ax.set_title(f'{net} / patience {p}')
            ax.set_xticks(transform(ticks),[f'{t*100:g}' for t in ticks])
            ax.set_yticks(transform(ticks),[f'{t*100:g}' for t in ticks])
            ax.grid(alpha=.22)
            if i==1: ax.set_xlabel('APCER ceiling (%)')
            if j==0: ax.set_ylabel('Minimum attainable BPCER (%)')
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path,metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('test','dev4','dev12','units','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    unit_rows=read(args.units)
    ref={(int(r['patience']),int(r['fold']),int(r['seed']),r['bn_state'],r['threshold']):r for r in unit_rows}
    if len(ref)!=96:
        raise ValueError('Expected the 96 verified public comparison rows')
    metric_rows,det,fold_ids,dev_ids=[],[],{},{}
    private_counts={}
    checkpoint_hashes = {}
    for p,devroot in ((4,args.dev4),(12,args.dev12)):
        for f in range(1,5):
            for seed in SEEDS:
                for net_index,net in enumerate(NETWORKS):
                    name=f'{net}_fold{f}_seed{seed}'
                    test_rows=read(args.test/f'p{p}_{name}_test_scores.csv')
                    for si,state in enumerate(STATES):
                        test=[r for r in test_rows if r['bn_state']==state]
                        dev=read(devroot/f'{name}_dev_{state}.csv')
                        ids,sub,y,s=prepare(test,net,f,seed,state,'test')
                        did,dsub,dy,ds=prepare(dev,net,f,seed,state,'dev')
                        pairing=(ids,sub,tuple(y))
                        dpair=(did,dsub,tuple(dy))
                        if fold_ids.setdefault(f,pairing)!=pairing or dev_ids.setdefault(f,dpair)!=dpair:
                            raise ValueError('Frame/label/identifier pairing differs across conditions')
                        hashes={r['checkpoint_sha256'] for r in test + dev}
                        if len(hashes) != 1:
                            raise ValueError('Test and development checkpoint hashes differ')
                        checksum = hashes.pop()
                        if not re.fullmatch(r'[0-9a-f]{64}', checksum):
                            raise ValueError('Invalid checkpoint SHA-256')
                        if checkpoint_hashes.setdefault((p, f, seed, net), checksum) != checksum:
                            raise ValueError('The two BN states point to different parent checkpoints')
                        if set(ids)&set(did) or set(sub)&set(dsub) or len(dy)!=400:
                            raise ValueError('Unexpected fit of the dev/test partitions')
                        pre='livenessnet' if net_index==0 else 'resnet18'
                        for rule in ('fixed_0.5','dev_eer'):
                            published=ref[p,f,seed,state,rule]
                            cut=float(published[pre+'_threshold'])
                            if rule == 'dev_eer' and original_dev_threshold(dy, ds) != cut:
                                raise ValueError('The original threshold does not reproduce on development scores')
                            observed=errors(y,s,cut)
                            expected=[float(Fraction(published[pre+'_'+m])) for m in ('apcer','bpcer')]
                            if len(y)!=int(published['n_frames']) or not np.allclose(observed,expected,rtol=0,atol=1e-12):
                                raise ValueError('Public test errors do not reproduce')
                        auc,eer,grid=score_metrics(y,s)
                        cuts,da,db=profile(dy,ds)
                        row={'patience':p,'fold':f,'seed':seed,'network':net,'bn_state':state,'test_auroc':auc,'test_eer':eer}
                        for target in (.05,.10):
                            idx=np.where(da<=target)[0][-1]
                            t=float(cuts[idx])
                            rates=errors(y,s,t)
                            col=str(int(100*target))
                            for key,val in [('threshold',t),('dev_apcer',da[idx]),('dev_bpcer',db[idx]),
                                            ('test_apcer',rates[0]),('test_bpcer',rates[1])]:
                                row[col+'pct_'+key]=float(val)
                        metric_rows.append(row)
                        for q,b in zip(GRID,grid):
                            det.append({**{k:row[k] for k in ('patience','fold','seed','network','bn_state')},
                                        'test_apcer_ceiling':q,'minimum_attainable_test_bpcer':b})
                        if p==4:
                            who=np.asarray(sub)
                            for person in set(who):
                                mask=who==person
                                d=np.array([(mask&(y==1)).sum(),(mask&(y==0)).sum()])
                                bad=np.array([(mask&(y==1)&(s<=.5)).sum(),(mask&(y==0)&(s>.5)).sum()])
                                key=(person,f,seed)
                                if key not in private_counts:
                                    private_counts[key]=[d,np.zeros((2,2),dtype=int)]
                                assert np.array_equal(private_counts[key][0],d)
                                private_counts[key][1][net_index]+=(1 if si==1 else -1)*bad
    people=sorted({key[0] for key in private_counts})
    if len(people)!=16:
        raise ValueError('The current study uses 16 manifest IDs. Analyze a corrected grouping under a revised protocol.')
    folds=[]
    counts=np.zeros((len(people),4,3,2),dtype=int)
    model_numer=np.zeros((len(people),4,3,2,2),dtype=int)
    for i,person in enumerate(people):
        here={f for g,f,s in private_counts if g==person}
        if len(here)!=1:
            raise ValueError('An identifier spans folds')
        f=here.pop()
        folds.append(f)
        for j,seed in enumerate(SEEDS):
            counts[i,f-1,j],model_numer[i,f-1,j]=private_counts[person,f,seed]
    keep=np.ones((4,3),dtype=bool)
    keep[2,2]=False
    comparison=model_numer[:,:,:,1,:]-model_numer[:,:,:,0,:]
    stat=bootstrap(counts,comparison,folds,keep)
    stat['model_changes']={name:bootstrap(counts,model_numer[:,:,:,i,:],folds,keep)['pooled']
                           for i,name in enumerate(NETWORKS)}
    def published_gap(state):
        return np.mean([float(Fraction(r['resnet18_acer'])-Fraction(r['livenessnet_acer']))
                        for (p,f,s,b,t),r in ref.items() if (p,b,t)==(4,state,'fixed_0.5')])
    expected=100*(published_gap('re-estimated')-published_gap('ordinary'))
    if not math.isclose(stat['pooled']['estimate_pp'],expected,abs_tol=1e-8):
        raise ValueError('Primary paired change does not reproduce the public result')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Choose a fresh output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    write(args.out/'score_metrics.csv',metric_rows)
    write(args.out/'det_grid.csv',det)
    summary=[]
    for patience in (4,12):
        for network in NETWORKS:
            rows=sorted((r for r in metric_rows if r['patience']==patience and r['network']==network),
                        key=lambda r:(r['fold'],r['seed']))
            for metric in ['test_auroc','test_eer','5pct_test_apcer','5pct_test_bpcer','10pct_test_apcer','10pct_test_bpcer']:
                o=np.array([r[metric] for r in rows if r['bn_state']=='ordinary'])
                a=np.array([r[metric] for r in rows if r['bn_state']=='re-estimated'])
                summary.append({'patience':patience,'network':network,'metric':metric,'paired_units':len(o),
                                'stored_mean':float(o.mean()),'reestimated_mean':float(a.mean()),
                                'mean_change':float((a-o).mean())})
    write(args.out/'paired_summary.csv',summary)
    draw_det(det,args.out/'det_curves.pdf')
    (args.out/'sensitivity.json').write_text(json.dumps(stat,indent=2,allow_nan=False)+'\n')
    print(f'{len(metric_rows)} checkpoint-state summaries. Outputs contain aggregate measurements only.')


if __name__=='__main__':
    main()
