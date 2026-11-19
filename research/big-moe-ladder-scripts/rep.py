import json,collections,numpy as np
from common import *
R=json.load(open('results.json')); GP=json.load(open('glyd_pack.json'))
mans={k:json.load(open(f'manifest_{k}.json')) for k in('glm','nemo')}
pk={(r['model'],r['ix']):r for r in GP}
out=[]
def P(s=''): out.append(s)
LV=[6,5,4,3,2]
ALL={}
for k in('glm','nemo'):
    id,sha,T,G=groups(k)
    mix={e['ix']:e for e in mans[k]['samples']}
    # results for this model: ix collide across models -> split by order
    ALL[k]=(id,sha,T,G)
# results.json order: nemo then glm
nn=len(mans['nemo']['samples']); RS={'nemo':R[:nn],'glm':R[nn:]}
def agg(k):
    id,sha,T,G=ALL[k]; res=RS[k]; man=mans[k]['samples']
    sk=collections.defaultdict(list)
    for r,e in zip(res,man):
        assert r['ix']==e['ix']; r['_e']=e; sk[(r['group'],r['kind'])].append(r)
    K={}  # (group,kind) -> dict
    for g,l in G.items():
        for t in l:
            if t['name'].endswith('scale_inv'): 
                kk=(g,kind(t['name'][:-len('_scale_inv')]+'')) 
                K.setdefault((g,kind(t['name'])[:-len('_scale_inv')]),dict(numel=0,bytes=0,dtype='F8_E4M3'))['bytes']+=t['nbytes']; continue
            d=K.setdefault((g,kind(t['name'])),dict(numel=0,bytes=0,dtype=t['dtype']))
            d['numel']+=t['numel']; d['bytes']+=t['nbytes']
    for (g,kd),d in K.items():
        s=sk.get((g,kd),[])
        d['n_samples']=len(s)
        if not s: continue
        m=lambda f:float(np.mean([x[f] for x in s if f in x])) if any(f in x for x in s) else None
        if d['dtype']=='BF16':
            for f in('H16','Hexp','Hsm'): d[f]=m(f)
            gp=[pk[(k,x['ix'])] for x in s if (k,x['ix']) in pk]
            if gp: d['glyd']=sum(p['packed'] for p in gp)/sum(p['raw'] for p in gp)
        elif m('H8') is not None: d['H8']=m('H8')
        if any('rel6' in x for x in s):
            d['rms']=m('rms')
            for b in LV: d[f'rel{b}']=m(f'rel{b}'); d[f'Hc{b}']=m(f'Hc{b}')
    return K
KK={k:agg(k) for k in('glm','nemo')}
def size(d,level):
    """bytes of this kind at a level: orig|ent|exp|glyd|N|eN"""
    if level=='orig': return d['bytes']
    if level=='ent':
        if 'H16' in d: return d['numel']*d['H16']/8
        if 'H8' in d: return d['numel']*d['H8']/8+(d['bytes']-d['numel'])
        return d['bytes']
    if level=='glyd':
        return d['bytes']*d['glyd'] if 'glyd' in d else d['bytes']
    if isinstance(level,int):
        return d['numel']*(level+0.5)/8 if f'rel{level}' in d else d['bytes']
    if level[0]=='e':
        b=int(level[1:]); return d['numel']*d[f'Hc{b}']/8 if f'rel{b}' in d else d['bytes']
def gsize(k,g,level):
    return sum(size(d,level) for (gg,kd),d in KK[k].items() if gg==g)
def gw(k,g,f,num='numel'):
    xs=[(d['numel'],d[f],d) for (gg,kd),d in KK[k].items() if gg==g and f in d]
    if not xs: return None
    return sum(n*v for n,v,_ in xs)/sum(n for n,_,_ in xs)
def grel(k,g,b):
    xs=[(d['numel'],d[f'rel{b}'],d['rms']) for (gg,kd),d in KK[k].items() if gg==g and f'rel{b}' in d]
    if not xs: return None
    return float(np.sqrt(sum(n*r*r*s*s for n,r,s in xs)/sum(n*s*s for n,r,s in xs)))
def gparams(k,g): return sum(d['numel'] for (gg,kd),d in KK[k].items() if gg==g)
def gcov(k,g):
    t=sum(d['numel'] for (gg,kd),d in KK[k].items() if gg==g); c=sum(d['numel'] for (gg,kd),d in KK[k].items() if gg==g and d['n_samples']);return c/t if t else 0
ORDER={'glm':['routed_experts','shared_expert','attn_mla_fp8','attn_mla_bf16','attn_indexer','attn_kda','dense_mlp','embed_head','mtp_other','vision','router','other'],
       'nemo':['routed_experts','shared_expert','mamba','attn','latent_proj','embed_head','mtp_other','router','other']}
GB=lambda x:f'{x/1e9:.2f}'
json.dump({k:{'|'.join(a):b for a,b in v.items()} for k,v in KK.items()},open('kinds.json','w'),indent=1)
for k,title in(('glm','GLM-5.3-Flash'),('nemo','Nemotron-3-Super-120B-A12B-BF16')):
    id,sha,T,G=ALL[k]; man=mans[k]
    P(f'## {title}')
    P(f'`{id}` @ `{sha}`. Sampled {len(man["samples"])} tensors / windows, {man["total_bytes"]/1e9:.2f} GB read via HTTP Range (manifest: `big-moe-ladder-manifest-{k}.json`).')
    P()
    P('### Groups (read from headers; params exclude FP8 scale tensors)')
    P('| group | tensors | params (B) | stored GB | dtype | samples | % params sampled |'); P('|---|---|---|---|---|---|---|')
    tp=tb=0
    for g in ORDER[k]:
        l=[t for t in G.get(g,[])]; 
        if not l: continue
        w=[t for t in l if not t['name'].endswith('scale_inv')]
        ns=sum(d['n_samples'] for (gg,kd),d in KK[k].items() if gg==g)
        pp=sum(t['numel'] for t in w); bb=sum(t['nbytes'] for t in l); tp+=pp; tb+=bb
        P(f'| {g} | {len(w)} | {pp/1e9:.3f} | {GB(bb)} | {"/".join(sorted({t["dtype"] for t in w}))} | {ns} | {100*gcov(k,g):.0f}% |')
    P(f'| **total** | | **{tp/1e9:.2f}** | **{GB(tb)}** | | | |'); P()
    P('### Lossless bounds (measured on samples; order-0 entropy per sampled tensor, param-weighted over kinds)')
    P('BF16 groups: H16 = entropy of the 16-bit word; H(exp) = of the 8-bit exponent; H(s+m) = of sign+7-bit mantissa; "exp-coded" = H(exp)+8 (exponent entropy-coded, sign+mantissa raw); Glyd real = bytes of `pack_tiered` (penguin) / raw, measured on the same sampled tensors.')
    P('| group | H16 | H(exp) | H(s+m) | saving vs 16b (H16) | exp-coded bits | Glyd penguin real (bits/w) | Glyd saving |'); P('|---|---|---|---|---|---|---|---|')
    for g in ORDER[k]:
        h=gw(k,g,'H16')
        if h is None: continue
        he=gw(k,g,'Hexp'); hs=gw(k,g,'Hsm'); gl=gw(k,g,'glyd')
        P(f'| {g} | {h:.3f} | {he:.3f} | {hs:.3f} | {100*(1-h/16):.1f}% | {he+8:.3f} | {16*gl:.3f} | {100*(1-gl):.1f}% |' if gl else f'| {g} | {h:.3f} | {he:.3f} | {hs:.3f} | {100*(1-h/16):.1f}% | {he+8:.3f} | n/a | n/a |')
    P()
    if k=='glm':
        P('FP8 groups: H8 = entropy of the 8-bit E4M3 codes (bits/weight); scale overhead = 128x128 block F32 scale = 0.002 bits/weight.')
        P('| group | H8 | saving vs 8b | scale overhead bits/w |'); P('|---|---|---|---|')
        for g in ORDER[k]:
            h=gw(k,g,'H8')
            if h is None: continue
            sb=sum(d['bytes']-d['numel'] for (gg,kd),d in KK[k].items() if gg==g and d['dtype']=='F8_E4M3'); 
            P(f'| {g} | {h:.3f} | {100*(1-h/8):.1f}% | {8*sb/gparams(k,g):.4f} |')
        P()
    P('### Lossy: block-32 symmetric IntN, bf16 scale chosen to minimise MSE (measured on samples; error vs the stored original, GLM FP8 dequantised first)')
    P('Cell = relative RMSE ||W-W^||/||W|| / stored bits per weight (N+0.5) / entropy-coded bits per weight (entropy of codes + 0.5 for the scale).')
    P('| group | '+' | '.join(f'N={b}' for b in LV)+' |'); P('|---|'+'---|'*len(LV))
    for g in ORDER[k]:
        if grel(k,g,6) is None: continue
        P(f'| {g} | '+' | '.join(f'{grel(k,g,b):.4f} / {b+0.5:.1f} / {gw(k,g,f"Hc{b}"):.2f}' for b in LV)+' |')
    P()
    # per-kind relative RMSE at 4 for sensitivity ranking
    P('Per-kind detail (vision tower omitted: 124 kinds, all near rel4 0.092; N=4 rel RMSE, N=3, H16 or H8, samples):'); P('| group | kind | params (M) | rel4 | rel3 | H16/H8 | n |'); P('|---|---|---|---|---|---|---|')
    for (g,kd),d in sorted(KK[k].items(),key=lambda x:(ORDER[k].index(x[0][0]),x[0][1])):
        if 'rel4' in d and g!='vision': P(f'| {g} | {kd} | {d["numel"]/1e6:.1f} | {d["rel4"]:.4f} | {d["rel3"]:.4f} | {d.get("H16",d.get("H8")):.3f} | {d["n_samples"]} |')
    P()
    # totals
    gs=[g for g in ORDER[k] if g in G]
    always=[g for g in gs if g not in('routed_experts','router','other')]
    def tot(spec):  # spec: dict group->level, default orig
        return sum(gsize(k,g,spec.get(g,'orig')) for g in gs)
    P('### Total size if every quantisable group used level X (computed: header params x measured bits; router and norms/other stay original)')
    P('| level | GB | vs original |'); P('|---|---|---|')
    o=tot({})
    rows=[('original',{}),('lossless entropy bound (H16/H8)',{g:'ent' for g in gs}),('Glyd penguin (real packer ratio on BF16 groups; FP8 groups at H8)',{g:('glyd' if any('glyd' in d for (gg,kd),d in KK[k].items() if gg==g) else 'ent') for g in gs})]
    for b in LV: rows.append((f'all groups {b}-bit stored ({b+0.5} bpw)',{g:b for g in gs}))
    for b in LV: rows.append((f'all groups {b}-bit entropy-coded',{g:f'e{b}' for g in gs}))
    for n,s in rows: P(f'| {n} | {GB(tot(s))} | {100*tot(s)/o:.1f}% |')
    P()
    P('### Mixed plans (computed). always-on = every group except routed experts, router, norms/other (attention/Mamba, shared expert, dense, latent proj, embeddings+head, MTP non-expert'+(', vision' if k=='glm' else '')+')')
    P('| plan | stored GB | experts entropy-coded GB (always-on stored) | everything entropy-coded GB | experts GB (stored) |'); P('|---|---|---|---|---|')
    plans=[('lossless only (Glyd penguin / H8 on all)',None)]
    for a,e in((6,5),(6,4),(6,3),(6,2),(5,4),(8,4)):
        plans.append((f'always-on {a}-bit, experts {e}-bit',(a,e)))
    for n,pl in plans:
        if pl is None:
            s={g:('glyd' if any('glyd' in d for (gg,kd),d in KK[k].items() if gg==g) else 'ent') for g in gs}; P(f'| {n} | {GB(tot(s))} | | | |'); continue
        a,e=pl
        if a==8:
            continue
        s={g:a for g in always}; s['routed_experts']=e
        s2=dict(s); s2['routed_experts']=f'e{e}'
        s3={g:f'e{a}' for g in always}; s3['routed_experts']=f'e{e}'
        P(f'| {n} | {GB(tot(s))} | {GB(tot(s2))} | {GB(tot(s3))} | {GB(gsize(k,"routed_experts",e))} |')
    P()
    ALL[k]=ALL[k]+(tot,always,gs,o)
open('tables.md','w').write('\n'.join(out))
print('\n'.join(out))
