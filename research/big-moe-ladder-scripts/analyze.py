import json,numpy as np,ml_dtypes,sys
from multiprocessing import Pool
CAP=16_000_000
def H(cnt):
    p=cnt[cnt>0]/cnt.sum(); return float(-(p*np.log2(p)).sum())
bf=ml_dtypes.bfloat16
def fit_scale(w,q):
    amax=np.maximum(np.abs(w).max(-1,keepdims=True),1e-30)
    best=None;bs=None
    for r in np.linspace(0.70,1.0,16,dtype=np.float32):
        s=(amax*r/q).astype(bf).astype(np.float32)
        s=np.where(s==0,np.float32(1e-38),s)  # guard: bf16 scale of an all-zero run
        e=((np.clip(np.round(w/s),-q,q)*s-w)**2).sum(-1,keepdims=True)
        if best is None: best,bs=e,s
        else:
            m=e<best; best=np.where(m,e,best); bs=np.where(m,s,bs)
    return bs
def int_tensor(W,bits,block=32):
    q=(1<<(bits-1))-1; R,C=W.shape; hist=np.zeros(2*q+1); se=0.0
    for r0 in range(0,R,512):
        w=W[r0:r0+512].reshape(-1,C//block,block)
        s=fit_scale(w,q); c=np.clip(np.round(w/s),-q,q)
        se+=float(((c*s-w)**2).sum()); hist+=np.bincount((c+q).astype(np.int64).ravel(),minlength=2*q+1)
    return se,H(hist)+16/block
def one(e):
    k=e['model'];raw=open(f'samples/{k}/{e["ix"]}.bin','rb').read();r=dict(ix=e['ix'],group=e['group'],kind=e['kind'],tensor=e['tensor'],dtype=e['dtype'],n=e['rows']*int(np.prod(e['shape'][1:])))
    if e['dtype']=='BF16':
        u=np.frombuffer(raw,dtype=np.uint16)
        r['H16']=H(np.bincount(u,minlength=65536)); r['Hexp']=H(np.bincount((u>>7)&255,minlength=256)); r['Hsm']=H(np.bincount(((u>>8)&128)|(u&127),minlength=256))
        W=u.view(bf).astype(np.float32)
    elif e['dtype']=='F8_E4M3':
        u=np.frombuffer(raw,dtype=np.uint8); r['H8']=H(np.bincount(u,minlength=256))
        sc=np.frombuffer(open(f'samples/{k}/{e["ix"]}.bin.scale','rb').read(),dtype=np.float32).reshape(-1,e['shape'][1]//128)
        W=u.view(ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(e['rows'],-1)
        W=(W.reshape(e['rows']//128,128,sc.shape[1],128)*sc[:,None,:,None]).reshape(e['rows'],-1)
        r['scale_bits_per_w']=e['scale']['bytes']*8/r['n']
    else: return r
    if len(e['shape'])==2 and e['shape'][1]%32==0:
        W=W.reshape(e['rows'],-1); R=max(128,(CAP//W.shape[1])//128*128) if e['rows']>=128 else e['rows']
        W=np.ascontiguousarray(W[:min(R,e['rows'])]); nrm=float((W.astype(np.float64)**2).sum())
        r['lossy_n']=int(W.size); r['rms']=float(np.sqrt((W.astype(np.float64)**2).mean()))
        for b in(6,5,4,3,2):
            se,h=int_tensor(W,b); r[f'rel{b}']=float(np.sqrt(se/nrm)); r[f'Hc{b}']=h
    return r
if __name__=='__main__':
    E=[e for k in('nemo','glm') for e in json.load(open(f'manifest_{k}.json'))['samples']]
    with Pool(8) as p: R=p.map(one,E,chunksize=1)
    json.dump(R,open('results.json','w')); print(len(R))
