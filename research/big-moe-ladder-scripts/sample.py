import json,sys,os,random,hashlib,re,requests,time
from concurrent.futures import ThreadPoolExecutor
from common import *
WIN=64*1024*1024
def pick(l,k):
    l=sorted(l); 
    if len(l)<=k: return l
    return sorted({l[0],l[len(l)//2],l[-1]})[:k] if k==3 else [l[0],l[-1]]
def plan(k):
    id,sha,T,G=groups(k); P=[]
    rng=random.Random(7)
    for g,l in G.items():
        if g=='other': continue
        W=[t for t in l if not t['name'].endswith('scale_inv')]
        if g=='routed_experts':
            exp_re=re.compile(r'experts\.(\d+)\.(\w+)\.weight$')
            byk={}
            for t in W:
                m=exp_re.search(t['name']); byk.setdefault(m.group(2),[]).append((layer(t['name']),int(m.group(1)),t['name']))
            for kk,lst in byk.items():
                Ls=sorted({x[0] for x in lst}); Ls=pick(Ls,3)
                for L in Ls:
                    ids=sorted(e for (l2,e,n) in lst if l2==L); 
                    for e in rng.sample(ids,4):
                        n=[n for (l2,e2,n) in lst if l2==L and e2==e][0]; P.append((g,n,[0.5]))
            continue
        byk={}
        for t in W: byk.setdefault(kind(t['name']),[]).append(t)
        for kk,lst in byk.items():
            if g=='vision' and len(lst[0]['shape'])!=2: continue
            if g in('vision',) and lst[0]['numel']<100000: continue
            Ls=sorted(lst,key=lambda t:(layer(t['name']) or 0,t['name']))
            ch=[Ls[0],Ls[len(Ls)//2],Ls[-1]] if len(Ls)>=3 else Ls
            seen=set(); 
            for t in ch:
                if t['name'] in seen: continue
                seen.add(t['name'])
                fr=[0.25,0.75] if g=='embed_head' else [0.5]
                P.append((g,t['name'],fr))
    return id,sha,T,P
S3='/private/tmp/claude-501/-Users-suryakoritala/d09db505-88bb-4450-a82f-95fd523dc0ee/scratchpad'
def rd(id,sha,shard,a,b):
    u=f'https://huggingface.co/{id}/resolve/{sha}/{shard}'
    for i in range(6):
        try:
            r=requests.get(u,headers={'Range':f'bytes={a}-{b-1}'},timeout=120)
            if r.status_code==206 and len(r.content)==b-a: return r.content
        except Exception as e: pass
        time.sleep(2*(i+1))
    raise RuntimeError(f'fail {shard} {a} {b}')
def run(k):
    id,sha,T,P=plan(k); os.makedirs(f'{S3}/samples/{k}',exist_ok=True)
    jobs=[]
    for (g,n,fr) in P:
        t=T[n]; rowb=t['nbytes']//t['shape'][0]; rows=t['shape'][0]
        wr=rows if t['nbytes']<=WIN else (WIN//rowb)//128*128
        for f in fr:
            r0=0 if wr==rows else int((rows-wr)*f)//128*128
            jobs.append((g,n,r0,wr))
    est=sum(wr*(T[n]['nbytes']//T[n]['shape'][0]) for g,n,r0,wr in jobs)
    print(k,len(jobs),'samples',est/1e9,'GB',flush=True)
    def one(ix_job):
        ix,(g,n,r0,wr)=ix_job; t=T[n]; rowb=t['nbytes']//t['shape'][0]
        a=t['begin']+r0*rowb; b=a+wr*rowb; data=rd(id,sha,t['shard'],a,b)
        fn=f'{S3}/samples/{k}/{ix}.bin'; open(fn,'wb').write(data)
        e=dict(ix=ix,model=k,group=g,tensor=n,kind=kind(n),layer=layer(n),dtype=t['dtype'],shape=t['shape'],shard=t['shard'],begin=a,end=b,row_start=r0,rows=wr,sha256=hashlib.sha256(data).hexdigest(),bytes=b-a)
        if t['dtype']=='F8_E4M3':
            sn=n[:-len('weight')]+'weight_scale_inv'; st=T[sn]; srow=st['nbytes']//st['shape'][0]
            sa=st['begin']+(r0//128)*srow; sb=sa+((wr+127)//128)*srow; sd=rd(id,sha,st['shard'],sa,sb)
            open(fn+'.scale','wb').write(sd)
            e['scale']=dict(tensor=sn,shape=st['shape'],begin=sa,end=sb,sha256=hashlib.sha256(sd).hexdigest(),bytes=sb-sa)
        return e
    with ThreadPoolExecutor(8) as ex: M=list(ex.map(one,enumerate(jobs)))
    tot=sum(e['bytes']+e.get('scale',{}).get('bytes',0) for e in M)
    json.dump(dict(id=id,sha=sha,total_bytes=tot,samples=M),open(f'{S3}/manifest_{k}.json','w'),indent=1)
    print(k,'done',len(M),tot/1e9,'GB')
run(sys.argv[1])
