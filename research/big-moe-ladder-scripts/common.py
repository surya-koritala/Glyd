import json,re,math
DT={'BF16':2,'F8_E4M3':1,'F32':4,'F16':2,'I64':8}
def load(k):
    d=json.load(open(f'/private/tmp/claude-501/-Users-suryakoritala/d09db505-88bb-4450-a82f-95fd523dc0ee/scratchpad/{k}_hdr.json'))
    T={}
    for f,s in d['shards'].items():
        for n,t in s['tensors'].items():
            a,b=t['data_offsets']; T[n]=dict(name=n,dtype=t['dtype'],shape=t['shape'],shard=f,begin=s['data_start']+a,end=s['data_start']+b,numel=math.prod(t['shape']) if t['shape'] else 1,nbytes=b-a)
    return d['id'],d['sha'],T
def layer(n):
    m=re.search(r'layers\.(\d+)\.',n); return int(m.group(1)) if m else None
def kind(n): return re.sub(r'\d+','N',re.sub(r'^.*?layers\.\d+\.','',n)) if layer(n) is not None else n
def classify_glm(n,t):
    L=layer(n)
    if n.endswith('weight_scale_inv'): base=n[:-len('_scale_inv')]
    else: base=n
    if '.visual.' in n: return 'vision'
    if n in('lm_head.weight','model.language_model.embed_tokens.weight'): return 'embed_head'
    if L is None: return 'other'
    if L==45:
        return 'routed_experts' if '.mlp.experts.' in n else 'mtp_other'
    if '.mlp.experts.' in n: return 'routed_experts'
    if '.mlp.shared_experts.' in n: return 'shared_expert'
    if re.search(r'\.mlp\.(gate|up|down)_proj\.',n): return 'dense_mlp'
    if n.endswith('mlp.gate.weight'): return 'router'
    if '.self_attn.indexer.' in n:
        return 'attn_indexer' if t['shape'].__len__()==2 and t['numel']>=100000 else 'other'
    if '.self_attn.' in n:
        if re.search(r'\.(kv_a_proj_with_mqa|q_a_proj|q_b_proj)\.|\.o_proj\.',n) and (t['dtype']=='F8_E4M3' or n.endswith('scale_inv')): return 'attn_mla_fp8'
        if 'kv_b_proj' in n: return 'attn_mla_bf16'
        if len(t['shape'])==2 and t['numel']>=100000 and t['dtype']=='BF16': return 'attn_kda'
    return 'other'
def classify_nemo(n,t):
    L=layer(n)
    if n in('lm_head.weight','backbone.embeddings.weight'): return 'embed_head'
    if '.mixer.experts.' in n: return 'routed_experts'
    if '.shared_experts.' in n: return 'shared_expert'
    if 'latent_proj' in n: return 'latent_proj'
    if n.endswith('mixer.gate.weight'): return 'router'
    if re.search(r'mixer\.(q|k|v|o)_proj',n) and n.startswith('backbone'): return 'attn'
    if re.search(r'mixer\.(in|out)_proj',n): return 'mamba'
    if n.startswith('mtp.'):
        if re.search(r'mixer\.(q|k|v|o)_proj',n) or 'eh_proj' in n: return 'mtp_other'
        return 'other'
    return 'other'
def groups(k):
    id,sha,T=load(k); cl=classify_glm if k=='glm' else classify_nemo
    G={}
    for n,t in T.items(): t['group']=cl(n,t); G.setdefault(t['group'],[]).append(t)
    return id,sha,T,G
if __name__=='__main__':
    import sys
    for k in('glm','nemo'):
        id,sha,T,G=groups(k); tot=sum(t['nbytes'] for t in T.values())
        print(k,id,sha,'tensors',len(T),'GB',tot/1e9,'params(all non-scale)',sum(t['numel'] for t in T.values() if not t['name'].endswith('scale_inv'))/1e9)
        for g,l in sorted(G.items(),key=lambda x:-sum(t['nbytes'] for t in x[1])):
            w=[t for t in l if not t['name'].endswith('scale_inv')]
            print(f'  {g:16s} n={len(l):6d} params={sum(t["numel"] for t in w)/1e9:8.3f}B bytes={sum(t["nbytes"] for t in l)/1e9:8.3f}GB dt={set(t["dtype"] for t in w)}')
