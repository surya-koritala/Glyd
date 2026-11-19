import json,struct,requests,sys
from concurrent.futures import ThreadPoolExecutor
M={'glm':('zai-org/GLM-5.3-Flash','eb9eb208eb0d988989d07a6a12d0fdeb5f52574a'),
   'nemo':('nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16','2dc98e2afe4face0e4ce40972a915c45368bd34a')}
for k,(id,sha) in M.items():
    api=requests.get(f'https://huggingface.co/api/models/{id}?revision={sha}').json()
    shards=sorted(s['rfilename'] for s in api['siblings'] if s['rfilename'].endswith('.safetensors'))
    def h(f):
        u=f'https://huggingface.co/{id}/resolve/{sha}/{f}'
        n=struct.unpack('<Q',requests.get(u,headers={'Range':'bytes=0-7'}).content)[0]
        j=json.loads(requests.get(u,headers={'Range':f'bytes=8-{7+n}'}).content)
        j.pop('__metadata__',None)
        return f,8+n,j
    with ThreadPoolExecutor(8) as ex: r=list(ex.map(h,shards))
    out={f:{'data_start':ds,'tensors':j} for f,ds,j in r}
    json.dump({'id':id,'sha':sha,'shards':out},open(f'{k}_hdr.json','w'))
    print(k,len(shards),sum(len(v['tensors']) for v in out.values()))
