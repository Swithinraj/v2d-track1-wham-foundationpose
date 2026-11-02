import torch,numpy as np,sys
for name in sys.argv[1:]:
    b=torch.load(name,map_location='cpu',weights_only=False)
    print(name.split('/')[-1],type(b).__name__,list(b.keys())[:40])
    def show(d,prefix=''):
        for k,v in d.items():
            if isinstance(v,dict) and len(prefix)<8: print(prefix+k+':'); show(v,prefix+'  ')
            elif hasattr(v,'shape'): print(prefix+k,tuple(v.shape),str(v.dtype)[:12])
            elif isinstance(v,(list,tuple)): print(prefix+k,'list',len(v),str(v[0])[:30] if len(v) else '')
            else: print(prefix+k,str(v)[:60])
    show({k:v for k,v in b.items() if k not in('frames',)})
    if 'frames' in b: print('frames',len(b['frames']),b['frames'][:2],b['frames'][-1:])
