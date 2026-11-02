import torch,sys
b=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
print('top',[k for k in b.keys()])
for k in b:
    if k in('config','mhr_inference','checkpoint','frame_meta','frames'): continue
    v=b[k]
    if isinstance(v,dict):
        print(k,{q:(tuple(x.shape) if hasattr(x,'shape') else str(x)[:30]) for q,x in list(v.items())[:30]})
    elif hasattr(v,'shape'): print(k,tuple(v.shape))
    else: print(k,str(v)[:80])
print(b['frame_meta'][0], b['frames'][:2])
