# Contact sheet of the first scored frame per episode.
import cv2,numpy as np,pandas as pd
s=pd.read_parquet('official/v2d_submission_kit/data/track_1_sample_submission.parquet',columns=['row_id'])
p=s.row_id.str.split('_',expand=True); p.columns=['t','ep','fr','arr','pt']
a=p[p.arr=='0'].drop_duplicates(['ep','fr']); g=a.groupby('ep').fr.apply(lambda x:[int(v) for v in x])
tiles=[]
for ep,fr in g.items():
    e=int(ep); cap=cv2.VideoCapture(f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{e:06d}.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES,fr[len(fr)//2]); ok,im=cap.read(); cap.release()
    if not ok: im=np.zeros((1152,1536,3),np.uint8)
    im=cv2.resize(im,(384,288)); cv2.putText(im,f'ep{e} f{fr[len(fr)//2]} n={len(fr)}',(6,22),0,.6,(0,255,255),2); tiles.append(im)
while len(tiles)%6: tiles.append(np.zeros_like(tiles[0]))
rows=[np.hstack(tiles[i:i+6]) for i in range(0,len(tiles),6)]
cv2.imwrite('work/analysis/sheet.jpg',np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,80])
