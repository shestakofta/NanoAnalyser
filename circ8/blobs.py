# -*- coding: utf-8 -*-
import numpy as np
from skimage.feature import blob_log
from skimage.transform import rescale
def blobs(D, M, s, s_work=0.5, rmin_nm=3, rmax_nm=40, thr=0.05, overlap=0.5, num_sigma=15):
    s_eff=max(s,s_work)                     # снимки с крупным пикселем не увеличиваем (иначе очень долго)
    f=s/s_eff
    Dw=rescale(np.clip(D,0,None)*M, f, anti_aliasing=True, preserve_range=True)
    Dw=Dw/ (np.percentile(Dw[Dw>0],99) if (Dw>0).any() else 1)
    smin=rmin_nm/s_eff/np.sqrt(2); smax=rmax_nm/s_eff/np.sqrt(2)
    B=blob_log(Dw, min_sigma=smin, max_sigma=smax, num_sigma=num_sigma, threshold=thr, overlap=overlap, log_scale=True)
    out=[]
    for y,x,sg in B:
        Y,X=y/f,x/f
        if 0<=int(Y)<M.shape[0] and 0<=int(X)<M.shape[1] and M[int(Y),int(X)]:
            out.append((Y,X,sg*np.sqrt(2)/f))
    return out
