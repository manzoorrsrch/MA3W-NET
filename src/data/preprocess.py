import os, json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

TARGET_SPACING = (1.0,1.0,1.0)
CLIP_Z = 5.0
MIN_SIDE = 128

def sitk_resample_to_spacing(img, out_spacing, is_label):
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    out_size = [
        int(round(orig_size[i]*(orig_spacing[i]/out_spacing[i])))
        for i in range(3)
    ]
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    res = sitk.ResampleImageFilter()
    res.SetInterpolator(interp)
    res.SetOutputSpacing(out_spacing)
    res.SetSize(out_size)
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    res.SetDefaultPixelValue(0)
    return res.Execute(img)

def zscore_clip(vol, mask=None, clip=CLIP_Z):
    if mask is None: mask = (vol!=0)
    if mask.any():
        mean = vol[mask].mean()
        std  = vol[mask].std()+1e-8
    else:
        mean, std = vol.mean(), vol.std()+1e-8
    v = (vol-mean)/std
    return np.clip(v, -clip, clip)

def tight_bbox(mask):
    coords = np.where(mask)
    if len(coords[0])==0:
        Z,Y,X = mask.shape
        return (0,Z,0,Y,0,X)
    z0,y0,x0 = min(coords[0]),min(coords[1]),min(coords[2])
    z1,y1,x1 = max(coords[0])+1,max(coords[1])+1,max(coords[2])+1
    return (z0,z1,y0,y1,x0,x1)

def pad_to_min(vol, min_side):
    z,y,x = vol.shape
    padz = max(0,min_side-z)
    pady = max(0,min_side-y)
    padx = max(0,min_side-x)
    pad = ((padz//2,padz-padz//2),
           (pady//2,pady-pady//2),
           (padx//2,padx-padx//2))
    return np.pad(vol,pad,'constant'),pad

def process_case(row, cache_dir):
    cid = row["case_id"]
    out_dir = Path(cache_dir)/cid
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir/"t1c.npy").exists() and (out_dir/"seg.npy").exists():
        return

    imgs = {}
    for m in ["t1c","t1n","t2w","t2f","seg"]:
        p = row[f"path_{m}"]
        img = sitk.ReadImage(p)
        img = sitk_resample_to_spacing(img, TARGET_SPACING, is_label=(m=="seg"))
        imgs[m] = img

    arrs = {m: sitk.GetArrayFromImage(imgs[m]) for m in imgs}
    seg = arrs["seg"].astype(np.uint8)

    mask = (arrs["t1c"]!=0)|(arrs["t1n"]!=0)|(arrs["t2w"]!=0)|(arrs["t2f"]!=0)
    vols = {}
    for m in ["t1c","t1n","t2w","t2f"]:
        vols[m] = zscore_clip(arrs[m].astype(np.float32), mask)

    support = mask | (seg>0)
    z0,z1,y0,y1,x0,x1 = tight_bbox(support)
    for m in vols:
        vols[m] = vols[m][z0:z1, y0:y1, x0:x1]
    seg = seg[z0:z1, y0:y1, x0:x1]

    for m in vols:
        vols[m], _ = pad_to_min(vols[m], MIN_SIDE)
    seg, _ = pad_to_min(seg, MIN_SIDE)

    np.save(out_dir/"t1c.npy", vols["t1c"].astype(np.float32))
    np.save(out_dir/"t1n.npy", vols["t1n"].astype(np.float32))
    np.save(out_dir/"t2w.npy", vols["t2w"].astype(np.float32))
    np.save(out_dir/"t2f.npy", vols["t2f"].astype(np.float32))
    np.save(out_dir/"seg.npy",  seg.astype(np.uint8))

    meta = dict(case_id=cid,
                spacing=list(TARGET_SPACING),
                crop=[z0,z1,y0,y1,x0,x1],
                shape=list(seg.shape))
    with open(out_dir/"meta.json","w") as f:
        json.dump(meta,f,indent=2)
