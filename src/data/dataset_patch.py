import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

PATCH = 112
TUMOR_CENTER_PROB = 0.6

class BratsPatchDataset(Dataset):
    """
    Patch-level dataset for BRATS2023/24 NPZ cache.
    Crops 112³ patches, biased toward tumor locations.
    """
    def __init__(self, case_ids, cache_dir, patch=PATCH,
                 tumor_center_prob=TUMOR_CENTER_PROB, augment=True):
        self.ids = case_ids
        self.cache = Path(cache_dir)
        self.patch = patch
        self.tumor_center_prob = tumor_center_prob
        self.augment = augment

    def _load_case(self, cid):
        d = self.cache / cid
        t1c = np.load(d/"t1c.npy", mmap_mode='r')
        t1n = np.load(d/"t1n.npy", mmap_mode='r')
        t2w = np.load(d/"t2w.npy", mmap_mode='r')
        t2f = np.load(d/"t2f.npy", mmap_mode='r')
        seg = np.load(d/"seg.npy", mmap_mode='r')
        vol = np.stack([t1c,t1n,t2w,t2f], axis=0)  # (C,Z,Y,X)
        return vol, seg

    def _rand_center(self, seg):
        Z,Y,X = seg.shape
        if np.random.rand() < self.tumor_center_prob and (seg>0).any():
            coords = np.array(np.where(seg>0)).T
            tz,ty,tx = coords[np.random.randint(len(coords))]
        else:
            tz,ty,tx = np.random.randint(Z), np.random.randint(Y), np.random.randint(X)
        return tz,ty,tx

    def _crop_patch(self, vol, seg, cz, cy, cx):
        ps = self.patch
        C,Z,Y,X = vol.shape
        z0 = np.clip(cz - ps//2, 0, Z-ps)
        y0 = np.clip(cy - ps//2, 0, Y-ps)
        x0 = np.clip(cx - ps//2, 0, X-ps)
        z1,y1,x1 = z0+ps, y0+ps, x0+ps
        v = vol[:, z0:z1, y0:y1, x0:x1]
        s = seg[z0:z1, y0:y1, x0:x1]
        return v, s

    def _augment(self, v, s):
        if np.random.rand()<0.5:
            v = v[:,:,:,::-1]; s = s[:,:,::-1]
        if np.random.rand()<0.5:
            v = v[:,:,::-1,:]; s = s[:,::-1,:]
        if np.random.rand()<0.5:
            v = v[:,::-1,:,:]; s = s[::-1,:,:]
        if np.random.rand()<0.15:
            ch = np.random.randint(4)
            v[ch] = 0.0
        return v, s

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cid = self.ids[idx]
        vol, seg = self._load_case(cid)
        cz,cy,cx = self._rand_center(seg)
        v, s = self._crop_patch(vol, seg, cz,cy,cx)
        if self.augment:
            v, s = self._augment(v, s)

        v = torch.from_numpy(v.copy()).float()
        s = torch.from_numpy(s.copy()).long()

        # one-hot mask
        num_classes = 4
        oh = torch.zeros((num_classes,)+s.shape, dtype=torch.float32)
        for c in range(num_classes):
            oh[c] = (s==c).float()

        return dict(image=v, target=oh, case_id=cid)
