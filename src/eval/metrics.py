import torch
import numpy as np

def one_hot_labels(lbl, C=4):
    # lbl: (1,1,Z,Y,X)
    oh = torch.zeros((1,C)+lbl.shape[2:], dtype=torch.float32, device=lbl.device)
    for c in range(C):
        oh[:,c] = (lbl==c).float()
    return oh

def compute_composites(pred_oh, gt_oh):
    # expects shape (1,C,Z,Y,X)
    comps = {}
    comps["ET"] = (pred_oh[:,3], gt_oh[:,3])
    comps["TC"] = ((pred_oh[:,1] + pred_oh[:,3]).bool(),
                   (gt_oh[:,1]   + gt_oh[:,3]).bool())
    comps["WT"] = ((pred_oh[:,1:] .sum(dim=1)>0),
                   (gt_oh[:,1:]   .sum(dim=1)>0))
    return comps

def safe_comp_metrics(pred_bin, gt_bin):
    # pred_bin, gt_bin shape: (1,Z,Y,X) boolean tensors
    pred = pred_bin.squeeze().cpu().numpy().astype(np.uint8)
    gt   =   gt_bin.squeeze().cpu().numpy().astype(np.uint8)

    inter = np.logical_and(pred,gt).sum()
    denom = pred.sum()+gt.sum()
    dice = 2*inter/denom if denom>0 else 0.0

    if pred.sum()==0 or gt.sum()==0:
        hd = np.nan
    else:
        from scipy.spatial.distance import directed_hausdorff
        P = np.column_stack(np.where(pred>0))
        G = np.column_stack(np.where(gt>0))
        h1 = directed_hausdorff(P,G)[0]
        h2 = directed_hausdorff(G,P)[0]
        hd = max(h1,h2)
    return float(dice), float(hd)
