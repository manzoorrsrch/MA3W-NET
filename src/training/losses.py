import torch
import torch.nn as nn
import torch.nn.functional as F

bce_logits = nn.BCEWithLogitsLoss()

def soft_dice_loss(logits, target, eps=1e-5):
    probs = torch.softmax(logits, dim=1)
    dims = (0,2,3,4)
    inter = (probs*target).sum(dim=dims)
    denom = (probs+target).sum(dim=dims)
    dice = (2*inter+eps)/(denom+eps)
    return 1-dice.mean()

def focal_loss(logits, target, gamma=2.0, alpha=None, eps=1e-8):
    p = torch.softmax(logits,dim=1).clamp(eps,1-eps)
    ce = -(target*torch.log(p))
    if alpha is not None:
        a = torch.tensor(alpha,device=logits.device).view(1,-1,1,1,1)
        ce = ce*a
    fl = ((1-p)**gamma)*ce
    return fl.sum(dim=1).mean()

def dice_metric(y_pred, y_true, eps=1e-5):
    dims = (0,2,3,4)
    inter = (y_pred*y_true).sum(dim=dims)
    denom = (y_pred+y_true).sum(dim=dims)
    dice = (2*inter+eps)/(denom+eps)
    return dice

def argmax_onehot(logits):
    arg = logits.argmax(dim=1, keepdim=True)
    return torch.zeros_like(logits).scatter_(1,arg,1.0)

def compute_loss(logits, bmap_logits, target,
                 boundary_weight=0.2, gamma=2.0):
    Ld = soft_dice_loss(logits,target)
    Lf = focal_loss(logits,target,gamma=gamma)

    with torch.no_grad():
        tumor_mask = (target[:,1:,...].sum(dim=1,keepdim=True)>0)
        e = torch.zeros_like(tumor_mask, dtype=torch.bool)
        e[:,:,1:,:,:] |= (tumor_mask[:,:,1:,:,:]!=tumor_mask[:,:,:-1,:,:])
        e[:,:,:,1:,:] |= (tumor_mask[:,:,:,1:,:]!=tumor_mask[:,:,:,:-1,:])
        e[:,:,:,:,1:] |= (tumor_mask[:,:,:,:,1:]!=tumor_mask[:,:,:,:,:-1])
        e = e.float()

    Lb = bce_logits(bmap_logits, e)
    total = Ld + Lf + boundary_weight*Lb
    return total, dict(dice=Ld.item(), focal=Lf.item(), bce=Lb.item())
