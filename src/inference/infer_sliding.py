import numpy as np
import torch
from monai.inferers import sliding_window_inference

def sw_logits(model, vol4, roi=112, overlap=0.5):
    device = next(model.parameters()).device
    x = torch.from_numpy(vol4[None]).float().to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        logits = sliding_window_inference(
            inputs=x,
            roi_size=(roi,roi,roi),
            overlap=overlap,
            sw_batch_size=1,
            predictor=lambda t: model(t)[0]
        )
    return logits[0].float().cpu().numpy()
