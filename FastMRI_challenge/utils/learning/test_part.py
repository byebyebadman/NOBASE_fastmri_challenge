import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet
from utils.model.archs.NAFNet_arch import NAFNet

class ConcatenateModels(nn.Module):
    def __init__(self, varnet_model, nafnet_model):
        super(ConcatenateModels, self).__init__()
        self.varnet_model = varnet_model
        self.nafnet_model = nafnet_model

    def forward(self, kspace, mask):
        varnet_output = self.varnet_model(kspace, mask)
        nafnet_output = self.nafnet_model(varnet_output)
        
        return nafnet_output


def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    varnetmodel = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    nafnetmodel = NAFNet(img_channel=1, width=8, middle_blk_num=1,
                      enc_blk_nums=[1, 1, 1, 12], dec_blk_nums=[1, 1, 1, 1])
    
    model = ConcatenateModels(varnetmodel, nafnetmodel)
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)