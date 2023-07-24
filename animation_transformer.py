import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

from PIL import Image

from dataset import AnimationDataset
from network import AnimationTransformer
from utils import gen_rand_labels

from zhenglin.dl.utils import weights_init_normal, LinearLambdaLR

parser = argparse.ArgumentParser()

#### dataset args
parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--patch_size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')

### training args
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--end_epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--embed_dim', type=int, default=256, help='initial learning rate')
parser.add_argument('--head_num', type=int, default=4, help='initial learning rate')
parser.add_argument('--resume', action="store_true", help='continue training from a checkpoint')
args = parser.parse_args()

### set gpu device
DEVICE = 0

# Networks
generator = AnimationTransformer(args.embed_dim, args.head_num).to(DEVICE)
generator.apply(weights_init_normal)

if args.resume:
    generator.load_state_dict(torch.load('./models/model_20.pth', map_location=DEVICE))

### if rich
# model = nn.DataParallel(model, device_ids=[0, 1])

### Lossess
criterion_forward = torch.nn.CrossEntropyLoss().to(DEVICE)
criterion_cycle = torch.nn.CrossEntropyLoss().to(DEVICE)

### argsimizers & LR schedulers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LinearLambdaLR(args.start_epoch, args.end_epoch, args.decay_epoch).step)

### Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor

### Dataset loader
transforms_ = [transforms.Resize(int(args.size*1.12), Image.BICUBIC),
               transforms.RandomCrop(args.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataset = AnimationDataset(args.dataroot, transforms_=transforms_, unaligned=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=args.n_cpu)

###### Training ######
for epoch in range(args.start_epoch, args.end_epoch + 1):
    for i, batch in enumerate(dataloader):
        info0, pth0 = batch['info_ref'], batch['patches_ref']
        info1, pth1 = batch['info_target'], batch['patches_target']

        info0_tensor = Variable(info0.type(Tensor)).to(DEVICE)
        pth0_tensor = Variable(pth0.type(Tensor)).to(DEVICE)
        info1_tensor = Variable(info1.type(Tensor)).to(DEVICE)
        pth1_tensor = Variable(pth1.type(Tensor)).to(DEVICE)

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        pred_color_ids = generator(ref=pth0_tensor,
                                   target=pth1_tensor,
                                   ref_bboxes=info0_tensor[:-1],
                                   target_bboxes=info1_tensor[:-1],
                                   labels=info0_tensor[-1]
                                   )

        loss_forward = criterion_forward(info1_tensor[-1], pred_color_ids)

        """
            In case correspondance label is not available,  
            assign each segment with a random ID, and use
            cycle consistency to help learning weak matching
        """

        rand_cor_labels = gen_rand_labels(pth0.shape[0])
        rand_labels_tensor = Variable(rand_cor_labels.type(Tensor)).to(DEVICE)

        ### ref to target
        pred_cor_labels_A2B = generator(ref=pth0_tensor,
                                        target=pth1_tensor,
                                        ref_bboxes=info0_tensor[:-1],
                                        target_bboxes=info1_tensor[:-1],
                                        color_ids=rand_labels_tensor
                                        )
        ### target to ref
        pred_cor_labels_B2A = generator(ref=pth1_tensor,
                                        target=pth0_tensor,
                                        ref_bboxes=info1_tensor[:-1],
                                        target_bboxes=info0_tensor[:-1],
                                        color_ids=pred_cor_labels_A2B
                                        )

        loss_cycle = criterion_cycle(pred_cor_labels_A2B, pred_cor_labels_B2A)

        loss_G = loss_forward + loss_cycle * 0.25

        loss_G.backward()
        optimizer_G.step()

        # wandb.log({"loss_G": loss_G.item()})

    ### Update learning rates
    lr_scheduler_G.step()

    ### show images and save models
    if epoch % 20 == 0:
        torch.save(generator.state_dict(), f'output/model_{epoch}.pth')
###################################
