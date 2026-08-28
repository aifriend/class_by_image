import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from gan.paint.Discriminator import Discriminator
from gan.paint.Generator import Generator


epochs = 100
Batch_Size = 64
lr = 0.0002
beta1 = 0.5
over = 4
wtl2 = 0.999


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='dataset/train', help='path to dataset')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use (0 for CPU)')
    opt = parser.parse_args()

    os.makedirs("result/train/cropped", exist_ok=True)
    os.makedirs("result/train/real", exist_ok=True)
    os.makedirs("result/train/recon", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    transform = transforms.Compose([transforms.Resize(128),
                                    transforms.CenterCrop(128),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dset.ImageFolder(root=opt.dataroot, transform=transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=Batch_Size,
                                             shuffle=True, num_workers=2)

    ngpu = opt.ngpu
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    resume_epoch = 0

    netG = Generator()
    netG.apply(weights_init)

    netD = Discriminator()
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    criterionMSE = nn.MSELoss()

    netD.to(device)
    netG.to(device)
    criterion.to(device)
    criterionMSE.to(device)

    real_label = 1
    fake_label = 0

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    for epoch in range(resume_epoch, epochs):
        for i, data in enumerate(dataloader, 0):
            real_cpu, _ = data
            real_cpu = real_cpu.to(device)
            real_center_cpu = real_cpu[:, :, int(128 / 4):int(128 / 4) + int(128 / 2),
                              int(128 / 4):int(128 / 4) + int(128 / 2)]
            batch_size = real_cpu.size(0)
            input_cropped = real_cpu.clone()
            real_center = real_center_cpu.clone()
            input_cropped[:, 0, int(128 / 4 + over):int(128 / 4 + 128 / 2 - over),
            int(128 / 4 + over):int(128 / 4 + 128 / 2 - over)] = 2 * 117.0 / 255.0 - 1.0
            input_cropped[:, 1, int(128 / 4 + over):int(128 / 4 + 128 / 2 - over),
            int(128 / 4 + over):int(128 / 4 + 128 / 2 - over)] = 2 * 104.0 / 255.0 - 1.0
            input_cropped[:, 2, int(128 / 4 + over):int(128 / 4 + 128 / 2 - over),
            int(128 / 4 + over):int(128 / 4 + 128 / 2 - over)] = 2 * 123.0 / 255.0 - 1.0

            # start the discriminator by training with real data---
            netD.zero_grad()
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_center).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train the discriminator with fake data---
            fake = netG(input_cropped)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # train the generator now---
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG_D = criterion(output, label)

            wtl2Matrix = real_center.clone()
            wtl2Matrix.fill_(wtl2 * 10)
            wtl2Matrix[:, :, int(over):int(128 / 2 - over), int(over):int(128 / 2 - over)] = wtl2

            errG_l2 = (fake - real_center).pow(2)
            errG_l2 = errG_l2 * wtl2Matrix
            errG_l2 = errG_l2.mean()

            errG = (1 - wtl2) * errG_D + wtl2 * errG_l2

            errG.backward()

            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d / %d][%d / %d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.item(), errG_D.item(), errG_l2.item(), D_x, D_G_z1,))

            if i % 100 == 0:
                vutils.save_image(real_cpu,
                                  'result/train/real/real_samples_epoch_%03d.png' % (epoch))
                vutils.save_image(input_cropped,
                                  'result/train/cropped/cropped_samples_epoch_%03d.png' % (epoch))
                recon_image = input_cropped.clone()
                recon_image[:, :, int(128 / 4):int(128 / 4 + 128 / 2), int(128 / 4):int(128 / 4 + 128 / 2)] = fake
                vutils.save_image(recon_image,
                                  'result/train/recon/recon_center_samples_epoch_%03d.png' % (epoch))


if __name__ == '__main__':
    main()
