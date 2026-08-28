import torch

from gan.paint.Discriminator import Discriminator as PaintDiscriminator
from gan.paint.Generator import Generator as PaintGenerator
from gan.run_gan import Discriminator as DcganDiscriminator
from gan.run_gan import Generator as DcganGenerator


def test_dcgan_generator_output_shape():
    net_g = DcganGenerator(nc=3, nz=100, ngf=64)
    noise = torch.randn(2, 100, 1, 1)
    assert net_g(noise).shape == (2, 3, 64, 64)


def test_dcgan_discriminator_output_shape():
    net_d = DcganDiscriminator(nc=3, ndf=64)
    image = torch.randn(2, 3, 64, 64)
    assert net_d(image).shape == (2, 1, 1, 1)


def test_paint_generator_reconstructs_center():
    # 128x128 input -> 64x64 reconstructed center
    net_g = PaintGenerator()
    cropped = torch.randn(2, 3, 128, 128)
    assert net_g(cropped).shape == (2, 3, 64, 64)


def test_paint_discriminator_scores_center():
    net_d = PaintDiscriminator()
    center = torch.randn(2, 3, 64, 64)
    out = net_d(center)
    assert out.shape == (2, 1, 1, 1)
    assert out.view(-1).shape == (2,)
