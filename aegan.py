import os
import json

import torch
from torch import nn
from torch import optim
import torchvision as tv
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

class Generator(nn.Module):
    def __init__(self, channels, kernels, strides, initial_shape, latent_dim=100, batchnorm=True):
        """A generator for mapping a latent space to a sample space.

        The sample space for this generator is single-channel, 28x28 images
        with pixel intensity ranging from -1 to +1.

        Args:
            latent_dim (int): latent dimension ("noise vector")
            batchnorm (bool): Whether or not to use batch normalization
        """
        super(Generator, self).__init__()
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.latent_dim = latent_dim
        self.initial_shape = initial_shape
        self.batchnorm = batchnorm
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.projections = nn.ModuleList()

        self.projections.append(nn.Linear(
                self.latent_dim,
                np.prod(self.initial_shape),
                bias=False))
        if self.batchnorm:
            self.projections.append(
                    nn.BatchNorm1d(np.prod(self.initial_shape)))
        leaky_relu = nn.LeakyReLU()
        self.projections.append(leaky_relu)

        # Convolutions
        self.conv = nn.ModuleList()
        last_channels = self.initial_shape[0]
        for index in range(len(self.channels)):
            padding = 2 if self.strides[index] == 1 else 1  # TODO: fix this
            self.conv.append(nn.ConvTranspose2d(
                    in_channels=last_channels,
                    out_channels=self.channels[index],
                    kernel_size=self.kernels[index],
                    stride=self.strides[index],
                    padding=padding,
                    bias=False))
            last_channels = self.channels[index]
            if index + 1 != len(self.channels):
                if self.batchnorm:
                    self.conv.append(nn.BatchNorm2d(self.channels[index]))
                self.conv.append(leaky_relu)
        else:
            self.conv.append(nn.Tanh())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        intermediate = input_tensor
        for module in self.projections:
            intermediate = module(intermediate)

        intermediate = intermediate.view(-1, *self.initial_shape)

        for module in self.conv:
            intermediate = module(intermediate)
        return intermediate


class Encoder(nn.Module):
    def __init__(self, channels, kernels, strides, image_dims, latent_dim, device="cpu", batchnorm=True):
        """An image encoder."""
        super(Encoder, self).__init__()
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.image_dims = image_dims
        self.latent_dim = latent_dim
        self.batchnorm = batchnorm
        self.device = device
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.conv = nn.ModuleList()
        last_channels = self.image_dims[0]
        leaky_relu = nn.LeakyReLU()
        for index in range(len(self.channels)):
            self.conv.append(nn.Conv2d(
                    in_channels=last_channels,
                    out_channels=self.channels[index],
                    kernel_size=self.kernels[index],
                    stride=self.strides[index],
                    padding=self.kernels[index] // 2,
                    bias=False))
            last_channels = self.channels[index]
            if self.batchnorm:
                self.conv.append(nn.BatchNorm2d(self.channels[index]))
            self.conv.append(leaky_relu)

        self.linear = nn.ModuleList()
        self.width = last_channels * np.prod(self.image_dims[1:]) // np.prod(self.strides)**2
        self.linear.append(nn.Linear(self.width, 128))
        if self.batchnorm:
            self.conv.append(nn.BatchNorm1d(128))
        self.linear.append(leaky_relu)
        self.linear.append(nn.Linear(128, 128))
        if self.batchnorm:
            self.conv.append(nn.BatchNorm1d(128))
        self.linear.append(leaky_relu)
        self.linear.append(nn.Linear(128, self.latent_dim))

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        intermediate = input_tensor + torch.randn(input_tensor.size(), device=self.device) * 0.1
        for module in self.conv:
            intermediate = module(intermediate)

        intermediate = intermediate.view(-1, self.width)

        for module in self.linear:
            intermediate = module(intermediate)

        return intermediate

class DiscriminatorImage(nn.Module):
    def __init__(self, channels, kernels, strides, image_dims, device="cpu", batchnorm=True):
        """A discriminator for discerning real from generated images."""
        super(DiscriminatorImage, self).__init__()
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.image_dims = image_dims
        self.batchnorm = batchnorm
        self.device = device
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.conv = nn.ModuleList()
        last_channels = self.image_dims[0]
        leaky_relu = nn.LeakyReLU()
        for index in range(len(self.channels)):
            self.conv.append(nn.Conv2d(
                    in_channels=last_channels,
                    out_channels=self.channels[index],
                    kernel_size=self.kernels[index],
                    stride=self.strides[index],
                    padding=self.kernels[index] // 2,
                    bias=False))
            last_channels = self.channels[index]
            if self.batchnorm:
                self.conv.append(nn.BatchNorm2d(self.channels[index]))
            self.conv.append(leaky_relu)

        self.classifier = nn.ModuleList()
        self.width = last_channels * np.prod(self.image_dims[1:]) // np.prod(self.strides)**2
        self.classifier.append(nn.Linear(self.width, 1))
        self.classifier.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        intermediate = input_tensor + torch.randn(input_tensor.size(), device=self.device) * 0.1
        for module in self.conv:
            intermediate = module(intermediate)


        intermediate = intermediate.view(-1, self.width)

        for module in self.classifier:
            intermediate = module(intermediate)

        return intermediate


class DiscriminatorLatent(nn.Module):
    def __init__(self, widths, latent_dim=100, device="cpu", batchnorm=True):
        """A discriminator for discerning real from generated images."""
        super(DiscriminatorLatent, self).__init__()
        self.latent_dim = latent_dim
        self.widths = widths
        self.batchnorm = batchnorm
        self.device = device
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.linear = nn.ModuleList()
        last_width = self.latent_dim
        leaky_relu = nn.LeakyReLU()
        for width in self.widths:
            self.linear.append(nn.Linear(
                    last_width,
                    width,
                    bias=True))
            last_width = width
            if self.batchnorm:
                self.linear.append(nn.BatchNorm1d(width))
            self.linear.append(leaky_relu)

        self.linear.append(nn.Linear(last_width, 1))
        self.linear.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        intermediate = input_tensor
        for module in self.linear:
            intermediate = module(intermediate)
        return intermediate


class GAN():
    def __init__(self, latent_dim, noise_fn, dataloader, param_file,
                 batch_size=32, device='cpu', lr_d=1e-3, lr_g=2e-4):
        """A very basic DCGAN class for generating MNIST digits

        Args:
            generator: a Ganerator network
            discriminator: A Discriminator network
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            dataloader: a pytorch dataloader for loading images
            batch_size: training batch size. Must match that of dataloader
            device: cpu or CUDA
            lr_d: learning rate for the discriminator
            lr_g: learning rate for the generator
        """
        with open(param_file, 'r') as f:
            params = json.load(f)
        self.latent_dim = latent_dim
        self.image_dims = params["image_dims"]
        self.device = device

        self.generator = Generator(
                latent_dim=self.latent_dim,
                **params["generator"]
                ).to(self.device)
        self.discriminator_image = DiscriminatorImage(
                image_dims=self.image_dims,
                device=self.device,
                **params["discriminator_image"]
                ).to(device)

        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.criterion = nn.BCELoss()
        self.optim_d = optim.Adam(self.discriminator_image.parameters(),
                                  lr=lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=lr_g, betas=(0.5, 0.999))
        self.target_ones = torch.ones((batch_size, 1), device=device)
        self.target_zeros = torch.zeros((batch_size, 1), device=device)

    def generate_samples(self, latent_vec=None, num=None):
        """Sample images from the generator.

        Images are returned as a 4D tensor of values between -1 and 1.
        Dimensions are (number, channels, height, width). Returns the tensor
        on cpu.

        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None

        If latent_vec and num are None then use self.batch_size
        random latent vectors.
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        samples = samples.cpu()  # move images to cpu
        return samples

    def train_step_generator(self):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()

        latent_vec = self.noise_fn(self.batch_size)
        generated = self.generator(latent_vec)
        classifications = self.discriminator_image(generated)
        loss = self.criterion(classifications, self.target_ones)
        loss.backward()
        self.optim_g.step()
        return loss.item()

    def train_step_discriminator(self, real_samples):
        """Train the discriminator one step and return the losses."""
        self.discriminator_image.zero_grad()

        # real samples
        pred_real = self.discriminator_image(real_samples)
        loss_real = self.criterion(pred_real, self.target_ones)

        # generated samples
        latent_vec = self.noise_fn(self.batch_size)
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator_image(fake_samples)
        loss_fake = self.criterion(pred_fake, self.target_zeros)

        # combine
        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self.optim_d.step()
        return loss_real.item(), loss_fake.item()

    def train_epoch(self, print_frequency=10, max_steps=0):
        """Train both networks for one epoch and return the losses.

        Args:
            print_frequency (int): print stats every `print_frequency` steps.
            max_steps (int): End epoch after `max_steps` steps, or set to 0
                             to do the full epoch.
        """
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        for batch, (real_samples, _) in enumerate(self.dataloader):
            real_samples = real_samples.to(self.device)
            ldr_, ldf_ = self.train_step_discriminator(real_samples)
            loss_d_real_running += ldr_
            loss_d_fake_running += ldf_
            loss_g_running += self.train_step_generator()
            if print_frequency and (batch+1) % print_frequency == 0:
                print(f"{batch+1}/{len(self.dataloader)}:"
                      f" G={loss_g_running / (batch+1):.3f},"
                      f" Dr={loss_d_real_running / (batch+1):.3f},"
                      f" Df={loss_d_fake_running / (batch+1):.3f}",
                      end='\r',
                      flush=True)
            # if ldr_ > 90 or ldf_ > 90:
            #     print("\nNew Discriminator")
            #     self.discriminator = Discriminator().to(self.device)
            #     self.optim_d = optim.Adam(self.discriminator.parameters(),
            #                               lr=self.lr_d, betas=(0.5, 0.999))
            if max_steps and batch == max_steps:
                break
        if print_frequency:
            print()
        loss_g_running /= batch
        loss_d_real_running /= batch
        loss_d_fake_running /= batch
        return (loss_g_running, (loss_d_real_running, loss_d_fake_running))

class AEGAN():
    def __init__(self, latent_dim, noise_fn, dataloader, param_file,
                 batch_size=32, device='cpu'):
        """A very basic DCGAN class for generating MNIST digits

        Args:
            generator: a Ganerator network
            discriminator: A Discriminator network
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            dataloader: a pytorch dataloader for loading images
            batch_size: training batch size. Must match that of dataloader
            device: cpu or CUDA
            lr_d: learning rate for the discriminator
            lr_g: learning rate for the generator
        """
        with open(param_file, 'r') as f:
            params = json.load(f)
        self.latent_dim = latent_dim
        self.image_dims = params["image_dims"]
        self.device = device

        self.generator = Generator(
                latent_dim=self.latent_dim,
                **params["generator"]
                ).to(self.device)
        self.encoder = Encoder(
                image_dims=self.image_dims,
                latent_dim=self.latent_dim,
                device=self.device,
                **params["encoder"]
                ).to(self.device)
        self.discriminator_image = DiscriminatorImage(
                image_dims=self.image_dims,
                device=self.device,
                **params["discriminator_image"]
                ).to(device)
        self.discriminator_latent = DiscriminatorLatent(
                device=self.device,
                latent_dim=self.latent_dim,
                **params["discriminator_latent"],
                ).to(self.device)

        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size

        self.alphas =  params["alpha"]

        self.criterion_gen = nn.BCELoss()
        self.criterion_recon_image = nn.L1Loss()
        self.criterion_recon_latent = nn.MSELoss()
        self.optim_di = optim.Adam(self.discriminator_image.parameters(),
                                   lr=1e-3, betas=(0.5, 0.999))
        self.optim_dl = optim.Adam(self.discriminator_latent.parameters(),
                                   lr=1e-3, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=2e-4, betas=(0.5, 0.999))
        self.optim_e = optim.Adam(self.encoder.parameters(),
                                  lr=2e-4, betas=(0.5, 0.999))
        self.target_ones = torch.ones((batch_size, 1), device=device)
        self.target_zeros = torch.zeros((batch_size, 1), device=device)

    def generate_samples(self, latent_vec=None, num=None):
        """Sample images from the generator.

        Images are returned as a 4D tensor of values between -1 and 1.
        Dimensions are (number, channels, height, width). Returns the tensor
        on cpu.

        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None

        If latent_vec and num are None then use self.batch_size
        random latent vectors.
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        samples = samples.cpu()  # move images to cpu
        return samples

    def train_step_generators(self, X):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()
        self.encoder.zero_grad()

        Z = self.noise_fn(self.batch_size)

        X_hat = self.generator(Z)
        Z_hat = self.encoder(X)
        X_tilde = self.generator(Z_hat)
        Z_tilde = self.encoder(X_hat)

        X_hat_confidence = self.discriminator_image(X_hat)
        Z_hat_confidence = self.discriminator_latent(Z_hat)
        X_tilde_confidence = self.discriminator_image(X_tilde)
        Z_tilde_confidence = self.discriminator_latent(Z_tilde)

        X_hat_loss = self.criterion_gen(X_hat_confidence, self.target_ones)
        Z_hat_loss = self.criterion_gen(Z_hat_confidence, self.target_ones)
        X_tilde_loss = self.criterion_gen(X_tilde_confidence, self.target_ones)
        Z_tilde_loss = self.criterion_gen(Z_tilde_confidence, self.target_ones)

        X_recon_loss = self.criterion_recon_image(X_tilde, X) * self.alphas["reconstruct_image"]
        Z_recon_loss = self.criterion_recon_latent(Z_tilde, Z) * self.alphas["reconstruct_latent"]

        X_loss = (X_hat_loss + X_tilde_loss) / 2 * self.alphas["discriminate_image"]
        Z_loss = (X_hat_loss + X_tilde_loss) / 2 * self.alphas["discriminate_latent"]
        loss = X_loss + Z_loss + X_recon_loss + Z_recon_loss
        loss.backward()
        self.optim_e.step()
        self.optim_g.step()

        return X_loss.item(), Z_loss.item(), X_recon_loss.item(), Z_recon_loss.item()

    def train_step_discriminators(self, X):
        """Train the discriminator one step and return the losses."""
        self.discriminator_image.zero_grad()
        self.discriminator_latent.zero_grad()

        Z = self.noise_fn(self.batch_size)

        with torch.no_grad():
            X_hat = self.generator(Z)
            Z_hat = self.encoder(X)
            X_tilde = self.generator(Z_hat)
            Z_tilde = self.encoder(X_hat)

        X_confidence = self.discriminator_image(X)
        X_hat_confidence = self.discriminator_image(X_hat)
        X_tilde_confidence = self.discriminator_image(X_tilde)
        Z_confidence = self.discriminator_latent(Z)
        Z_hat_confidence = self.discriminator_latent(Z_hat)
        Z_tilde_confidence = self.discriminator_latent(Z_tilde)

        X_loss = 2 * self.criterion_gen(X_confidence, self.target_ones)
        X_hat_loss = self.criterion_gen(X_hat_confidence, self.target_zeros)
        X_tilde_loss = self.criterion_gen(X_tilde_confidence, self.target_zeros)
        Z_loss = 2 * self.criterion_gen(Z_confidence, self.target_ones)
        Z_hat_loss = self.criterion_gen(Z_hat_confidence, self.target_zeros)
        Z_tilde_loss = self.criterion_gen(Z_tilde_confidence, self.target_zeros)

        loss_images = (X_loss + X_hat_loss + X_tilde_loss) / 4
        loss_latent = (Z_loss + Z_hat_loss + Z_tilde_loss) / 4
        loss = loss_images + loss_latent
        loss.backward()
        self.optim_di.step()
        self.optim_dl.step()

        return loss_images.item(), loss_latent.item()

    def train_epoch(self, print_frequency=10, max_steps=0):
        """Train both networks for one epoch and return the losses.

        Args:
            print_frequency (int): print stats every `print_frequency` steps.
            max_steps (int): End epoch after `max_steps` steps, or set to 0
                             to do the full epoch.
        """
        ldx, ldz, lgx, lgz, lrx, lrz = 0, 0, 0, 0, 0, 0
        for batch, (real_samples, _) in enumerate(self.dataloader):
            real_samples = real_samples.to(self.device)
            ldx_, ldz_ = self.train_step_discriminators(real_samples)
            ldx += ldx_
            ldz += ldz_
            lgx_, lgz_, lrx_, lrz_ = self.train_step_generators(real_samples)
            lgx += lgx_
            lgz += lgz_
            lrx += lrx_
            lrz += lrz_
            if print_frequency and (batch+1) % print_frequency == 0:
                print(f"{batch+1}/{len(self.dataloader)}:"
                      f" G={lgx / (batch+1):.3f},"
                      f" E={lgz / (batch+1):.3f},"
                      f" Dx={ldx / (batch+1):.3f},"
                      f" Dz={ldz / (batch+1):.3f}",
                      f" Rx={lrx / (batch+1):.3f}",
                      f" Rz={lrz / (batch+1):.3f}",
                      end='\r',
                      flush=True)
            # if ldr_ > 90 or ldf_ > 90:
            #     print("\nNew Discriminator")
            #     self.discriminator = Discriminator().to(self.device)
            #     self.optim_d = optim.Adam(self.discriminator.parameters(),
            #                               lr=self.lr_d, betas=(0.5, 0.999))
            if max_steps and batch == max_steps:
                break
        if print_frequency:
            print()
        lgx /= batch
        lgz /= batch
        ldx /= batch
        ldz /= batch
        lrx /= batch
        lrz /= batch
        return lgx, lgz, ldx, ldz, lrx, lrz


def save_images(GAN, vec, filename):
    images = GAN.generate_samples(vec)
    ims = tv.utils.make_grid(images, normalize=True)
    ims = ims.numpy().transpose((1,2,0))
    ims = np.array(ims*255, dtype=np.uint8)
    image = Image.fromarray(ims)
    image.save(filename)


def main():
    import matplotlib.pyplot as plt
    from time import time
    try:
        os.makedirs("results/generated")
        os.makedirs("results/reconstructed")
    except:
        pass
    batch_size = 32
    latent_dim = 64
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = tv.transforms.Compose([
            tv.transforms.ColorJitter(hue=0.5),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            #tv.transforms.Resize(64),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ])
    dataset = ImageFolder(
            root=os.path.join("data", "pokemon"),
            transform=transform
            )
    dataloader = DataLoader(dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
            )
    X = iter(dataloader)
    test_ims, _ = next(X)
    test_ims_show = tv.utils.make_grid(test_ims, normalize=True)
    test_ims_show = test_ims_show.numpy().transpose((1,2,0))
    test_ims_show = np.array(test_ims_show*255, dtype=np.uint8)
    image = Image.fromarray(test_ims_show)
    image.save("results/reconstructed/test_images.png")

    noise_fn = lambda x: torch.randn((x, latent_dim), device=device)
    test_noise = noise_fn(32)
    gan = AEGAN(latent_dim, noise_fn, dataloader, "params/pokemon_96.json", device=device, batch_size=batch_size)
    start = time()
    for i in range(1000):
        print(f"Epoch {i+1}; Elapsed time = {int(time() - start)}s")
        gan.train_epoch()

        save_images(gan, test_noise, f"results/generated/gen.{i:04d}.png")

        with torch.no_grad():
            reconstructed = gan.generator(gan.encoder(test_ims.cuda())).cpu()
        reconstructed = tv.utils.make_grid(reconstructed, normalize=True)
        reconstructed = reconstructed.numpy().transpose((1,2,0))
        reconstructed = np.array(reconstructed*255, dtype=np.uint8)
        reconstructed = Image.fromarray(reconstructed)
        reconstructed.save(f"results/reconstructed/gen.{i:04d}.png")

    images = gan.generate_samples()
    ims = tv.utils.make_grid(images, normalize=True)
    plt.imshow(ims.numpy().transpose((1,2,0)))
    plt.show()


if __name__ == "__main__":
    main()
