import os
import pickle
import datetime
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from dataloader import TrajectoryDataset, NumPyTrajectoryDataset


class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel,
            padding=pad,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        super().__init__()
        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out1 = self.bottle(x)
        out2 = self.m1(out1)
        out3 = self.m2(out2)
        out = self.m3(out3)
        # out = self.m3(self.m2(self.m1(self.bottle(x))))
        return out.view(32, -1)


class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=stride, stride=stride
        )
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))


class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        super().__init__()
        self.decoder_input_size = decoder_input_size
        self.m1 = DecoderModule(256, 2, stride=1)
        self.m2 = DecoderModule(2, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")

    def forward(self, x):
        out = x.view(32, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)


class VAE(nn.Module):
    def __init__(self, dataset):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()
        self.n_latent_features = 64
        pooling_kernel = [2, 2]
        encoder_output_size = 64
        color_channels = 3
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size

        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.bn = nn.BatchNorm1d(self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)

        self.train_loader, self.test_loader = self.load_data(dataset)
        # history
        self.history = {"loss": [], "val_loss": []}

        # model name
        self.model_name = dataset
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def _bottleneck(self, h):
        mu, logvar = self.bn(self.fc1(h)), self.bn(self.fc2(h))
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def sampling(self):
        # assume latent features space ~ N(0, 1)
        z = torch.randn(64, self.n_latent_features).to(self.device)
        z = self.fc3(z)
        # decode
        return self.decoder(z)

    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        # # Bottle-neck
        z, mu, logvar = self._bottleneck(h)
        # decoder
        z = self.fc3(z)
        d = self.decoder(z)
        return d, mu, logvar

    def load_data(self, dataset):
        data_transform = transforms.Compose([transforms.ToTensor()])
        train = NumPyTrajectoryDataset("trajs", train=True)
        test = NumPyTrajectoryDataset("trajs", train=False)
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=32, shuffle=True, num_workers=1
        )
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=32, shuffle=True, num_workers=0
        )
        return train_loader, test_loader

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=True)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (BCE + KLD).cuda(self.device)

    def init_model(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        if self.device == "cuda":
            self = self.cuda()
            torch.backends.cudnn.benchmark = True
        self.to(self.device)

    # Train
    def fit_train(self, epoch):
        self.train()
        print(f"\nEpoch: {epoch+1:d} {datetime.datetime.now()}")
        train_loss = 0
        samples_cnt = 0
        for batch_idx, (inputs, _, _, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            inputs = inputs / 255
            inputs = inputs.to(self.device)
            recon_batch, mu, logvar = self(inputs)
            loss = self.loss_function(recon_batch, inputs, mu, logvar)
            print(loss)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            samples_cnt += inputs.size(0)
            if batch_idx % 50 == 0:
                torch.save(self.state_dict(), "vae_weights_train.pt")
                print(
                    batch_idx,
                    len(self.train_loader),
                    f"Loss: {train_loss/samples_cnt:f}",
                )

        self.history["loss"].append(train_loss / samples_cnt)

    def test(self, epoch):
        self.eval()
        val_loss = 0
        samples_cnt = 0
        self.load_state_dict(torch.load("vae_weights_train.pt"))
        with torch.no_grad():
            for batch_idx, (inputs, _, _, _) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                inputs = inputs / 255
                recon_batch, mu, logvar = self(inputs)
                val_loss += self.loss_function(recon_batch, inputs, mu, logvar).item()
                samples_cnt += inputs.size(0)
                if batch_idx == 0:
                    save_image(
                        inputs,
                        f"{self.model_name}/input_epoch_{str(epoch)}.png",
                        nrow=8,
                    )
                    save_image(
                        recon_batch,
                        f"{self.model_name}/reconstruction_epoch_{str(epoch)}.png",
                        nrow=8,
                    )
                if batch_idx >= 25:
                    break

        print(batch_idx, len(self.test_loader), f"ValLoss: {val_loss/samples_cnt:f}")
        self.history["val_loss"].append(val_loss / samples_cnt)

        # sampling
        save_image(
            self.sampling(),
            f"{self.model_name}/sampling_epoch_{str(epoch)}.png",
            nrow=8,
        )

    # save results
    def save_history(self):
        with open(f"{self.model_name}/{self.model_name}_history.dat", "wb") as fp:
            pickle.dump(self.history, fp)


def main():
    net = VAE("latteart-recon")
    net.init_model()
    # net.fit_train(0)
    net.test(0)
    # net.save_history()


if __name__ == "__main__":
    main()
