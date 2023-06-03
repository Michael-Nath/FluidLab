import os
import pickle
import datetime
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.utils import save_image
from dataloader import NumPyTrajectoryDataset
import argparse


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
        self.bottle = EncoderModule(color_channels, 32, stride=2, kernel=2, pad=0)
        self.m1 = EncoderModule(32, 64, stride=2, kernel=2, pad=0)
        self.m2 = EncoderModule(64, 64, stride=1, kernel=3, pad=1)
        self.m3 = EncoderModule(64, 128, stride=1, kernel=3, pad=1)

    def forward(self, x):
        out1 = self.bottle(x)
        out2 = self.m1(out1)
        out3 = self.m2(out2)
        out = self.m3(out3)
        return out.view(out.shape[0], -1)


class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, stride, pad, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=kernel, stride=stride, padding=pad
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
        self.m1 = DecoderModule(128, 64, kernel=4,stride=2, pad=1)
        self.m2 = DecoderModule(64, 32, kernel=4,stride=2, pad=1)
        self.m3 = DecoderModule(64, 32, kernel=4,stride=2, pad=1)
        self.bottle = DecoderModule(32, color_channels, kernel=3, stride=1, pad=1, activation="sigmoid")

    def forward(self, x):
        out = x.view(x.shape[0], 128, self.decoder_input_size, self.decoder_input_size)
        out = self.bottle(self.m2(self.m1(out)))
        return out


class VAE(nn.Module):
    def __init__(self, model_name, dataset):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()
        self.n_latent_features = 32
        pooling_kernel = [2,2]
        encoder_output_size = 64
        color_channels = 3
        n_neurons_middle_layer = 128 * encoder_output_size * encoder_output_size

        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.bn = nn.BatchNorm1d(self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)

        self.train_loader, self.val_loader, self.test_loader = self.load_data(dataset)
        # history
        self.history = {"loss": [], "val_loss": []}

        # model name
        self.model_name = model_name
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
        train = NumPyTrajectoryDataset(dataset, train=True)
        train, val = random_split(train, [0.7, 0.3])
        test = NumPyTrajectoryDataset(dataset, train=False)
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=64, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val, batch_size=64, shuffle=True, num_workers=2
        )
        # test_loader = torch.utils.data.DataLoader(
        #     test, batch_size=32, shuffle=True, num_workers=2
        # )
        test_loader = None
        return train_loader, val_loader, test_loader

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=True)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (BCE + KLD).cuda(self.device)

    def init_model(self):
        wandb.init(
        project="vae-training-fluid-manip",
        config={
            "learning-rate": 1e-3
            }
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        if self.device == "cuda":
            self = self.cuda()
            torch.backends.cudnn.benchmark = True
        self.to(self.device)

    # Train
    def fit_train(self, epoch, out_weights_file):
        self.train()
        print(f"\nEpoch: {epoch+1:d} {datetime.datetime.now()}")
        train_loss = []
        samples_cnt = 0
        for batch_idx, (inputs, _, _, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            inputs = inputs / 255
            inputs = inputs.to(self.device)
            recon_batch, mu, logvar = self(inputs)
            loss = self.loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            wandb.log({"train_loss": loss})
            self.optimizer.step()
            train_loss.append(loss.item())
            samples_cnt += inputs.size(0)
            if batch_idx == 250:
                break
        print(f"Epoch {epoch + 1:d} | End of Epoch Train Loss: {loss.item()}")
        wandb.log({"End of Epoch Loss": loss.item()})

    def test(self, epoch, in_weights_file, out_recon_folder):
        if not os.path.exists(f"{self.model_name}/{out_recon_folder}"):
            os.mkdir(f"{self.model_name}/{out_recon_folder}")
        # self.train()
        val_loss = 0
        samples_cnt = 0
        # self.load_state_dict(torch.load(in_weights_file))
        with torch.no_grad():
            for batch_idx, (inputs, _, _, _) in enumerate(self.val_loader):
                inputs = inputs.to(self.device)
                inputs = inputs / 255
                recon_batch, mu, logvar = self(inputs)
                loss = self.loss_function(recon_batch, inputs, mu, logvar).item() 
                val_loss += loss
                samples_cnt += 1
                if batch_idx == 0:
                    save_image(
                        inputs,
                        f"{self.model_name}/{out_recon_folder}/input_epoch_{str(epoch)}.png",
                        nrow=8,
                    )
                    save_image(
                        recon_batch,
                        f"{self.model_name}/{out_recon_folder}/reconstruction_epoch_{str(epoch)}.png",
                        nrow=8,
                    )
                if batch_idx == 25:
                    break
        val_loss /= samples_cnt
        print(f"ValLoss: {val_loss}")
        wandb.log({"End of Epoch Val Loss": val_loss})
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_weights_file", type=str, default="vae_weights.pt")
    parser.add_argument("--in_weights_file", type=str, default="vae_weights.pt")
    parser.add_argument("--out_recon_folder", type=str, default="test_recons")
    parser.add_argument("--in_trajs_file", type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    net.init_model()
    net = VAE("latteart-recon", args.in_trajs_file)
    for i in range(50):
        net.fit_train(i, args.out_weights_file)
        torch.save(net.state_dict(), args.out_weights_file)
        with torch.no_grad():
            net.test(i, args.in_weights_file, args.out_recon_folder)
            torch.cuda.empty_cache()
    # net.save_history()
    
if __name__ == "__main__":
    main()
