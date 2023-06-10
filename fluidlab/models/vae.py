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
from fluidlab.models.dataloader import NumPyTrajectoryDataset
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
    def __init__(self, color_channels, pooling_kernels, n_latent_features):
        self.n_latent_features = n_latent_features
        input_dim = 256
        super().__init__()
        self.l = nn.ModuleList()
        start = 1
        if start == input_dim // self.n_latent_features:
            self.l.append(EncoderModule(color_channels, 128, stride=1, kernel=3, pad=1))
        else:
            self.l.append(EncoderModule(color_channels, 64, stride=1, kernel=3, pad=1))
        while start < (input_dim // self.n_latent_features):
            if start * 2 >= input_dim // self.n_latent_features: 
                self.l = self.l.append(EncoderModule(64, 128, stride=2, kernel=2, pad=0))
            else:
                self.l = self.l.append(EncoderModule(64, 64, stride=2, kernel=2, pad=0))
            start *= 2

    def forward(self, x):
        out = x.clone()
        for mod in self.l:
            out = mod(out)
        return out.view(out.shape[0], -1)


class DecoderModule(nn.Module):
    def __init__(
        self, input_channels, output_channels, kernel, stride, pad, activation="relu"
    ):
        super().__init__()
        self.convt = nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
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
        self.l = nn.ModuleList()
        start = decoder_input_size
        channels = 128
        while (start != 256 and channels // 2 > color_channels):
            self.l = self.l.append(DecoderModule(channels, channels // 2, kernel=4, stride=2, pad=1))
            start *= 2
            channels = channels // 2
        self.l = self.l.append(DecoderModule(
            channels, color_channels, kernel=3, stride=1, pad=1, activation="sigmoid"
        ))

    def forward(self, x):
        # x = x.view(x.shape[0], 128, self.decoder_input_size, self.decoder_input_size) 
        out = x.view(x.shape[0], 128, self.decoder_input_size, self.decoder_input_size)
        for mod in self.l:
            out = mod(out)
        return out


class VAE(nn.Module):
    def __init__(self, model_name, dataset, cp_fname, n_latent_features, standalone=True):
        self.cp_fname = cp_fname
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()
        pooling_kernel = [2, 2]
        self.encoder_output_size = self.n_latent_features = n_latent_features
        color_channels = 3
        n_neurons_middle_layer = 128 * self.encoder_output_size * self.encoder_output_size

        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, self.n_latent_features)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.bn = nn.BatchNorm1d(self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, self.encoder_output_size)
        if standalone:
            self.train_loader, self.val_loader, self.test_loader = self.load_data(
                dataset
            )
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

    def forward(self, x, encode_only=False):
        # Encoder
        h = self.encoder(x)
        # # Bottle-neck
        z, mu, logvar = self._bottleneck(h)
        if encode_only:
            return z, mu, logvar
        # decoder
        z = self.fc3(z)
        d = self.decoder(z)
        return d, mu, logvar

    def load_data(self, dataset):
        train = NumPyTrajectoryDataset(dataset, train=True, amnt_trajs_test=500)
        train, val = random_split(train, [0.7, 0.3])
        test = NumPyTrajectoryDataset(dataset, train=False, amnt_trajs_test=500)
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

    def loss_function(self, recon_x, x, mu, logvar, variational=True, weighting=1):
        if not variational:
            KLD = 0
        else:
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            KLD *= weighting
            wandb.log({"KLD_LOSS": KLD})
        BCE = F.binary_cross_entropy(recon_x, x, size_average=True)
        wandb.log({"MSE_LOSS": BCE})
        return (BCE + KLD).cuda(self.device)

    def init_model(self):
        wandb.init(project="vae-training-fluid-manip", name=self.cp_fname, config={"learning-rate": 1e-3})
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        if self.device == "cuda":
            self = self.cuda()
            torch.backends.cudnn.benchmark = True
        self.to(self.device)

    # Train
    def fit_train(self, epoch, no_variational=False, weighting=1):
        self.train()
        print(f"\nEpoch: {epoch+1:d} {datetime.datetime.now()}")
        train_loss = []
        samples_cnt = 0
        for batch_idx, (inputs, _, _, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            inputs = inputs.to(self.device).type("torch.cuda.FloatTensor")
            recon_batch, mu, logvar = self(inputs)
            loss = self.loss_function(recon_batch, inputs, mu, logvar, not no_variational, weighting)
            loss.backward()
            wandb.log({"train_loss": loss})
            self.optimizer.step()
            train_loss.append(loss.item())
            samples_cnt += inputs.size(0)
            if batch_idx == 250:
                break
        print(f"Epoch {epoch + 1:d} | End of Epoch Train Loss: {loss.item()}")
        wandb.log({"End of Epoch Loss": loss.item()})

    def test(self, epoch, out_recon_folder, no_variational=False, weighting=1):
        if not os.path.exists(f"{self.model_name}/{out_recon_folder}"):
            os.mkdir(f"{self.model_name}/{out_recon_folder}")
        # self.train()
        val_loss = 0
        samples_cnt = 0
        # self.load_state_dict(torch.load(in_weights_file))
        with torch.no_grad():
            for batch_idx, (inputs, _, _, _) in enumerate(self.val_loader):
                inputs = inputs.to(self.device).type("torch.cuda.FloatTensor")
                recon_batch, mu, logvar = self(inputs)
                loss = self.loss_function(recon_batch, inputs, mu, logvar, not no_variational, weighting).item()
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
    parser.add_argument("--n_latent_features", type=int, default=32)
    parser.add_argument("--no_variational", action="store_true")
    parser.add_argument("--weighting", type=float, default=1.0)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    net = VAE("latteart-recon", args.in_trajs_file, args.out_weights_file, args.n_latent_features)
    net.init_model()
    for i in range(15):
        net.fit_train(i, args.no_variational, args.weighting)
        torch.save(net.state_dict(), args.out_weights_file)
        with torch.no_grad():
            net.test(i, args.out_recon_folder, args.no_variational, args.weighting)
            torch.cuda.empty_cache()
    # net.save_history()


if __name__ == "__main__":
    main()
