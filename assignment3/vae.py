import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device("cuda" if args.cuda else "cpu")


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        ##Encoder
        self.encoder = nn.sequential(
                nn.Conv2d(1, 32, kernel_size=3, bias=False, stride=1),
                nn.ReLU(True),
                F.avg_pool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, bias=False, stride=1),
                nn.ReLU(True),
                F.avg_pool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 256, kernel_size=5, bias=False, stride=1),
                nn.ReLU(True),
                #outputs 2 vectors of size 100, mean vector and std vector
                #TO SEE 100*100 ??
                nn.Linear(256, 100*2)
            )


        ##Decoder
        self.decoder = nn.sequential(
        #Takes z latent variable of size 100
                nn.Linear(100, 256),
                nn.ReLU(True),
                nn.Conv2d(256, 64, kernel_size=5, bias=False, stride=1, padding=4),
                nn.UpsamplingBilinear2d(size=None, scale_factor=2),
                nn.Conv2d(64, 32, kernel_size=3, bias=False, stride=1, padding=2),
                nn.ReLU(True),
                nn.UpsamplingBilinear2d(size=None, scale_factor=2),
                nn.Conv2d(32, 16, kernel_size=3, bias=False, stride=1, padding=2),
                nn.ReLU(True),
                nn.Conv2d(16, 1, kernel_size=3, bias=False, stride=1, padding=2)

            )


    #Outputs mean/vector for z probability distribution
    def encode(self, x):
        z = self.encoder(x)
        print(z.shape)
        #first 100 for mean vector, the other 100 for std vector
        return z[:100], z(100:200)

    #Outputs reconstructed x
    def decode(self, z):
        x = self.decoder(z)
        return x

    #Reperameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z


    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    ELBO = BCE + KLD
    loss = -ELBO

    return loss



model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

def train(epoch):
    #Mode train
    model.train()

    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



if __name__ == "__main__":
    epochs = 20
    for epoch in range(epochs):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')