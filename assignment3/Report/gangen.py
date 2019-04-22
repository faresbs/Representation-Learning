class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU()
            nn.Linear(128, 256),
            nn.ReLU()
            nn.Linear(256, 512),
            nn.ReLU()
            nn.Linear(512, 1024),
            nn.ReLU()
            nn.Linear(1024, 2048),
            nn.ReLU()
            nn.Linear(2048, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img