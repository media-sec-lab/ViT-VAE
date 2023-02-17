import  torch
from    torch import nn
from model.my_vit import ViT




class VAE(nn.Module):



    def __init__(self):
        super(VAE, self).__init__()


        self.dropout = nn.Dropout(p=0.5)

        # [b, 784] => [b, 20]
        # u: [b, 10]
        # sigma: [b, 10]
        self.vit = ViT(
                        image_size = 64,
                        patch_size = 16,
                        num_classes = 20,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 2048,
                        dropout = 0.1,
                        emb_dropout = 0.1
        )


        self.linear_down = nn.Sequential(
            nn.Linear(1000,256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256,10),
            nn.Dropout(p=0.1),
        )
        self.linear_up = nn.Sequential(
            nn.Linear(10, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 1000),
            nn.BatchNorm1d(1000),
            nn.GELU(),
            nn.Linear(1000, 3*64*64),

        )





    def forward(self, noise):
        """

        :param x: [b, 1, 28, 28]
        :return:
        """
        # noise = torch.concat((noise,noise,noise),dim=1)
        m = noise.size(-1)

        batchsz = noise.size(0)
        out = self.vit(noise)
        # out = self.linear_down(out)

        mu, sigma = out.chunk(2, dim=1)

        # print(sigma)
        h = mu + sigma * torch.randn_like(sigma)
        # h = mu + torch.exp(sigma) * torch.randn_like(sigma)
        # if mode == 'train':
        #     h = mu + torch.exp(sigma) * torch.randn_like(sigma)
        # else:
        #     h = mu

        # print(mu.shape)
        x_hat = self.linear_up(h)
        x_hat = x_hat.reshape(batchsz,3,m,m)
        #
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (batchsz * m * m)






        return x_hat,kld
if __name__ == '__main__':
    img = torch.randn(2, 3, 64, 64).cuda()
    model = VAE().cuda()
    out = model(img)
    print(out[0].shape)

