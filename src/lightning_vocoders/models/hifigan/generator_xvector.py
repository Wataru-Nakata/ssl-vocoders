from .hifigan import weight_norm,Conv1d,ResBlock1,ResBlock2,ConvTranspose1d,init_weights,remove_weight_norm, LRELU_SLOPE
import torch.nn.functional as F
import torch.nn as nn
import torch

class FiLMLayer(nn.Module):
    def __init__(self,input_channels,intermediate_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels,intermediate_channels,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv1d(intermediate_channels, input_channels,kernel_size=3,stride=1,padding=1)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self,a:torch.Tensor,b:torch.Tensor):
        batch_size, K, D = a.size()
        Q  = b.size(1)
        a = a.transpose(1,2)
        output = self.conv2((self.leaky_relu(self.conv1(a)).transpose(1,2) + b).transpose(1,2))
        output = output.permute(0,2,1)
        assert output.size() == (batch_size,K,D)
        return output
class GeneratorWithXvector(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(h.num_input_channels, h.upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2**i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()

        self.feature_xvector_film = FiLMLayer(h.num_input_channels,h.xvector_dim)
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, feature,xvector):
        x = self.feature_xvector_film(feature,xvector.unsqueeze(1))
        x = self.conv_pre(x.transpose(1, 2))
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
