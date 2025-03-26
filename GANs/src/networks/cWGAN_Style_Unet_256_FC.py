import torch
import torch.nn as nn
import torch.nn.functional as F

"""
In their 2015 paper, He et. al. demonstrated that deep networks (e.g. a 22-layer CNN) would converge much earlier if the following input weight initialization strategy is employed:

1. Create a tensor with the dimensions appropriate for a weight matrix at a given layer, and populate it with numbers randomly chosen from a standard normal distribution.
2. Multiply each randomly chosen number by √2/√n where n is the number of incoming connections coming into a given layer from the previous layer’s output (also known as the “fan-in”).
3. Bias tensors are initialized to zero.
"""

class WSLinear(nn.Module):
    def __init__(
        self, in_features, out_features,
    ):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features)**0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        # initialize linear layer
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias
    
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()

        self.mapping = nn.Sequential(
            PixelNorm(),
            WSLinear(z_dim, w_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            WSLinear(w_dim, w_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            WSLinear(w_dim, w_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            WSLinear(w_dim, w_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            WSLinear(w_dim, w_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            WSLinear(w_dim, w_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            WSLinear(w_dim, w_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            WSLinear(w_dim, w_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.mapping(x)
    
class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN)
    """
    def __init__(self, channels, w_dim, double=False):
        super().__init__()
        if double:
            w_dim = w_dim*2
        self.instance_norm = nn.InstanceNorm3d(channels)
        self.style_scale = WSLinear(w_dim, channels)
        self.style_bias = WSLinear(w_dim, channels)

    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return style_scale * x + style_bias
    
class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)   
    
class WSConv3d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        super(WSConv3d, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1, 1)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv1 = WSConv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = WSConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x
    
    
class Critic(nn.Module):
    def __init__(self, in_channels=1024, img_channels=2):
        super(Critic, self).__init__()
        
        self.leaky = nn.LeakyReLU(0.2)
        self.avg_pool = nn.AvgPool3d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

       
        self.block_1 = ConvBlock(in_channels=img_channels, out_channels=in_channels//64, kernel_size=3, stride=1, padding=1)
        self.block_2 = ConvBlock(in_channels=in_channels//64, out_channels=in_channels//32, kernel_size=3, stride=1, padding=1)
        self.block_3 = ConvBlock(in_channels=in_channels//32, out_channels=in_channels//16, kernel_size=3, stride=1, padding=1)  
        self.block_4 = ConvBlock(in_channels=in_channels//16, out_channels=in_channels//8, kernel_size=3, stride=1, padding=1)
        self.block_5 = ConvBlock(in_channels=in_channels//8, out_channels=in_channels//4, kernel_size=3, stride=1, padding=1)
        self.block_6 = ConvBlock(in_channels=in_channels//4, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1)
        

        self.final_block = nn.Sequential(
            WSConv3d(in_channels//2, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv3d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv3d(
                in_channels, 1, kernel_size=1, padding=0, stride=1
            ),  # we use this instead of linear layer
        )

    def forward(self, x):
        out = self.avg_pool(self.block_1(x))
        out = self.avg_pool(self.block_2(out))
        out = self.avg_pool(self.block_3(out))
        out = self.avg_pool(self.block_4(out))
        out = self.avg_pool(self.block_5(out))
        out = self.avg_pool(self.block_6(out))
        #out = self.avg_pool(self.block_8(out))

        return self.final_block(out).view(out.shape[0], -1)
    
class InjectNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]), device=x.device)
        return x + self.weight * noise
    
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, w_dim, double, last=False):
        super(GenBlock, self).__init__()      
        if not last:
            self.conv1 = WSConv3d(in_channels=in_channels+in_channels//2, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv1 = WSConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = WSConv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.inject_noise1 = InjectNoise(out_channels)
        self.inject_noise2 = InjectNoise(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim, double=double)
        self.adain2 = AdaIN(out_channels, w_dim, double=double)

    def forward(self, x, w):
        """
        Step 1 -> Does conv1 using x as input (x comes from the previous layer)
        Step 2 -> Injects noise
        Step 3 -> Leaky relu
        Step 4 -> Adain with the output of leaky relu and w (style noise)
        Repeat
        """
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)
        return x
    
class Decoder(nn.Module):
    def __init__(self, z_dim, w_dim, in_channels, img_channels=1, skip_latent=True, tahn_act=False):
        super(Decoder, self).__init__()
        self.skip_latent = skip_latent
        if skip_latent:
            double = True
        else:
            double = False

        self.tahn_act = tahn_act

        self.const = nn.Parameter(torch.ones((1, in_channels, 4, 4, 4)))
        self.bias = nn.Parameter(torch.ones(in_channels))
        
        self.map = MappingNetwork(z_dim, w_dim)
        
        self.initial_adain1 = AdaIN(in_channels, w_dim, double=double)
        self.initial_adain2 = AdaIN(in_channels, w_dim, double=double)
        self.initial_noise1 = InjectNoise(in_channels)
        self.initial_noise2 = InjectNoise(in_channels)
                
        self.initial_conv = WSConv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

        self.genBlock_2 = GenBlock(in_channels=in_channels//1, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1, w_dim=w_dim, double=double) #8**3
        self.genBlock_3 = GenBlock(in_channels=in_channels//2, out_channels=in_channels//4, kernel_size=3, stride=1, padding=1, w_dim=w_dim, double=double) #16**3
        self.genBlock_4 = GenBlock(in_channels=in_channels//4, out_channels=in_channels//8, kernel_size=3, stride=1, padding=1, w_dim=w_dim, double=double) #32**3
        self.genBlock_5 = GenBlock(in_channels=in_channels//8, out_channels=in_channels//16, kernel_size=3, stride=1, padding=1, w_dim=w_dim, double=double) #64**3
        self.genBlock_6 = GenBlock(in_channels=in_channels//16, out_channels=in_channels//32, kernel_size=3, stride=1, padding=1, w_dim=w_dim, double=double) #128**3
        self.genBlock_7 = GenBlock(in_channels=in_channels//32, out_channels=in_channels//64, kernel_size=3, stride=1, padding=1, w_dim=w_dim, double=double, last=False) #(256,256,256)
        #self.genBlock_8 = GenBlock(in_channels=in_channels//64, out_channels=in_channels//128, kernel_size=3, stride=1, padding=1, w_dim=w_dim, double=double, last=True) #(256,512,128)
        
        self.final_layer =  WSConv3d(in_channels=in_channels//64, out_channels=img_channels, kernel_size=1, stride=1, padding=0)
            
        self.Tanh = nn.Tanh()

    def forward(self, noise, encoder_list):
        skip_0, skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, latent_tensor = encoder_list
        # The skip before the latent_tensor is never used
    
        latent_n_noise = torch.cat([latent_tensor, noise], dim = 1)
        #######
        # Mapping noise
        w = self.map(latent_n_noise) # Map noise to create w
        if self.skip_latent: # False
            w_cat_latent = torch.cat([latent_tensor, w], dim = 1).to(w.device)
        else:
            w_cat_latent = w
        #######
        #######
        # Preparing input
        # Takes the constant input of shape b, in_channels, 4, 4, 4 
        batch_size = noise.shape[0]
        x = self.const.expand(batch_size, -1, -1, -1, -1)
        x = x + self.bias.view(1, -1, 1, 1, 1)
        #######
        # First block
        # Adds noise, and inputs into the adain with the mapped noise
        x = self.initial_adain1(self.leaky(self.initial_noise1(x)), w_cat_latent)
        # Skip before conv
        # Not doing skip connection in the first conv

        #x = torch.cat((x, skip_7), dim=1)
        # convolution
        x = self.initial_conv(x)
        # Add more noise, leaky relu, and input into the adain
        out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w_cat_latent)
        #######
        #######
        # Middle blocks
        # Upsample and  generator block
        upscaled = F.interpolate(out, scale_factor=2, mode="trilinear")  #8x8x8   
        upscaled = torch.cat((upscaled, skip_5), dim=1)
        out = self.genBlock_2(upscaled, w_cat_latent)
        #
        upscaled = F.interpolate(out, scale_factor=2, mode="trilinear") #16x16x16
        upscaled = torch.cat((upscaled, skip_4), dim=1)
        out = self.genBlock_3(upscaled, w_cat_latent)
        #
        upscaled = F.interpolate(out, scale_factor=2, mode="trilinear") #32x32x32
        upscaled = torch.cat((upscaled, skip_3), dim=1)
        out = self.genBlock_4(upscaled, w_cat_latent)
        #
        upscaled = F.interpolate(out, scale_factor=2, mode="trilinear") #64x64x64
        upscaled = torch.cat((upscaled, skip_2), dim=1)
        out = self.genBlock_5(upscaled, w_cat_latent)
        #
        upscaled = F.interpolate(out, scale_factor=2, mode="trilinear") #128x128x128
        upscaled = torch.cat((upscaled, skip_1), dim=1)
        out = self.genBlock_6(upscaled, w_cat_latent)
        #
        upscaled = F.interpolate(out, scale_factor=2, mode="trilinear") #256x256x256
        upscaled = torch.cat((upscaled, skip_0), dim=1)
        out = self.genBlock_7(upscaled, w_cat_latent)
        #
        #upscaled = F.interpolate(out, scale_factor=(1,2,1), mode="trilinear")
        #out = self.genBlock_8(upscaled, w_cat_latent)
        #######
        #######
        # Last layer
        final_out = self.final_layer(out)
        if self.tahn_act: 
            final_out = self.Tanh(final_out)
        return final_out

class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncBlock, self).__init__()

        self.conv1 = WSConv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=1024, latent_dim=512, img_channels=2):
        super(Encoder, self).__init__()
        
        self.leaky = nn.LeakyReLU(0.2)
        self.avg_pool = nn.AvgPool3d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        #self.block_1 = EncBlock(in_channels=img_channels, out_channels=in_channels//64, kernel_size=3, stride=1, padding=1)
        self.in_block = EncBlock(in_channels=img_channels, out_channels=in_channels//64, kernel_size=3, stride=1, padding=1)
        self.block_1 = EncBlock(in_channels=in_channels//64, out_channels=in_channels//32, kernel_size=3, stride=1, padding=1)
        self.block_2 = EncBlock(in_channels=in_channels//32, out_channels=in_channels//16, kernel_size=3, stride=1, padding=1)
        self.block_3 = EncBlock(in_channels=in_channels//16, out_channels=in_channels//8, kernel_size=3, stride=1, padding=1)  
        self.block_4 = EncBlock(in_channels=in_channels//8, out_channels=in_channels//4, kernel_size=3, stride=1, padding=1)
        self.block_5 = EncBlock(in_channels=in_channels//4, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1)
        self.block_6 = EncBlock(in_channels=in_channels//2, out_channels=in_channels//1, kernel_size=3, stride=1, padding=1)
        
        
        self.final_block = nn.Sequential(
            WSConv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv3d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv3d(
                in_channels, latent_dim, kernel_size=1, padding=0, stride=1
            ),  # we use this instead of linear layer
        )

    def forward(self, x):
        out_0 = self.in_block(x)
        out_1 = self.avg_pool(self.block_1(out_0))
        out_2 = self.avg_pool(self.block_2(out_1))
        out_3 = self.avg_pool(self.block_3(out_2))
        out_4 = self.avg_pool(self.block_4(out_3))
        out_5 = self.avg_pool(self.block_5(out_4))
        out_6 = self.avg_pool(self.block_6(out_5))
       
        out_7 = self.final_block(out_6).view(out_6.shape[0], -1)

        return [out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7]
    
class Generator(nn.Module):
    def __init__(self, in_channels, latent_dim, IN_CHANNEL_G,OUT_CHANNEL_G, z_dim, w_dim, skip_latent, tahn_act):
        super(Generator, self).__init__()    

        self.z_dim = z_dim

        self.enc = Encoder(in_channels=in_channels, latent_dim=latent_dim, img_channels=IN_CHANNEL_G)
        self.dec = Decoder(z_dim=z_dim*2, w_dim=w_dim, in_channels=in_channels, img_channels=OUT_CHANNEL_G, skip_latent=skip_latent, tahn_act=tahn_act)
    
    def forward(self, x):
        encoder_list = self.enc(x)
        batch_size = x.shape[0]
        noise = torch.randn(batch_size, self.z_dim).to(x.device)
        #latent_n_noise = torch.cat([latent_tensor, noise], dim = 1).to(latent_tensor.device)
        fake_image = self.dec(noise=noise, encoder_list=encoder_list)
        
        return fake_image
