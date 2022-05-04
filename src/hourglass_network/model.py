import torch
from torch import nn

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()
        conv_mod = nn.Conv2d(int(in_channels), 
                             int(n_filters), 
                             kernel_size=k_size, 
                             padding=padding, 
                             stride=stride, 
                             bias=bias, 
                             dilation=dilation)
        self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True))
    
    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class ConvDown1(nn.Module):
    def __init__(self, in_size, out_size):
        super(ConvDown1, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
        
    def forward(self, x):
        x = self.conv1(x)
        unpooled_shape = x.size()
        x, indices = self.maxpool(x)
        return x, indices, unpooled_shape
    
class ConvUp1(nn.Module):
    def __init__(self, in_size, out_size):
        super(ConvUp1, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        
    def forward(self, x, indices, output_shape):
        x = self.unpool(input=x, indices=indices, output_size=output_shape)
        x = self.conv1(x)
        return x

class ConvEncoderDecoderV3(nn.Module):
    def __init__(self, inp_dim):
        super(ConvEncoderDecoderV3, self).__init__()
        self.inp_dim = inp_dim
        # Common functions
        self.sigmoid = nn.Sigmoid()
        # Encoder
        self.down1 = ConvDown1(self.inp_dim, 64)
        self.down2 = ConvDown1(64, 128)
        self.down3 = ConvDown1(128, 256)
        self.down4 = ConvDown1(256, 512)
        self.down5 = ConvDown1(512, 512)
        # Decoder
        self.up5 = ConvUp1(512, 512)
        self.up4 = ConvUp1(512, 256)
        self.up3 = ConvUp1(256, 128)
        self.up2 = ConvUp1(128, 64)
        self.up1 = ConvUp1(64, self.inp_dim)
        
    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)
        out = self.sigmoid(up1)

        return out

class ConvEncoderDecoderV2(nn.Module):
    def __init__(self, inp_dim):
        super(ConvEncoderDecoderV2, self).__init__()
        self.inp_dim = inp_dim
        # Common functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Encoder
        self.conv1 = nn.Conv2d(inp_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up1_bn = nn.BatchNorm2d(256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2_bn = nn.BatchNorm2d(128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up3_bn = nn.BatchNorm2d(64)
        
        # Output
        self.out_conv = nn.ConvTranspose2d(64, inp_dim, kernel_size=2, stride=2)
        
        
    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # Encoder
        # Conv1
        #print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        #print(x.size())
        # Conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        #print(x.size())
        # Conv3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        #print(x.size())
        # Conv4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)
        #print(x.size())
        # Decoder
        # Up1
        x = self.up1(x)
        #print(x.size())
        x = self.up1_bn(x)
        x = self.relu(x)
        # Up2
        x = self.up2(x)
        #print(x.size())
        x = self.up2_bn(x)
        x = self.relu(x)
        # Up3
        x = self.up3(x)
        #print(x.size())
        x = self.up3_bn(x)
        x = self.relu(x)

        x = self.out_conv(x)
        x = self.sigmoid(x)
        #print(x.size())
        
        return x
    
class ConvEncoderDecoderCor(nn.Module):
    def __init__(self, inp_dim, extra_layer=False):
        super(ConvEncoderDecoderCor, self).__init__()
        self.extra_layer = extra_layer
        self.inp_dim = inp_dim
        # Common functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Encoder
        self.conv1 = nn.Conv2d(inp_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        if self.extra_layer:
            self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.pool5 = nn.MaxPool2d(2, 2)
        
        # Decoder
        if self.extra_layer:
            self.up1e = nn.Upsample(scale_factor=2, mode='nearest')
            self.up1e_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.up1e_bn = nn.BatchNorm2d(512)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1_conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.up1_bn = nn.BatchNorm2d(256)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.up2_bn = nn.BatchNorm2d(128)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3_conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.up3_bn = nn.BatchNorm2d(64)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4_conv = nn.Conv2d(64, 7, kernel_size=3, stride=1, padding=1)
        self.up4_bn = nn.BatchNorm2d(7)
        
        # Output
        #self.out_conv = nn.Conv2d(64, inp_dim, kernel_size=1, stride=1, padding=0)
        self.out_conv = nn.Conv2d(7, 7, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # Encoder
        #print(x.size())
        # Conv1
        x = self.conv1(x)
        #print(x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        #print(x.size())
        # Conv2
        x = self.conv2(x)
        #print(x.size())
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        #print(x.size())
        # Conv3
        x = self.conv3(x)
        #print(x.size())
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        #print(x.size())
        # Conv4
        x = self.conv4(x)
        #print(x.size())
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)
        if self.extra_layer:
            # Conv5
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.relu(x)
            x = self.pool5(x)
            
            x = self.up1e(x)
            x = self.up1e_conv(x)
            x = self.up1e_bn(x)
            x = self.relu(x)
        #print(x.size())
        # Decoder
        # Up1
        x = self.up1(x)
        #print(x.size())
        x = self.up1_conv(x)
        #print(x.size())
        x = self.up1_bn(x)
        x = self.relu(x)
        # Up2
        x = self.up2(x)
        #print(x.size())
        x = self.up2_conv(x)
        #print(x.size())
        x = self.up2_bn(x)
        x = self.relu(x)
        # Up3
        x = self.up3(x)
        #print(x.size())
        x = self.up3_conv(x)
        #print(x.size())
        x = self.up3_bn(x)
        x = self.relu(x)
        # Up4
        x = self.up4(x)
        #print(x.size())
        x = self.up4_conv(x)
        #print(x.size())
        x = self.up4_bn(x)
        x = self.relu(x)
        # Output layer
        x = self.out_conv(x)
        #print(x.size())
        x= self.sigmoid(x)
        #print(x.size())
        
        return x
    
class ConvEncoderDecoder(nn.Module):
    def __init__(self, inp_dim, extra_layer=False, old_version=False):
        super(ConvEncoderDecoder, self).__init__()
        self.extra_layer = extra_layer
        self.old_version = old_version
        self.inp_dim = inp_dim
        # Common functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Encoder
        self.conv1 = nn.Conv2d(inp_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        if self.extra_layer:
            self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.pool5 = nn.MaxPool2d(2, 2)
        
        # Decoder
        if self.extra_layer:
            self.up1e = nn.Upsample(scale_factor=2, mode='nearest')
            self.up1e_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.up1e_bn = nn.BatchNorm2d(512)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.up1_bn = nn.BatchNorm2d(512)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.up2_bn = nn.BatchNorm2d(256)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.up3_bn = nn.BatchNorm2d(128)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4_conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.up4_bn = nn.BatchNorm2d(64)
        
        # Output
        if self.old_version:
            self.out_conv = nn.Conv2d(64, inp_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.out_conv = nn.Conv2d(64, 7, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # Encoder
        #print(x.size())
        # Conv1
        x = self.conv1(x)
        #print(x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        #print(x.size())
        # Conv2
        x = self.conv2(x)
        #print(x.size())
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        #print(x.size())
        # Conv3
        x = self.conv3(x)
        #print(x.size())
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        #print(x.size())
        # Conv4
        x = self.conv4(x)
        #print(x.size())
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)
        if self.extra_layer:
            # Conv5
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.relu(x)
            x = self.pool5(x)
            
            x = self.up1e(x)
            x = self.up1e_conv(x)
            x = self.up1e_bn(x)
            x = self.relu(x)
        #print(x.size())
        # Decoder
        # Up1
        x = self.up1(x)
        #print(x.size())
        x = self.up1_conv(x)
        #print(x.size())
        x = self.up1_bn(x)
        x = self.relu(x)
        # Up2
        x = self.up2(x)
        #print(x.size())
        x = self.up2_conv(x)
        #print(x.size())
        x = self.up2_bn(x)
        x = self.relu(x)
        # Up3
        x = self.up3(x)
        #print(x.size())
        x = self.up3_conv(x)
        #print(x.size())
        x = self.up3_bn(x)
        x = self.relu(x)
        # Up4
        x = self.up4(x)
        #print(x.size())
        x = self.up4_conv(x)
        #print(x.size())
        x = self.up4_bn(x)
        x = self.relu(x)
        # Output layer
        x = self.out_conv(x)
        #print(x.size())
        x= self.sigmoid(x)
        #print(x.size())
        
        return x

def main():
    model = ConvEncoderDecoderCor(10, extra_layer=False)
    input = torch.randn(1,10,256,256, requires_grad=True)
    print(input.shape)
    out = model(input)
    print(out.shape)
    #if cuda.is_available():
    #    device = 'cuda:0'
    #else:
    #    device = 'cpu'
    #model.to(device)

if __name__ == "__main__":
    main()