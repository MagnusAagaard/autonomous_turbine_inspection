from torch import nn
    
class ConvEncoderDecoder(nn.Module):
    def __init__(self, inp_dim):
        super(ConvEncoderDecoder, self).__init__()
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
        self.out_conv = nn.Conv2d(64, inp_dim, kernel_size=1, stride=1, padding=0)
        
        
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
    model = ConvEncoderDecoder(10)
    #if cuda.is_available():
    #    device = 'cuda:0'
    #else:
    #    device = 'cpu'
    #model.to(device)

if __name__ == "__main__":
    main()