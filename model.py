from torch.nn import Module, Conv2d, Sequential, ReLU, ConvTranspose2d
from torch import cat

#---------------------------
#        CNN Encoder
#---------------------------
class CNNEncoder(Module):
    def __init__(self):
        super().__init__()
        self.level1_layers = Sequential(
            Conv2d(1,16, (3,3),stride=(1,1),padding='same'),
            ReLU(inplace=True),
            Conv2d(16, 16, (3,3),stride=(1,1),padding='same'),
            ReLU(inplace=True)
        )
        self.level2_layers = Sequential(
            Conv2d(in_channels=16, out_channels=1, kernel_size=(2,2),stride=(2,2)),
            Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True)
        )
        self.level3_layers = Sequential(
            Conv2d(in_channels=32, out_channels=1, kernel_size=(2,2),stride=(2,2)),
            Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True)
        )
        self.level4_layers = Sequential(
            Conv2d(in_channels=64, out_channels=1, kernel_size=(2,2),stride=(2,2)),
            Conv2d(in_channels=1, out_channels=128, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True)
        )
        self.level5_layers = Sequential(
            Conv2d(in_channels=128, out_channels=1, kernel_size=(2,2),stride=(2,2)),
            Conv2d(in_channels=1, out_channels=256, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True)
        )
        self.output_layer = Conv2d(in_channels=256, out_channels=1, kernel_size=(2,2),stride=(2,2))
        

    def forward(self,input):
        level1_output = self.level1_layers(input)
        level2_output = self.level2_layers(level1_output)
        level3_output = self.level3_layers(level2_output)
        level4_output = self.level4_layers(level3_output)
        level5_output = self.level5_layers(level4_output)
        output = self.output_layer(level5_output)
        down = [level1_output,level2_output,level3_output,level4_output,level5_output]
        return output,down

class CNNDecoder(Module):
    def __init__(self):
        super().__init__()
        self.input_layer=Sequential(
            ConvTranspose2d(in_channels = 1, out_channels=1, kernel_size=(2,2), stride=(2,2))
        )
        self.level5_layer = Sequential(
            Conv2d(in_channels=257, out_channels=256, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels = 256, out_channels=1, kernel_size=(2,2), stride=(2,2))
        )
        self.level4_layer = Sequential(
            Conv2d(in_channels=129, out_channels=128, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels = 128, out_channels=1, kernel_size=(2,2), stride=(2,2))
        )
        self.level3_layer = Sequential(
            Conv2d(in_channels=65, out_channels=64, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels = 64, out_channels=1, kernel_size=(2,2), stride=(2,2))
        )
        self.level2_layer = Sequential(
            Conv2d(in_channels=33, out_channels=32, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels = 32, out_channels=1, kernel_size=(2,2), stride=(2,2))
        )
        self.level1_layer = Sequential(
            Conv2d(in_channels=17, out_channels=16, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros'),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels = 16, out_channels=1, kernel_size=(2,2), stride=(2,2))
        )
        self.output_layer = Sequential(
            Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3),stride=(1,1),padding='same',padding_mode='zeros')
        )

    def forward(self,x,down):
        input = cat((self.input_layer(x),down[4]))
        #print("size of input layer output: ", self.input_layer(x).size())
        #print("size of down[4]: ",down[4].size())
        #print("size of input: ",input.size())
        level5_output = cat((self.level5_layer(input),down[3]))
        level4_output = cat((self.level4_layer(level5_output),down[2]))
        level3_output = cat((self.level3_layer(level4_output),down[1]))
        level2_output = cat((self.level2_layer(level3_output),down[0]))
        level1_output = self.level1_layer(level2_output)
        output = self.output_layer(level1_output)
        return output


