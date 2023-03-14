import argparse
from model import CNNDecoder,CNNEncoder
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch

from torch.autograd import Variable
import config_file as cfg
from data import Dataset3D
from torch.utils.data import DataLoader
import calculator



def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

def train(epoch):
    #set models to training mode
    encoder.train()
    decoder.train()
    running_loss = 0.
    last_loss = 0.
  
    # getting the training set
    for i, data in enumerate(training_loader):
        ctp_img,tmax,cbv,cbf = data
    # clearing the Gradients of the model parameters
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        # prediction for this batch
        concat_output = []
        skip = []
        for t in range(ctp_img.shape[2]):
            encoder_input = ctp_img[t,:,:]
            encoder_output,down = encoder(encoder_input)
            concat_output.append(encoder_output)
            skip.append(down)
        connections = []
        for i in range(len(skip[0])):
            subtract = []
            for j in range(ctp_img.shape[2]):
                subtract = skip[j][i] if j == 0 else torch.abs(subtract-skip[i][j])
            connections.append(subtract)
        
        kt = decoder(concat_output,connections)

        tmax_pred = calculator.calculate_tmax(kt)
        cbv_pred = calculator.calculate_cbv(kt)
        cbf_pred = calculator.calculate_cbf(kt)

        loss = loss_fn(tmax_pred,tmax)+loss_fn(cbv_pred,cbv)+loss_fn(cbf_pred,cbf)
        loss.backward()
        #?: use one optimizer or two
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    print("epoch ",epoch,". loss: ",last_loss)
    return last_loss


if __name__ == "__main__":
    args = parse_args()
    encoder_save_path = cfg.encoder_save_path
    decoder_save_path = cfg.decoder_save_path
        
    #data loader
    ctp_image_path = cfg.ctp_image_path
    tmax_image_path =cfg.tmax_image_path
    cbv_image_path =cfg.cbv_image_path
    cbf_image_path =cfg.cbf_image_path


    training_data = Dataset3D(ctp_image_path,tmax_image_path,cbv_image_path,cbf_image_path)
    training_loader = DataLoader(training_data, batch_size=1, shuffle=True)

    encoder = CNNEncoder()
    decoder = CNNDecoder()
    encoder_optimizer = Adam(encoder.parameters(), lr=0.07)
    decoder_optimizer = Adam(decoder.parameters(), lr=0.07)

    loss_fn = CrossEntropyLoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        loss_fn = loss_fn.cuda()

    print("===================encoder=======================")
    print("structure:")
    print(encoder)
    print("optimizer:")
    print(encoder_optimizer)

    print("\n===================decoder========================")
    print(decoder)

    train(1)
    torch.save(encoder.state_dict(), encoder_save_path)
    torch.save(decoder.state_dict(), decoder_save_path)
