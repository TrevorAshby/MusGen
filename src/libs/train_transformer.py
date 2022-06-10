#import sys
#sys.path.insert(0, '../src/libs')

# our file imports
import defusedxml
from dataset import *
from music_transformer import *
from song_classification import *
from transformer_training_helpers import *

import argparse
from torch.autograd import Variable

# musicautobot
from musicautobot.config import *
from musicautobot.numpy_encode import *
from musicautobot.utils.midifile import *
from musicautobot.music_transformer import *
from musicautobot.utils.file_processing import process_all

def run_epoch(data_iter, model, loss_compute, print_every):#, bad_idxs):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    i = 0
    collection_of_losses = []
    #print(len(data_iter))
    for _, batch in enumerate(data_iter):
        #out, w2 = model.forward(batch.src, batch.trg, 
        #                    batch.src_mask, batch.trg_mask)
        #print('ITERATION: ', _)
        #try:
        batch = batch.cuda()
        batch = batch.long()
        batch = batch.squeeze(0)
        PAD = 0
        trg = batch[:,:-1]
        trg_y = batch[:, 1:]
        src_mask = (batch != PAD).unsqueeze(-2)
        trg_mask = (trg != PAD).unsqueeze(-2)
        trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).type_as(trg_mask.data))
        ntokens = (trg_y != PAD).data.sum()
        
        out, w2 = model.forward(batch, trg, src_mask, trg_mask)
        #print("HEREEE")
        loss = loss_compute(out, trg_y, ntokens.item())

        total_loss += loss
        # total_tokens += ntokens
        tokens += ntokens
        if i % print_every == 1:
            collection_of_losses.append(torch.div(loss,ntokens).cpu())
            elapsed = time.time() - start
            print("\tepoch Step: %d/%d Loss: %f Tokens per Sec: %f" %
                    (i, len(data_iter), torch.div(loss,ntokens), tokens / elapsed))
            start = time.time()
            tokens = 0
        #except:
        #    continue
        total_tokens += ntokens
        i += 1
    return (total_loss / total_tokens), collection_of_losses

# My version of Greedy Decoding
def train_my_model(epochs, d_model_in, num_layers, lr_in, print_every):
  #---------- TRAIN -----------
  #V = len(all_data.vocab) # 312
  V = 312
  criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
  myModel = make_model(V, V, N=num_layers, d_model=d_model_in)
  myModel.cuda() # uncomment for GPU
  model_opt = NoamOpt(myModel.src_embed[0].d_model, 1, 400, torch.optim.Adam(myModel.parameters(), lr=lr_in, betas=(0.9, 0.98), eps=1e-9))
  #the_train_data = data_gen(all_data.train_ds)
  the_train_data = dataloader
  collection_of_losses = np.array([])
  for epoch in range(epochs):
    print("EPOCH: ", epoch)
    myModel.train()
    totalLoss_div_totalTokens, collection_of_losses_inp = run_epoch(the_train_data, myModel, SimpleLossCompute(myModel.generator, criterion, model_opt), print_every)#, bad_idxs)
    collection_of_losses = np.concatenate([collection_of_losses, collection_of_losses_inp])
    #model.eval()
    #print(run_epoch(the_train_data, model, SimpleLossCompute(model.generator, criterion, None), bad_idxs))
  return myModel, collection_of_losses

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="A model training script.")
    arg_parser.add_argument("--numpy_path", help="A path to the .npy training files to be used. Default=\'../numpy_path\'", default="../numpy_path")
    arg_parser.add_argument("--save_path", help="A path where the model should be saved. Default=\'../models\'", default="../models")
    arg_parser.add_argument("--model_name", help="The name that the model should be saved under. Default=\'lr_trainer_epochs\'", default="chumbucket")
    arg_parser.add_argument("--epochs", help="The number of epochs that the model should be trained for. Default=10", default=10, type=int)
    arg_parser.add_argument("--dset_dim", help="The subsection length to split dataset songs into. i.e. context window. Default=150", default=150, type=int)
    arg_parser.add_argument("--d_model", help="The dimention of the model. Default=128", default=128, type=int)
    arg_parser.add_argument("--num_layers", help="The number of layers within the Transformer. Default=2", default=2, type=int)
    arg_parser.add_argument("--trainer", help="The name of the person who trained the model. Default=MusGen", default="MusGen")
    arg_parser.add_argument("--lr", help="The learning rate to be used in training. Default=0.6", default=0.6, type=float)
    arg_parser.add_argument("--print_every", help="The epoch step number that when reached, prints training details out to console. Default=200", default=200, type=int)

    args = arg_parser.parse_args()

    numpy_data_path = args.numpy_path
    model_save_path = args.save_path
    model_name = args.model_name
    epochs = args.epochs
    dset_dim = args.dset_dim
    d_model = args.d_model
    num_layers = args.num_layers
    trainer = args.trainer
    lr = args.lr
    print_every = args.print_every

    if model_name == "chumbucket":
        model_name = f"{lr}_{trainer}_{epochs}_LM"

    myDs = MidiDataset(numpy_data_path, dset_dim, 'N/A', '.npy')
    dataloader = DataLoader(myDs, 1, shuffle=True) # Batch set to 1 so that each song is treated as a batch.

    myModel, losses = train_my_model(epochs, d_model, num_layers, lr, print_every)

    torch.save(myModel, model_save_path + '/' + model_name + '.pt')
    config_file = open(model_save_path + '/' + model_name + '_config.txt', 'w')
    config_file.write(f"Model Location: {model_save_path}/{model_name}.pt\n")
    config_file.write(f"Model params:\n\tEpochs:{epochs}\n\tDset_dim:{dset_dim}\n\tD_model:{d_model}\n\tNum_layers:{num_layers}\n\tLearning Rate:{lr}\n\tTrained by:{trainer}")

    
