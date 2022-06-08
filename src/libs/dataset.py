from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
from music_transformer import subsequent_mask
from torch.autograd import Variable

# musicautobot
from musicautobot.numpy_encode import *
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.utils.midifile import *
from musicautobot.utils.file_processing import process_all

# step 3
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

# step 2
def reshape_my_data(input_data):
  og_len = len(input_data) # song length
  LEN_MINIBATCH = 150
  remainder = og_len % LEN_MINIBATCH
  final_beg = input_data[:og_len-remainder] # with beginning
  final_end = input_data[remainder:] # with end
  return final_beg.reshape([int(len(input_data)/LEN_MINIBATCH), LEN_MINIBATCH]), final_end.reshape([int(len(input_data)/LEN_MINIBATCH), LEN_MINIBATCH])

# step 1
def data_gen(input_data):
  final_batch = []
  for i in range(len(input_data)):
    data = input_data[i][0].to_tensor().cpu() # 300 removed cuda here
    data_beg, data_end = reshape_my_data(data)
    #data[:, 0] = 1
    src = Variable(data_beg, requires_grad=False)
    tgt = Variable(data_beg, requires_grad=False)

    src2 = Variable(data_end, requires_grad=False)
    tgt2 = Variable(data_end, requires_grad=False)
    #final_batch.append(Batch(src, tgt, 0)) # CPU
    #final_batch.append(Batch(src2, tgt, 0)) # CPU
    final_batch.append(Batch(src.cuda(), tgt.cuda(), 0)) # GPU
    final_batch.append(Batch(src2.cuda(), tgt.cuda(), 0)) # GPU
  return np.array(final_batch)

def create_ovlap_tensor(song_tensor, dset_dim):
    remainder = song_tensor.shape[0] % dset_dim
    num_subsects = int(song_tensor.shape[0] / dset_dim)

    ovlap_tensor = torch.zeros((num_subsects, dset_dim))
    # print('remainder: ', remainder)
    # print('subsects: ', num_subsects)
    # print('overlap: ', overlap_len)
    
    #print(ovlap_tensor.shape)

    for subsect in range(num_subsects): # 0 -> 148
        if subsect > int(num_subsects / 2):
            beg = (dset_dim * subsect) - (remainder)
            end = (dset_dim * (subsect + 1)) - (remainder)
        else:
            beg = (dset_dim * subsect)
            end = (dset_dim * (subsect + 1))
        # print('1: ', ovlap_tensor[subsect].shape)
        # print('2: ', len(song_tensor[beg:end]))
        ovlap_tensor[subsect] = song_tensor[beg:end]
    return ovlap_tensor

class MidiDataset(Dataset):
    def __init__(self, numpy_path, dset_dim, device, extension):
        # list for holding all filenames
        self.filenames = []
        self.folder_path = numpy_path
        # read all file names from path
        self.filenames = os.listdir(numpy_path)
        self.dset_dim = dset_dim
        self.device = device
        self.extension = extension
        
    def __getitem__(self, index):
        song_file = self.filenames[index]
        # load MusicItem from .npy extension
        if self.extension == '.npy':
            song_arr = np.load(self.folder_path + '/' + song_file)
            song_MuIt = MusicItem.from_npenc(song_arr, MusicVocab.create())
        # load MusicItem from .mid extension
        elif self.extension == '.mid':
            song_MuIt = MusicItem.from_file(self.folder_path + '/' + song_file, MusicVocab.create())
        # if self.device == 'cpu':
            # this tensor is going to be too long, need to reshape
        song_tensor = song_MuIt.to_tensor().cpu()
        ovlap_tensor = create_ovlap_tensor(song_tensor, self.dset_dim)
            # print('ovlap_tensor.shape: ', ovlap_tensor.shape)
        # elif self.device == 'gpu':
        #     # this tensor is going to be too long, need to reshape
        #     song_tensor = song_MuIt.to_tensor()
        #     ovlap_tensor = create_ovlap_tensor(song_tensor, self.dset_dim)
        #     del song_tensor
        #     ovlap_tensor = ovlap_tensor.cuda()
            # print('ovlap_tensor.shape: ', ovlap_tensor.shape)

        # return the overlapped tensor (subsectioned song)
        return ovlap_tensor
    def __len__(self):
        return len(self.filenames)
        #pass

# if __name__ == '__main__':
#     start = time.time()
#     myDs = MidiDataset('./src/numpy_path', 500, 'cpu', '.npy')
#     print(myDs[0].shape)
#     end = time.time()
#     print('np elapsed: ', end - start)


    # start = time.time()
    # myDs2 = MidiDataset('./src/mid_data_collections/mid_0_to_10000', 120, 'cpu', '.mid')
    # print(myDs2[0])
    # end = time.time()
    # print('Mid elapsed: ', end - start)
    # mv = MusicVocab.create()
    # print(mv.stoi)