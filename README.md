# MusGen
A repository of the Music Transformer and Neural Transposer designed to be able to generate music in a relative major/minor depending on input.

## Script Usage
- <b>train_transformer.py</b>: <i> Located in /src/libs/.</i>
  - Arguments:
    - <i>numpy_path: A path to the .npy training files to be used. Default=../numpy_path</i>
    - <i>save_path: A path where the model should be saved. Default=../models</i>
    - <i>model_name: The name that the model should be saved under. Default={lr}_{trainer}_{epochs}</i>
    - <i>epochs: The number of epochs that the model should be trained for. Default=10</i>
    - <i>dset_dim: The subsection length to split dataset songs into. i.e. context window. Default=150</i>
    - <i>d_model: The dimention of the model. Default=128</i>
    - <i>num_layers: The number of layers within the Transformer. Default=2</i>
    - <i>trainer: The name of the person who trained the model. Default=MusGen</i>
    - <i>lr: The learning rate to be used in training. Default=0.6 </i>
    - <i>print_every: The epoch step number that when reached, prints training details out to console. Default=200</i>
  - Usage:
    - <i>python3 train_transformer.py --arg1 arg1val --arg2 arg2val ... etc.</i>


## File Structure
- <b>/src/</b>: <i>finished/completed files, files belong here after believing that minimal changes will need to be made to them.</i>
  - <b>/libs/</b>: <i>A place for helper files and functions. Additional definitions here as well, i.e. Transformer model.</i>
  - <b>/mid_data_collections/</b>: <i>the collection of 10,000 .mid files separated into subsections for convenience.</i>
  - <b>/misc/</b>: <i>A folder for random things.</i>
  - <b>/models/</b>: <i>Where saved models are stored.</i>
  - <b>/neural_mode_classifier/</b>: <i> the notebook containing the CNN for classifying spectrograms, as well as a folder containing trained models</i>
  - <b>/numpy_data_collections/</b>: <i>the collection of converted .midi files to .npy files for easier loading into training data, etc. Separated into subsections for convenience.</i>
  - <b>/numpy_path/</b>: <i>The path where the complete set of converted .npy music files are stored to be read into program.</i>
  - <b>/song_data/</b>: <i>Files are stored here that are related to the dataset, songnames, IDs, etc.</i> 
  - <b>/util/</b>: <i>A place for utility files to be stored. i.e., "create_npy.py" for .mid to .npy conversion.</i>
- <b>/dev/</b>: <i>A development folder used to develope files/methods/etc., store temp files, that when completed will be moved to '/src/'.</i>
