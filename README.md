# MusGen
A repository of the Music Transformer and Neural Transposer designed to aide Chatbots evoke emotion through deterministically keyed music. 




## File Structure
- <b>/src/</b>: <i>finished/completed files, files belong here after believing that minimal changes will need to be made to them.</i>
  - <b>/libs/</b>: <i>A place for helper files and functions. Additional definitions here as well, i.e. Transformer model.</i>
  - <b>/mid_data_collections/</b>: <i>the collection of 10,000 .mid files separated into subsections for convenience.</i>
  - <b>/misc/</b>: <i>A folder for random things.</i>
  - <b>/models/</b>: <i>Where saved models are stored.</i>
  - <b>/numpy_data_collections/</b>: <i>the collection of converted .midi files to .npy files for easier loading into training data, etc. Separated into subsections for convenience.</i>
  - <b>/numpy_path/</b>: <i>The path where the complete set of converted .npy music files are stored to be read into program.</i>
  - <b>/song_data/</b>: <i>Files are stored here that are related to the dataset, songnames, IDs, etc.</i> 
  - <b>/util/</b>: <i>A place for utility files to be stored. i.e., "create_npy.py" for .mid to .npy conversion.</i>
- <b>/dev/</b>: <i>A development folder used to develope files/methods/etc., store temp files, that when completed will be moved to '/src/'.</i>
