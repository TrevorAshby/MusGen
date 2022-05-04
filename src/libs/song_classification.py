import numpy as np
from collections import Counter

# note integer converted to note string value
def int_to_note(integer):
  integer = integer % 12
  if integer == 0:
    return 'C'
  if integer == 1:
    return 'C#|Db'
  if integer == 2:
    return 'D'
  if integer == 3:
    return 'D#|Eb'
  if integer == 4:
    return 'E'
  if integer == 5:
    return 'F'
  if integer == 6:
    return 'F#|Gb'
  if integer == 7:
    return 'G'
  if integer == 8:
    return 'G#|Ab'
  if integer == 9:
    return 'A'
  if integer == 10:
    return 'A#|Bb'
  if integer == 11:
    return 'B'

def pos_to_key(pos):
  keys = ['C_major', 'C#_major', 'D_major', 'Db_major', 'E_major', 'Eb_major', 'F_major', 'F#_major',\
          'G_major', 'Gb_major', 'A_major', 'Ab_major', 'B_major', 'Bb_major', 'C_minor', 'C#_minor',\
          'D_minor', 'D#_minor', 'E_minor', 'Eb_minor', 'F_minor', 'F#_minor', 'G_minor', 'G#_minor',\
          'A_minor', 'B_minor', 'Bb_minor']
  return keys[pos]

major_keys = {
  # MAJOR KEYS
  'C_major' : ['C','D','E','F','G','A','B'], # 1
  'C#_major' : ['C#|Db','D#|Eb','F','F#|Gb','G#|Ab','A#|Bb','B'], # 2
  #'Cb_major' : ['B','C#|Db','D#|Eb','E','F#|Gb','G#|Ab','A#|Bb'], # 3
  'D_major' : ['D','E','F#|Gb','G','A','B','C#|Db'], # 4
  'Db_major' : ['C#|Db','D#|Eb','F','F#|Gb','G#|Ab','A#|Bb','C'], # 5
  'E_major' : ['E','F#|Gb','G#|Ab','A','B','C#|Db','D#|Eb'], # 6
  'Eb_major' : ['D#|Eb','F','G','G#|Ab','A#|Bb','C','D'], # 7
  'F_major' : ['F','G','A','A#|Bb','C','D','E'], # 8
  'F#_major' : ['F#|Gb','G#|Ab','A#|Bb','B','C#|Db','D#|Eb','F'], # 9
  'G_major' : ['G','A','B','C','D','E','F#|Gb'], # 10
  'Gb_major' : ['F#|Gb','G#|Ab','A#|Bb','B','C#|Db','D#|Eb','F'], # 11
  'A_major' : ['A','B','C#|Db','D','E','F#|Gb','G#|Ab'], # 12
  'Ab_major' : ['G#|Ab','A#|Bb','C','C#|Db','D#|Eb','F','G'], # 13
  'B_major' : ['B','C#|Db','D#|Eb','E','F#|Gb','G#|Ab','A#|Bb'], # 14
  'Bb_major' : ['A#|Bb','C','D','D#|Eb','F','G','A'] # 15
  }
minor_keys = {
  # MINOR KEYS
  'C_minor' : ['C','D','D#|Eb','F','G','G#|Ab','A#|Bb'], # 1
  'C#_minor' : ['C#|Db','D#|E','E','F#|Gb','G#|Ab','A','B'], # 2
  'D_minor' : ['D','E','F','G','A','A#|Bb','C'], # 3
  'D#_minor' : ['D#|Eb','F','F#|Gb','G#|Ab','A#|Bb','B','C#|Db'], # 4
  'E_minor' : ['E','F#|Gb','G','A','B','C','D'], # 5
  'Eb_minor' : ['D#|Eb','F','F#|Gb','G#|Ab','A#|Bb','B','C#|Db'], # 6
  'F_minor' : ['F','G','G#|Ab','A#|Bb','C','C#|Db','D#|Eb'], # 7
  'F#_minor' : ['F#|Gb','G#|Ab','A','B','C#|Db','D','E'], # 8
  'G_minor' : ['G','A','A#|Bb','C','D','D#|Eb','F'], # 9
  'G#_minor' : ['G#|Ab','A#|Bb','B','C#|Db','D#|Eb','E','F#|Gb'], # 10
  'A_minor' : ['A','B','C','D','E','F','G'], # 11
  'A#_minor' : ['A#|Bb','B','C#|Db','D#|Eb','F','F#|Gb','G#|Ab'], # 12
  #'Ab_minor' : ['G#|Ab','A#|Bb','B','C#|Db','D#|Eb','E','F#|Gb'], # 13
  'B_minor' : ['B','C#|Db','D','E','F#|Gb','G','A'], # 14
  'Bb_minor' : ['A#|Bb','C','C#|Db','D#|Eb','F','F#|Gb','G#|Ab'], # 15
  }

combined_keys = {
  # MAJOR KEYS
  'C_major' : ['C','D','E','F','G','A','B'], # 1
  'C#_major' : ['C#|Db','D#|Eb','F','F#|Gb','G#|Ab','A#|Bb','B'], # 2
  #'Cb_major' : ['B','C#|Db','D#|Eb','E','F#|Gb','G#|Ab','A#|Bb'], # 3
  'D_major' : ['D','E','F#|Gb','G','A','B','C#|Db'], # 4
  'Db_major' : ['C#|Db','D#|Eb','F','F#|Gb','G#|Ab','A#|Bb','C'], # 5
  'E_major' : ['E','F#|Gb','G#|Ab','A','B','C#|Db','D#|Eb'], # 6
  'Eb_major' : ['D#|Eb','F','G','G#|Ab','A#|Bb','C','D'], # 7
  'F_major' : ['F','G','A','A#|Bb','C','D','E'], # 8
  'F#_major' : ['F#|Gb','G#|Ab','A#|Bb','B','C#|Db','D#|Eb','F'], # 9
  'G_major' : ['G','A','B','C','D','E','F#|Gb'], # 10
  'Gb_major' : ['F#|Gb','G#|Ab','A#|Bb','B','C#|Db','D#|Eb','F'], # 11
  'A_major' : ['A','B','C#|Db','D','E','F#|Gb','G#|Ab'], # 12
  'Ab_major' : ['G#|Ab','A#|Bb','C','C#|Db','D#|Eb','F','G'], # 13
  'B_major' : ['B','C#|Db','D#|Eb','E','F#|Gb','G#|Ab','A#|Bb'], # 14
  'Bb_major' : ['A#|Bb','C','D','D#|Eb','F','G','A'], # 15
  # MINOR KEYS
  'C_minor' : ['C','D','D#|Eb','F','G','G#|Ab','A#|Bb'], # 1
  'C#_minor' : ['C#|Db','D#|E','E','F#|Gb','G#|Ab','A','B'], # 2
  'D_minor' : ['D','E','F','G','A','A#|Bb','C'], # 3
  'D#_minor' : ['D#|Eb','F','F#|Gb','G#|Ab','A#|Bb','B','C#|Db'], # 4
  'E_minor' : ['E','F#|Gb','G','A','B','C','D'], # 5
  'Eb_minor' : ['D#|Eb','F','F#|Gb','G#|Ab','A#|Bb','B','C#|Db'], # 6
  'F_minor' : ['F','G','G#|Ab','A#|Bb','C','C#|Db','D#|Eb'], # 7
  'F#_minor' : ['F#|Gb','G#|Ab','A','B','C#|Db','D','E'], # 8
  'G_minor' : ['G','A','A#|Bb','C','D','D#|Eb','F'], # 9
  'G#_minor' : ['G#|Ab','A#|Bb','B','C#|Db','D#|Eb','E','F#|Gb'], # 10
  'A_minor' : ['A','B','C','D','E','F','G'], # 11
  'A#_minor' : ['A#|Bb','B','C#|Db','D#|Eb','F','F#|Gb','G#|Ab'], # 12
  #'Ab_minor' : ['G#|Ab','A#|Bb','B','C#|Db','D#|Eb','E','F#|Gb'], # 13
  'B_minor' : ['B','C#|Db','D','E','F#|Gb','G','A'], # 14
  'Bb_minor' : ['A#|Bb','C','C#|Db','D#|Eb','F','F#|Gb','G#|Ab'], # 15
  }

def calculate_occurence_score(song, scores):
  # split song into tokens and append anything that is a "note"
  collection_of_notes = []
  for it in song.to_text().split(' '):
    if it[0] == 'n':
      collection_of_notes.append(int_to_note(int(it[1:])))
  note_ctr = Counter(collection_of_notes)
  
  # SCORE FOR EACH KEY
  for idx in range(27):
    curr_key = pos_to_key(idx)
    curr_key_notes = combined_keys[curr_key]
    temp_idx = 0
    # FOR EVERY NOTE IN SONG
    for note in note_ctr:
      # IF NOTE IS IN KEY
      if note in curr_key_notes:
        # INCREASE SCORE FOR KEY numkeyoccurences / total notes
        scores[idx] += (list(note_ctr.values())[temp_idx] / sum(note_ctr.values()))
      else:
        # DECREASE SCORE FOR KEY numkeyoccurences / total notes IF NOT IN RIGHT KEY
        scores[idx] -= (list(note_ctr.values())[temp_idx] / sum(note_ctr.values()))

      temp_idx += 1
  #return note_ctr, sum(note_ctr.values())
  return scores, note_ctr

def calculate_first_last_note_score(song, scores, note_ctr, first_or_last):
  firstNote = 0
  lastNote = 0
  for it in song.to_text().split(' '):
    if it[0] == 'n':
      if firstNote == 0:
        firstNote = int(it[1:])
      lastNote = int(it[1:])

  for idx in range(27):
    curr_key = pos_to_key(idx)
    # check if first note matches key
    curr_key_split = curr_key.split('_')
    # if the note portion of the key is present in the note FIRST NOTE
    if first_or_last == 'first':
      if '|' in int_to_note(firstNote):
        curr_note_split = int_to_note(firstNote).split('|')
        if (curr_key_split[0] == curr_note_split[0]) or (curr_key_split[0] == curr_note_split[1]):
          #print('FOUND')
          scores[idx] += (note_ctr[int_to_note(firstNote)] / sum(note_ctr.values()))
        else:
          scores[idx] -= (note_ctr[int_to_note(firstNote)] / sum(note_ctr.values()))
      else:
        if (curr_key_split[0] == int_to_note(firstNote)):
          scores[idx] += (note_ctr[int_to_note(firstNote)] / sum(note_ctr.values()))
        else:
          scores[idx] -= (note_ctr[int_to_note(firstNote)] / sum(note_ctr.values()))
    #if the note portion of the key is present in the note LAST NOTE
    elif first_or_last == 'last':
      if '|' in int_to_note(lastNote):
        curr_note_split = int_to_note(lastNote).split('|')
        #print(lastNote)
        if (curr_key_split[0] == curr_note_split[0]) or (curr_key_split[0] == curr_note_split[1]):
          scores[idx] += (note_ctr[int_to_note(lastNote)] / sum(note_ctr.values()))
        else:
          scores[idx] -= (note_ctr[int_to_note(lastNote)] / sum(note_ctr.values()))
      else:
        if (curr_key_split[0] == int_to_note(lastNote)):
          scores[idx] += (note_ctr[int_to_note(lastNote)] / sum(note_ctr.values()))
        else:
          scores[idx] -= (note_ctr[int_to_note(lastNote)] / sum(note_ctr.values()))
    
  return scores


# THIS COMBINES THE TWO SCORES
def calculate_score(song):
  key_scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  key_scores, note_counts = calculate_occurence_score(song, key_scores.copy())
  # if not 1 max, do next
  count = np.count_nonzero(key_scores == max(key_scores))
  if count > 1:
    key_scores = calculate_first_last_note_score(song, key_scores.copy(), note_counts, 'first')
  # if not 1 max, do next
    count = np.count_nonzero(key_scores == max(key_scores))
    if count > 1:
      key_scores = calculate_first_last_note_score(song, key_scores.copy(), note_counts, 'last')
      count = np.count_nonzero(key_scores == max(key_scores))
  return pos_to_key(np.argmax(key_scores, axis=0)), note_counts