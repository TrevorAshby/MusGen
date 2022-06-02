FIND_KEY = "Eb_major"
KEY_FIELD_POS = 3
print("Finding all songs that are: ", FIND_KEY)

classified_songs = open("./classified_songs.txt", "r", encoding="utf8")
found_key_songs = open("./{}_songs.txt".format(FIND_KEY), "w")
lines = classified_songs.readlines()

for line in lines:
    fields = line.split('   ')
    #print("Field: ", fields[KEY_FIELD_POS])
    #print("Find key: ", FIND_KEY)
    if fields[KEY_FIELD_POS].strip() == FIND_KEY:
        #print("writing line")
        found_key_songs.write(line)
        # add line to new file
    #print(fields[KEY_FIELD_POS])
    #break