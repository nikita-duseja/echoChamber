import os
import sys

# file is 8,164 lines longs (contains that many user polarities)

with open("USER_POLARITY_BARBERA.txt") as inp:
    for line in inp:
        values    = line.split("\t") # id, polarity
        user_id   = values[0]
        polarity  = values[1]
        #print(polarity)
        
        # max is  2.439189
        # min is -2.519504
        if float(polarity) > 2:    # 2 to 3
            polarity_category = 5  # new category is 5
        elif float(polarity) > 1:  # 1 to 2
            polarity_category = 4  # new category is 4
        elif float(polarity) > 0:  # 0 to 1
            polarity_category = 3  # new category is 3
        elif float(polarity) > -1: # -1 to 0
            polarity_category = 2  # new category is 2
        elif float(polarity) > -2: # -2 to -1
            polarity_category = 1  # new category is 1
        else:                      # -3 to -2
            polarity_category = 0  # new category is 0

        with open("Categorized_User_Polarity.txt", "a+") as write_to_file:
            write_to_file.write(user_id + '\t' + str(polarity_category) + '\r')
