# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import csv

with open('training_data.txt', "r") as file:
    for line_num, line in enumerate(file):

        if "----" in line:
            continue # Skip the lines with ---- in it
        
        line_array = line.strip().split(' ') # Split the lines into array for cleaning

        #if line_array[1] == "----":
            #continue # Skip the lines with ---- in it

        if ":" in line_array:
            line_array.remove(":") # Remove the : in each line of log

        if "\"" in line_array:
            line_array.remove("\"") # Remove the " in each line of log
        
        for i in line_array:
            # Remove " from the msg field
            if '"' in i:
                index = line_array.index(i)
                line_array[index] = line_array[index].replace('"', "")
            
            # Combine msg field into one entry
            if "msg=" in i:
                index = line_array.index(i)
                line_array[index] = line_array[index] + " " + line_array[index + 1]
                line_array.pop(index + 1)

        time = ""
        if line_array[1] == "type=PROCTITLE":
            # Combine all commands into one element
            combined_commands = " ".join(line_array[3:])
            del line_array[3:]
            line_array.append(combined_commands)

            # Skip ausearch commands being ran
            if "ausearch" in line_array[3]:
                time = line_array[3]

        print(line_array)

        with open("training_data.csv", "a", newline='') as csv_file:
            if time == line_array[3]:
                continue
            else:
                write = csv.writer(csv_file)
                write.writerow(line_array)