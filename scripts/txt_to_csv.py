# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import csv
import sys
import shlex

# File to be converted from txt to csv format
if len(sys.argv) > 1:
    parameter = sys.argv[1]

time = "" # Variable to store the time of the log to avoid duplicates
with open(f'/home/ubuntu/Auditd_AI/data/{parameter}.txt', "r") as file:
    for line_num, line in enumerate(file):
        
        if "----" in line:
            continue # Skip the lines with ---- in it

        if "grantors=" not in line and ',' in line:
            line = line.replace(',', ' ') # Replace , with space to avoid issues in csv format

        if "'" in line:
            line = line.replace("'", "") # Remove ' from the log line to avoid issues in csv format
        
        line_array = line.strip().split(' ') # Split the lines into array for cleaning

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

            if "ausearch" in i or "date" in i or "attack_sim.sh" in i:
                time = line_array[2] # Store the time of the log to avoid duplicates
            
        for i in range(len(line_array)):
            if "msg=o" in line_array[i] or "msg=unit" in line_array[i] or "msg=cwd" in line_array[i]:
                temp_line = line_array[i].replace("msg=", "")
                line_array[i] = temp_line
                temp_array = line_array[i].split(" ")
                line_array[i] = temp_array[0]
                line_array.insert(i + 1, temp_array[1])

        if "DATA" in line_array[0] or "PRIVILEGE" in line_array[0]:
            combined_label = " ".join(line_array[0:2])
            line_array.pop(1)
            line_array[0] = combined_label

         # Store the time of the log to avoid duplicates
        if line_array[1] == "type=PROCTITLE":
            # Combine all commands into one element
            command_array = []
            command_array.append(line_array[3])
            for i in line_array[4:]:
                if "=" not in i:
                    command_array.append(i)
            combined_commands = " ".join(command_array)
            line_array[3] = combined_commands
            del line_array[4:len(command_array) + 3]

            #Skip ausearch commands being ran
            if "ausearch" in line_array[3] or "date" in line_array[3] or "attack_sim.sh" in line_array[3]:
                time = line_array[2]
        


        i = 0
        while i < len(line_array):
            if "cmd=ausearch" in line_array[i]:
                command_array = []
                command_array.append(line_array[i])
                
                # Count how many elements to combine
                elements_to_remove = 0
                for j in line_array[i + 1:]:
                    if "=" not in j:
                        command_array.append(j)
                        elements_to_remove += 1
                    else:
                        break  # Stop at next key=value pair
                
                # Combine the commands
                combined_commands = " ".join(command_array)
                line_array[i] = combined_commands
                
                # Remove the combined elements
                if elements_to_remove > 0:
                    del line_array[i + 1:i + 1 + elements_to_remove]
                
                print(line_array)
            
            i += 1  # Only increment after processing

        for i in range(len(line_array)):
            if "ausearch" in line_array[i]:
                time = line_array[2]

        with open(f"/home/ubuntu/Auditd_AI/data/{parameter}.csv", "a", newline='') as csv_file:
            if time == line_array[2]:
                continue
            else:
                write = csv.writer(csv_file)
                write.writerow(line_array)