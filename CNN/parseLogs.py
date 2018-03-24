import fnmatch
import os
import re
import sys

import numpy as np

DIGITS = r'\d+'
TARGET_STRING = "Episode * ended. Score *"

def main():

    if len(sys.argv) == 1:
        print("Need to pass in log files!")
        return

    input_files = sys.argv[1:]

    output_results = []

    for file in input_files:
        print(f"Parsing scores for {file}...")
        generation, score = parse_log_file(file)

        new_file_name = os.path.splitext(file)[0] + ".csv"
        print(f"Saving scores to {new_file_name}...")
        with open(new_file_name, 'wb+') as csv_file:
            stacked =  np.column_stack((generation, score))
            np.savetxt(csv_file, stacked, fmt='%s', delimiter=',')

    print("Done!")

    return

def parse_log_file(input_file):

    match_array = []

    generations = []
    scores = []

    with open(input_file, 'r') as log_file:
        for line in log_file:
            if fnmatch.fnmatch(line, TARGET_STRING):
                match_array.append(line)

    for line in match_array:
        numbers_in_line = re.findall(DIGITS, line)

        generations.append(int(numbers_in_line[0]))
        scores.append(int(numbers_in_line[1]))

    return generations, scores


if __name__ == "__main__":
    main()

