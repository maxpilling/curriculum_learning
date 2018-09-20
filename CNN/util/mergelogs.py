import os
import io
import glob
import fnmatch
import sys
import re

DIGITS = r"\d+"
TARGET_STRING = "Episode * ended. Score *"


def merge_logs(model_name):
    scores = []
    log_files = glob.glob(model_name + "*.log")
    log_files.sort(key=lambda x: os.path.getmtime(x))
    print("Found log files...")

    for log_file in log_files:
        print("Adding new log file...")
        match_array = []
        numbers_in_line = []
        file = open(log_file, "r")
        for line in file:
            if fnmatch.fnmatch(line, TARGET_STRING):
                match_array.append(line)

        for line in match_array:
            numbers_in_line = re.findall(DIGITS, line)
            scores.append(int(numbers_in_line[1]))

    print("Making model directory...")
    print("Creating CSV...")
    csv_file = open(model_name + "_summary.csv", "w")

    for episode_counter, score in enumerate(scores):
        csv_file.write(str(episode_counter) + "," + str(score) + "\n")

    print("Summary Below")
    print("Model:", model_name)
    print("Number of episodes:", len(scores))
    print("Max value:", max(scores))
    print("Min value:", min(scores))
    print("Mean value:", sum(scores) / len(scores))


if __name__ == "__main__":
    merge_logs(sys.argv[1])
