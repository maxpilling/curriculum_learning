import matplotlib.pyplot as plt
import numpy as np


# Open the file for analyzing and graphing
file1 = open("loss.txt", "r").readlines()
# Split the commas to get an array of values
resultLoss = file1[0].split(",")

# Remove the last comma and empty gap
resultLoss2 = np.array(resultLoss[:-1]).astype(np.float)

plt.plot(resultLoss2)
plt.show()


# Open the file for analyzing and graphing
file = open("score.txt", "r").readlines()
# Split the commas to get an array of values
result = file[0].split(",")

# Remove the last comma and empty gap
results = np.array(result[:-1]).astype(np.float)

results = np.add.reduceat(results, np.arange(0, len(results), 1)) / 1

plt.plot(results)
plt.show()
