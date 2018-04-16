import matplotlib.pyplot as plt
import numpy as np

# Open the file for analyzing and graphing
file = open("loss.txt", "r").readlines()
# Split the commas to get an array of values
result = file[0].split(",")

# Remove the last comma and empty gap
results = np.array(result[:-1]).astype(np.float)

plt.plot(results)
plt.show()
