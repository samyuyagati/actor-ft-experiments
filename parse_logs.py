import csv
import sys
import matplotlib.pyplot as plt

# Author: samyu@berkeley.edu
# Python script to parse task ID and duration data from output of 
# get_duration_from_logs.sh. Also constucts a bar graph showing each
# task's duration relative to the average duration of this set of
# tasks.

# TODO debug the reading arguments part
if len(sys.argv) == 3:
	in_file = str(sys.argv)[1]
	plot_file = str(sys.argv)[2]
else:
	in_file = "out.csv"
	plot_file = "duration_plot.png"

# Read in duration data from CSV file
process_ids = []
durations = []
with open(in_file, 'r') as fd:
	reader = csv.reader(fd, delimiter=" ")
	for row in reader:
		process_ids.append(row[0])
		durations.append(int(row[1][:-2])) # remove "ms" from duration

# Plot data (currently bar chart with horizontal line to
# show mean task duration) and save figure.
mean_duration = float(sum(durations)) / len(durations)
x_pos = [i for i in range(1, len(durations) + 1)]
plt.bar(x_pos, durations)
plt.xlabel("Task ID")
plt.ylabel("Task duration (ms)")
plt.title("Per-task durations (ms)")
plt.xticks(x_pos, process_ids, rotation="vertical")
plt.hlines(mean_duration, 0, x_pos[-1] + 1, colors=["k"], linestyles='dashed', label="Mean duration")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(plot_file)
plt.show()
