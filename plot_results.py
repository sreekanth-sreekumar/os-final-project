import matplotlib.pyplot as plt
import numpy as np

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

x = []
y = []

with open('training_results.txt', 'r') as f:
    lines = f.readlines()

for line in lines[1:]:
    splits = line.split(',')
    it = int(splits[0])
    value = float(splits[1][find_nth(splits[1], '(', 2)+1: find_nth(splits[1], ')', 1)])
    # value = float(splits[1][splits[1].find('(')+1:splits[1].find(')')])
    utl = float(splits[2][splits[2].find('(')+1:splits[2].find(')')])
    x.append(utl)
    y.append(value)

u = []
v = []
u1 = []
v1 = []
u2 = []
v2 = []

for i in range(0, len(x), 50):
    u.append(x[i])
    v.append(y[i])

with open('comparison_results.txt', 'r') as f:
    lines = f.readlines()

for line in lines[1:]:
    splits = line.split(',')
    sjf_reward = float(splits[1])
    sjf_util = float(splits[2])
    packer_reward = float(splits[3])
    packer_util = float(splits[4])
    u1.append(sjf_reward)
    v1.append(sjf_util)
    u2.append(packer_reward)
    v2.append(packer_util)

x1 = []
y1 = []
x2 = []
y2 = []
for i in range(0, len(u1), 50):
    x1.append(u1[i])
    y1.append(v1[i])
    x2.append(u2[i])
    y2.append(v2[i])

print(u, v)
print(x1, y1)
print(x2, y2)

print(x1, y1)
plt.plot(u,v, color='r')
# plt.plot(x1, y1, color='g')
# plt.plot(x2, y2, color='b')
plt.xlabel('Resource Utilization')
plt.ylabel('Total Job Slowdown')
# plt.legend(["Compact Scheduler", "Spread Scheduler"])
plt.show()