import numpy as np

# data
input = np.loadtxt('ori.data', unpack=True, dtype='int')
data = np.transpose(input)

FEATURE_COUNT = 1
buckets = [0] * (FEATURE_COUNT+45)
output = []

for i in range(len(data)):
    week = data[i]
    buckets[0] = week[0]  # 회차

    for w in range(1, len(week)):
        idx = week[w]
        buckets[idx] = 1
        if w == len(week)-1:
            output.append(buckets)
            buckets = [0] * (FEATURE_COUNT+45)

print(output)
p = '{0}'.format(output)
oriFile = open('input.data', 'w')
oriFile.write(p)


