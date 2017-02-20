import numpy as np

# data
input = np.loadtxt('feature_ori.data', unpack=True, dtype='int')
data = np.transpose(input)

FEATURE_COUNT = 12
buckets = [0] * (FEATURE_COUNT+45)
output = []

for i in range(len(data)):
    week = data[i]
    buckets[0] = week[0]  # 회차
    buckets[1] = week[1]  # 회차
    buckets[2] = week[2]  # 회차
    buckets[3] = week[3]  # 회차
    buckets[4] = week[4]  # 년
    buckets[5] = week[5]  # 년
    buckets[6] = week[6]  # 년
    buckets[7] = week[7]  # 년
    buckets[8] = week[8]  # 월
    buckets[9] = week[9]  # 월
    buckets[10] = week[10]  # 일
    buckets[11] = week[11]  # 일

    for w in range(12, len(week)):
        idx = week[w]
        buckets[idx+11] = 1
        if w == len(week)-1:
            output.append(buckets)
            buckets = [0] * (FEATURE_COUNT+45)

print(output)
p = '{0}'.format(output)
oriFile = open('feature_input.data', 'w')
oriFile.write(p)


