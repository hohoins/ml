import numpy as np

# 대략 한 2000개쯤 만들겠지
DATA_SIZE = 20000

# 반 0번 줄로 세팅
# 반 1번 줄로 세팅
cluster = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
]

cluster_count = len(cluster)
batch_count = DATA_SIZE / cluster_count

print("DATA_SIZE", DATA_SIZE)
print("cluster_count", cluster_count)
print("batch_count", batch_count)
print("DATA_SIZE/batch_count", DATA_SIZE-1/batch_count)

FEATURE_COUNT = 1
output = []

for i in range(DATA_SIZE-1, 0, -1):
    idx = i / batch_count
    output.append([i] + cluster[int(idx)])

print(output)
p = '{0}'.format(output)
oriFile = open('test_input.data', 'w')
oriFile.write(p)


