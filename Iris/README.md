

UCI iris를 딥러닝으로 100% 맞춰보자
================================================
<pre>
예전에 머신러닝 iris문제에 대해서
유투브에서 본것 같은데 어디서 본지는 모르겠는데
기억이 나서 한번 직접 해봐야지 하는 생각이듬

iris 데이터 주는곳 찾음 ㅋㅋ
http://archive.ics.uci.edu/ml/datasets/Iris


소스코드를 다운로드 하고
train.py를 실행 하시면 됩니다.


Attribute Information:
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris Setosa
-- Iris Versicolour
-- Iris Virginica

X정의
4개의 cm로 된 feature가 있거고

Y정의
3가지 꽃별로 레이블을 붙임
Iris Setosa [1, 0, 0]
Iris Versicolour [0, 1, 0]
Iris Virginica [0, 0, 1]


학습 결과 ㅋㅋ
Cost가 0이됨
하지만 test set에서 정확도가 97프로가 나옴
Epoch: 0001 cost= 0.630084217
Epoch: 8001 cost= 0.000019475
Epoch: 16001 cost= 0.000000111
Epoch: 24001 cost= 0.000000001
Epoch: 32001 cost= 0.000000000
Epoch: 40001 cost= 0.000000000
Epoch: 48001 cost= 0.000000000
Accuracy: 0.966667


그래서 필살기로
Dropout 0.9 하니까
100프로 맞춤
Epoch: 0001 cost= 0.679429412
Epoch: 8001 cost= 0.097615846
Epoch: 16001 cost= 0.093704201
Epoch: 24001 cost= 0.135772601
Epoch: 32001 cost= 0.165448636
Epoch: 40001 cost= 0.091057405
Epoch: 50001 cost= 0.103997864
Accuracy: 1.0

</pre>
