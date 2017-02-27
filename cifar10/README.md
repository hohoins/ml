CIFAR-10 딥러닝으로 학습해보기
<pre>

얼마전에 로또 학습시기다 맨탈 다털림
https://github.com/hohoins/ml/tree/master/lotto


그래도 그 후에 iris 학습 100프로 시키고 자신감 업됨
https://github.com/hohoins/ml/tree/master/Iris


그래서 이제 제법 사이즈 있게ㅋㅋ
[32 * 32 * 3] 5만장으로 학습 해볼려구 함
AWS도 아직 안했구, GPU도 없음 어떻게 하지 ㄷㄷ


사이트는 여기임
CIFAR-10 is a common benchmark in machine learning for image recognition.
http://www.cs.toronto.edu/~kriz/cifar.html


Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU.
We also demonstrate how to train a CNN over multiple GPUs.
Detailed instructions on how to get started available at:
http://tensorflow.org/tutorials/deep_cnn/


CNN으로 코드가 이미 다 되어 있음
실행만 하면 되지만 실제로 ㅋㅋㅋ TF 1.0에 맞게
좀 마이그레이션 해주어야 함


문제는 너무 느림 ㅋㅋㅋ 26시간 정도 해야지
예제에서 적용되어 있는 기본 학습 횟수 1000000번을 채울수 있음 ㅠㅠ


그냥 1일만 했음
<img src="https://github.com/hohoins/ml/blob/master/cifar10/eval.png?raw=true"/>


12시간 정도 하니까 정확도가 84프로 정도 나옴
<img src="https://github.com/hohoins/ml/blob/master/cifar10/eval_time.png?raw=true"/>


코스트
<https://github.com/hohoins/ml/blob/master/cifar10/train.png?raw=true/>


코스트
<https://github.com/hohoins/ml/blob/master/cifar10/train_time.png?raw=true/>


</pre>


