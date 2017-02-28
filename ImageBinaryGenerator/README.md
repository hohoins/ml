<pre>
  
딥러닝 CIFAR-10 CNN 예제에 내 사진을 넣어서 학습 시켜 보자  
http://blog.naver.com/cenodim/220946688251

번 목표는 내가 가지고 있는 사진을 이용해서
CNN 딥러닝을 시켜 보는 것임


일단 해본 코드는 GitHub에 있음
셈플 사진은 4장만 남기고 다지웠음
https://github.com/hohoins/ml/tree/master/ImageBinaryGenerator
ml/ImageBinaryGenerator at master · hohoins/ml · GitHub
github.com



이미 CNN예제들은 널렸는데
그 중에 교과서 적인 예제는 MNIST 손글씨 예제나, CIFAR-10 이미지 분류 예제들임
문제는 실행하면 서버에서 예제 이미지들을 바이너리로 가져와서 실행 시켜주는데
나중에 그 학습 이미지들을 내 사진으로 바꿀려고 하면
어떻게 해야 할지 감이 아에 안옴


그래서 일단 CIFAR-10예제를 돌려서 성공시킨 다음에
CIFAR-10 예제의 입력을 내 사진으로 바꾸었음
일단 CIFAR-10을 돌린 부분은 이전 포스팅에 있음
http://blog.naver.com/cenodim/220945521396

CIFAR-10 딥러닝으로 학습해보기
CIFAR-10 딥러닝으로 학습해보기 https://github.com/hohoins/ml/tree/master/cifar10ml/cifar10 at m...
blog.naver.com


https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR-10 사이트에 가보면 사진 6만장을 학습데이터로 제공함


https://www.tensorflow.org/tutorials/deep_cnn
텐서플로우 사이트에 가면 저 예제를 사용하는 방법이 잘 설명 되어 있음



https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/
CIFAR-10 예제는 텐서플로우 GIT-HUB에 있음


일단 CIFAR-10예제의 입력을 바꿔주려면
CIFAR-10 예제에 사용되는 Binaray파일의 구조를 잘알아야함

1. CIFAR-10에서 DATA SET을 다운로드 받음 "cifar-10-binaray.gz"파일이 다운로드 됨

2. 앞축 해제된 파일들을 확인함, 사진이 아니고 _1.bin, _2.bin ... bin 파일만 있음. 이 bin파일이 실제로 사진임

3. CIFAR-10에서 확인해보면 저 bin파일 내부에
label + Red + Greeb + Blue 
label + Red + Greeb + Blue
label + Red + Greeb + Blue
이런 형식으로 사진이 몇 만장이 기록되어 있음




CIFAR-10 바이너리 규격


이제 다 알았음
만들기만 하면 됨 OTL
     


일단 genBinFile.py라는 것을 만듬
genBinFile.py는 data라는 폴더 내부의 폴더들을 모두 bin으로 만들어줌
예로 
/data/book
/data/booth
/data/road
/data/speedmeter
이런식으로 4개의 폴더가 있고 각각에 폴더에 사진이 여러개씩 있음


그러면 genBinFile.py를 실행하면
book => label 0
booth=> label 1
road=> label 2
speedmeter => label 3
폴더들을 레이블에 매핑 시키고 레이블 별로 이미지들을 적용시키셔
bin을 만들어 냄



실행 결과임 

총 103장을 bin에 넣었고
bin파일 잘 만들어짐


그러면 확인을 해봐야지
loadBinFile.py를 실행하면
ㅋㅋㅋ 오 잘 저장 되었음
loadBinFile.py는 생성된 bin에서 
원하는 위치에 사진을 보여주는 기능이있음
아까 103장을 넣었으니 0~102까지의 값을 넣으면
사진을 확인 할수 있음 


 

CIFAR-10 예제를 열어서 조금 바꿔줘야함
하지만 걱정 안해도 됨 
이미 다 해서 GitHub에 있기 때문에 ㅋㅋ



  

입력 경로를 변경해준 cifar10_train.py 실행
오 사진이 103장 밖에 안되니까 엄청 빨리됨
오 한 10000번 정도 돌리면 코스트도 0.1정도로 떨어짐
오 진짜 별것 아닌거 같은데
진짜 레알 오래 걸림
아 레알 파이썬도 할 줄 모르는데
구글 검색 해가면서
멘탈로 해냄
ㅋㅋㅋㅋ


해본 코드는 GitHub에 있음
셈플 사진은 4장만 남기고 다지웠음
https://github.com/hohoins/ml/tree/master/ImageBinaryGenerator
ml/ImageBinaryGenerator at master · hohoins/ml · GitHub
github.com

</pre>
