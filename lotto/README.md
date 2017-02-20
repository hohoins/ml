deep learning classification으로 로또 1등을 예측해보자

1. 로또 데이터는 로또 site에서 excel로 다운로드 가능
http://www.nlotto.co.kr/gameResult.do?method=allWin
1회차 ~ 741회차 까지 다운로드 함

2. X, Y 데이터 정의
X는 회차 1~741, 1 feature로 설정
Y는 6개의 당첨 번호를 입력
[741회차 5, 21, 27, 34, 44, 45]
배열 45개를 잡아서 인덱스를 마킹함
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 1, 1]

3. genData.py
genData.py는 자연수로 된 당첨 번호를
45개 배열로 바꿔주는 프로그램
ori.data => input.data 작업을 해줌
ori.data, 741	5	21	27	34	44	45
input.data, [741, 0, 0, 0, 0, 1, ... ... 1, 1]

4. genNumber.py
로또 1등 예측을 해줌
(사실은 전혀 안되고 있음 ㅋㅋㅋ OTL)

