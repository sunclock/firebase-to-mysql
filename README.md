# firebase-to-mysql


## 개발 환경

### 서버

|라이브러리 이름|버전|
|------|---|
|body-parser|^1.19.0|
|cors|^2.8.5|
|express|^4.17.1|
|mysql2|^2.2.5|
|sequelize|^6.6.2|

### 클라이언트

|라이브러리 이름|버전|
|------|---|
|axios|^0.21.1|
|firebase|^8.6.8|
|react|^17.0.2|
|react-dom|^17.0.2|
|react-scripts|4.0.3|
|web-vitals|^1.0.1|

## 왜 만들었나?

### 문제 상황
기존의 [soul food 심리테스트](https://github.com/sunclock/soul_food) 에선 백엔드로 firebase를 사용했다. 
응답 데이터에 관한 통계를 만들고 싶은데, firebase 환경이나 react의 javascript 환경에선 데이터 조작이 어려웠다.

### 해결 방안 1 - 실패
데이터를 .csv 파일로 내보내기 하고 싶었는데, 그러려면 구글 플랫폼에 결제 수단을 등록하고 어떤 권한 설정을 해야했다.
그런데 구글 플랫폼 설정 오류를 해결하지 못했다.

### 해결 방안 2 - 성공
그래서 직접 firebase에서 mysql로 데이터베이스를 이전하는 코드를 작성하기로 했다. 
mysql workbench에서는 데이터를 .csv 파일로 내보내기 할 수 있기 때문이다.

### 아쉬운 점
좀 더 똑똑한 방법으로 이 문제를 해결할 수 있었을텐데, 먼 길을 돌아온 것 같다.
시간 문제로 궁여지책을 짜낼 수밖에 없었는데, 다음엔 좀 더 나은 방법으로 해결하고 싶다.

### 앞으로의 계획

- 누구든 손쉽게 웹 심리 테스트를 만들어볼 수 있도록 기존의 [soul food 심리테스트](https://github.com/sunclock/soul_food)를 확장할 계획이다. 
- 1차 구현 사항은 질문, 선택지, 결과 이미지, 결과 설명, 결과 로직을 등록하는 admin 기능이다.
- 2차 구현 사항은 결과를 쉽게 볼 수 있도록 분석하는 admin 기능이다.
- 백엔드는 동일하게 firebase를 이용한다. 처음 프로그래밍을 하는 사람도 쉽게 사용할 수 있도록 하기 위함이다.
