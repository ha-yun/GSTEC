# 7주차
## <1>, <2>
#### 김유빈 강사님
### NLP : natural language processing
- Papago, github의 copilot, .. Siri, Alexa, Google Assistant, 역사전(reverse-dictionary)

NLP는 단어 혹은 문서의 의미를 나타낼 수 있는 표현을 얻는 것에서부터 시작한다.  
가장 간단한 단어의 의미표현 방법으로는 one-hot vector가 있다.!  
- one hot vector로 단어의 의미를 표현하는 방법의 한계
    - 유사성을 알 수 없다.
    
Distributional semantics
      - 벡터 자체에 단어의 의미를 담자.
- Document Term Matrix(DTM) - count based  
    - 벡터 자체에 단어의 의미를 담는 방법의 예, 말뭉치로부터 DTM을 구축하고 단어벡터 or 문서벡터를 추출한다.
    - one-hot vector와는 달리 두 단어 or 두 문시 사이의 유사도를 정량적으로 측정할 수 있단느 점에서 의의를 갖는다.
- prediction based - Language models    
모델이 단어의 의미를 벡터에 담는 규칙을 스스로 학습하도록 해 dense vector를 얻자.
  P(next|context)를 최대화하는 방향으로 모델(주로 딥러닝)의 가중치를 최적화한다.
  - GPT3, BERT, Word2Vec...
    

- DTM(count-based), language Models(Prediction-based)는 모두 어떤 가정을 기반으로 고안된 의미표현 방법인가요?
  - 맥락을 통해 의미를 파악할 수 있다. (distributional semantics)

- Distributional Semantics 가정을 하는 의미 표현 방법론(count-based, prediction-based)의 한계 : 편향된 데이터를 학습,,

## Improving DTM with TFIDF
- Term-Frequency Inverse Document-Frequency
- DTM과 같은 단순 빈도수 기반 방법론의 문제 : 중요한 단어와 불필요한 단어를 구분하지 못한다.
- 문서를 대변하는 단어에(더 중요한) 가중치를 더 부여하는 방법 TFIDF weighting
- TFIDF(t,d) = TF(t, d) * IDF(t)
  - TF(t,d) = d에서 t가 출현하는 빈도
  - TFIDF의 값은 tf(d, t)에 비례한다.
  - 즉 d에 t가 많이 출현할수록 TFIDF의 값이 높아진다.

- DF(t) = 특정 단어 t가 등장한 문서의 수
  - 각 문서에 t가 몇번 등장했는지는 관심x, 오직 t가 등장한 문서의 수에만 관심을 가진다.

- IDF(t) = DF의 'Inverse', 즉 DF를 역수 취해준 것
  - 역수를 취하는 이유? 흔하지 않은 단어일수록 해당 문서를 대변한다고 가정하기 때문에..
  - Log를 씌우는 이유? Log를 씌우지 않으면, 흔하지 않은 단어에 부여되는 가중치가 너무 커져서
  - 1을 더하는 이유? DF(t)=0인 경우, division by zero를 방지하기 위해서.
- TFIDF의 값은 tf(d,t)에 비례한다.
- 즉 d에 t가 더 많이 출현할수록 TFIDF의 값이 높아진다.
- TFIDF의 값은 idf(t)에 비례한다.
- 즉 t가 흔하지 않은 단어일수록 TFIDF의 값이 높아진다.
- 결론적으로 t가 흔하지 않은 단어이면서, d에선 유독 많이 출현한다. -> d를 대변하는 단어일 가능성이 높다 -> TFIDF가 높다.

<hr>

## <3>

