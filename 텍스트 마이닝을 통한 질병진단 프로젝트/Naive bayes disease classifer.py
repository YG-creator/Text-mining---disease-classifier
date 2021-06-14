import json,os,sys,re
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier
import nltk #단어토큰화, 품사구분
from nltk.corpus import stopwords #금지어
from nltk.tag import pos_tag    #품사
from nltk.stem import WordNetLemmatizer     #어근 추출
from nltk import Text 

def get_wordnet_pos(pos_tag):
    """
    펜 트리뱅크 품사표기법(pos_tag()의 반환형태)을 받아서 WordNetLemmatizer에서 사용하는 품사표기(v, a, n, r)로 변환하는 함수.
    [매개변수]
        pos_tag: pos_tag()가 반환하는 품사
    [반환값]
        문자열: 동사-"v", 명사-'n', 형용사-'a', 부사-'r', 그외-None
    """
    if pos_tag.startswith('V'):
        return 'v'
    elif pos_tag.startswith('N'):
        return 'n'
    elif pos_tag.startswith('J'):
        return 'a'
    elif pos_tag.startswith('R'):
        return 'r'
    else:
        return None

    
#텍스트 전처리 함수
def tokenize_text2(text):
    result = ''
    # 소문자로 모두 변환
    text = text.lower()
    # 문장 단위로 토큰화
    sentence_tokens = nltk.sent_tokenize(text) #[문장, 문장, 문장]

    #stopwords 조회
    stop_words = stopwords.words('english')
    stop_words.extend(['although','unless', 'may']) 

    #원형복원을 위해 객체생성
    lemm = WordNetLemmatizer()

    # 반환한 값들을 모아놓을 리스트
    word_token_list = []

    # 문장단위로 처리
    for sentence in sentence_tokens:
        # word 토큰 생성
        word_tokens = nltk.regexp_tokenize(sentence, r'[A-Za-z]+')
        # 불용어(stopword)들 제거
        word_tokens = [word for word in word_tokens if word not in stop_words]

        #Stemming
        stemmer = nltk.stem.SnowballStemmer('english')
        word_tokens = [stemmer.stem(word) for word in word_tokens]

        # 원형 복원
        # 1. 품사부착
        word_tokens = pos_tag(word_tokens)
        # 2. lemmatize()용 품사로 변환
        word_tokens = [(word, get_wordnet_pos(tag)) for word, tag  in word_tokens if get_wordnet_pos(tag)!=None]
        # 3. 원형복원
        word_tokens = [ lemm.lemmatize(word, pos=tag) for word, tag in word_tokens]

        for token in word_tokens :
            if token == word_tokens[-1] :
                result += token
            else :
                result += token + ' '
            
    return result

#증상 입력
symtoms = input("ex) symptomA symptomB symptomC : ")
symptoms = tokenize_text2(symtoms)

#setting
diseaseclassifier = Trainer(tokenizer) #분류기
with open("dataset.csv", "r") as file: #Dataset 열기
    for i in file: #각 line마다
       i = i.strip()    #\n 지우기
       lines = i.split(",") #CSV <DISEASE> <SYMPTOM> 나누기
       diseaseclassifier.train(lines[1],  lines[0]) #TRAINING
diseaseclassifier = Classifier(diseaseclassifier.data, tokenizer)
classification = diseaseclassifier.classify(symptoms) #분류기에 입력

#질병 출력
for i in range(5): #top5 
   print (classification[i]) 
