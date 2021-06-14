#chromedriver.exe 필요
from bs4 import BeautifulSoup #웹크롤링
import re #웹크롤링
import requests #웹크롤링
from selenium import webdriver #구글 웹크롤링
import pandas as pd #행열로 이루어진 데이터 객체 만듦
import csv #dataset 작성때 필요
from selenium.common.exceptions import InvalidSessionIdException #에러처리

disease_list=[] # 질병명 리스트
not_include = [] # 증상을 찾지 못한 것들 error 찾는 용도

# 구글에서 질병에 대한 증상 웹크롤링
def get_symptoms(disease):   
    global symptoms_list
    symptoms_list= []
    url = 'https://www.google.com/search?q=symptoms of ' + disease
    driver = webdriver.Chrome()
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html,"html.parser")
    ul = soup.find("ul",{"class":"i8Z77e"})
    if ul == None:
        driver.close()
    else :
        lis = ul.findAll("li",{"class":"TrT0Xe"})
        for i in lis:
            symptoms_list.append(i.text)
        driver.close()
        
# widipiedia에서 질병명 가져오기
def get_disease_names():    
    #for alpha in alpha_list:
    url = 'https://simple.wikipedia.org/wiki/List_of_diseases'
    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html,"html.parser")
    ul = soup.find("div",{"class":"mw-parser-output"})
    lis = ul.findAll("li")
    for i in lis:
        disease_list.append(i.text)

get_disease_names() # 질병명 가져오기

# data set 작성 (1열=질병명,2열=증상)
f = open('test.csv','w',newline='',encoding='UTF-8')
wr = csv.writer(f)
for disease in disease_list[24:]:  #질병명
    get_symptoms(disease)  #증상 리스트
    if symptoms_list == []:
        not_include.append(disease)
        continue
    for symptom in symptoms_list:
        wr.writerow([disease,symptom])

# data set에 작성 안된 질병들
f = open('test_dataset.csv','w',newline='',encoding='UTF-8')
wr = csv.writer(f)
for n_in in not_include:
    wr.writerow([n_in])

f.close()
