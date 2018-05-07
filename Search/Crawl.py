#import urllib.request
#https://www.crummy.com/software/BeautifulSoup/bs4/doc/#find-all
from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "http://www.naver.com/"

page = urlopen(url).read()


soup = BeautifulSoup(page, "html.parser") #html.parser, lxml 파서도 있음
"""
"""
#print(soup.find_all(attrs={"class" : "newss"})) #class="newssa"
#print(soup.find_all(id=True)) #id라는게 있는 걸 모두 찾는다.
#re.compile("") => 정규 표현 사용할 때 사용
#print(soup.find_all("div", attrs={"newss"})) #div가 들어가면서 ="newss"인 거
#print(soup.find_all("a", class_="newssa")) #class 찾기