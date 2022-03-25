#提取排行榜第一页的电影名，主演，导演
url='https://movie.douban.com/top250'
head={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36 Edg/98.0.1108.62"}
resp=rq.get(url,headers=head)
data=resp.text
import csv
obj=re.compile(r'<em class="">(?P<num>\d+)</em>.*?<img width="100" alt="(?P<name>.*?)" src.*?导演: (?P<director>.*?)&',re.S)#预加载正则表达式
lst=obj.finditer(data)#返回迭代器
f= open("top250.csv","w",encoding='utf-8',newline='')
csvwriter=csv.writer(f)
for i in lst:#提取迭代器内容
    dic=i.groupdict()
    csvwriter.writerow(dic.values())
    print(dic)