#!/usr/bin/env python
# coding: utf-8

# In[306]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[197]:


import os
# 운영체제별 한글 폰트 설정
if os.name == 'posix': # Mac 환경 폰트 설정
    plt.rc('font', family='AppleGothic')
elif os.name == 'nt': # Windows 환경 폰트 설정
    plt.rc('font', family='Malgun Gothic')

plt.rc('axes', unicode_minus=False) # 마이너스 폰트 설정


# 글씨 선명하게 출력하는 설정
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[389]:


df = pd.read_csv('./data/users.csv')


# In[390]:


df.head(1)


# In[391]:


df.info()


# In[392]:


df = df.loc[:, ['age', 'job', 'sex', 'response']]


# In[393]:


df = df.dropna(how="any")


# In[394]:


strRes = df['response'].apply(str).to_frame()


# In[395]:


strRes = strRes.response.apply(lambda x: x.zfill(16)).to_frame()


# In[396]:


strRes.head(1)


# In[409]:


df["response"] = strRes["response"]


# In[410]:


df = df.loc[:, ['age', 'job', 'sex', 'response']]


# In[411]:


df.info()


# In[412]:


df.head()


# In[413]:


# 3 7 11 14
age = df.groupby(df['age'])
age.size()


# In[414]:


sex = df.groupby(df['sex'])
sex.size()


# In[415]:


job = df.groupby(df['job'])
job.size()


# In[417]:


df['서비스-테이블 간격 및 칸막이'] = df['response'].str[0] # 테이블 간격 1번
df['서비스-조용한 분위기'] = df['response'].str[1] # 소음 2번
df['음식-음식물 재사용 X'] = df['response'].str[3] # 음식물 재사용 4번
# df['시설-식당 청결'] = df['response'].str[4] # 조리과정/조리시설 청결 5번
df['서비스-종업원 청결도'] = df['response'].str[5] # 조리종사자 청결 6번
df['시설-손 소독'] = df['response'].str[7] # 손소독제 8번
df['시설-홀 정리 정돈'] = df['response'].str[8] # 홀 위생상태 9번
df['시설-음용 시설'] = df['response'].str[9] # 음용수 10번
df['서비스-환기'] = df['response'].str[11] # 환기시설 12번
df['시설-식당 청결'] = df['response'].str[12] # 식탁청결 13번
df['음식-뷔페 서비스 보호 덮개'] = df['response'].str[14] # 셀프서비스 위생 15번
df['음식-공용 집기류 위생'] = df['response'].str[15] # 냅킨통, 소스통, 수저통, 양념통, 집기 및 주변 청결 16번


# In[418]:


df.info()


# In[416]:


# 응답자 나이, 성별, 직업별 분포
all = df.groupby(['age', 'sex', 'job'])
all.size()


# In[421]:


stacked_bar_df = df.groupby(['sex', 'age']).size().unstack()
stacked_bar_df.plot(kind='bar')
plt.xticks(rotation=0, fontsize=13)
plt.title(label="응답자 성별-연령대 분포(total 180)")
plt.show()


# In[424]:


stacked_bar_df = df.groupby(['sex', 'job']).size().unstack()
stacked_bar_df.plot(kind='bar')
plt.xticks(rotation=0, fontsize=13)
plt.title(label="응답자 성별-직업 분포(total 180)")
plt.show()


# In[427]:


stacked_bar_df = df.groupby(['job', 'age']).size().unstack()
stacked_bar_df.plot(kind='bar')
plt.xticks(rotation=45)
plt.title(label="응답자 직업-나이 분포(total 180)")
plt.show()


# In[433]:


# 범주형 데이터 플롯 그리기
def drawCatPlot(col):
    df[col].value_counts().plot(kind='bar')
    plt.title(col+'(total 180)')
    plt.ylabel("응답자 수")
    plt.xlabel("고려 여부(1=신경 씀, 0=신경 안 씀)")
    plt.xticks(rotation=0)
    plt.savefig("1D "+ col+ "데이터 플롯.png")

cols = ['서비스-테이블 간격 및 칸막이', '서비스-조용한 분위기', 
       '서비스-종업원 청결도', '서비스-환기', '음식-음식물 재사용 X',
       '음식-뷔페 서비스 보호 덮개', '음식-공용 집기류 위생',
       '시설-손 소독', '시설-식당 청결', '시설-음용 시설', 
       '시설-홀 정리 정돈']


# In[434]:


for col in cols:
    drawCatPlot(col)


# In[461]:





# In[483]:


# 2D 범주형 그룹별 데이터 플롯 그리기
def draw2dCatPlot(col1, col2, title):
    stacked_bar_df = df.groupby([col1, col2]).size().unstack()
    stacked_bar_df.plot(kind='bar', stacked=True)
    plt.title(title+'(total 180)')
    plt.ylabel("응답자 수")
    plt.xlabel("고려 여부(0=신경 안 씀, 1=신경 씀)")
    plt.xticks(rotation=0)
    plt.rcParams["figure.figsize"] = (15,5) # 연령대, 응답, 직업 x축일 때
#    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"] # 성별 x축일 때

    plt.savefig("2D "+ "x축="+col1+" "+ title+" 데이터 플롯.png")


# In[469]:


# (연령대/응답) x축 = 연령대 
for col in cols:
    draw2dCatPlot('age', col, col)


# In[479]:


# (연령대/응답) x축 = 응답
for col in cols:
    draw2dCatPlot(col, 'age', col)


# In[482]:


# (성별/응답) x축 = 성별
for col in cols:
    draw2dCatPlot('sex', col, col)


# In[485]:


# (직업/응답) x축 = 직업
for col in cols:
    draw2dCatPlot('job', col, col)


# In[503]:


# 3D 범주형 그룹별 데이터 플롯 그리기
def draw3dCatPlot(col1, col2, col3, title):
    stacked_bar_df = df.groupby([col2, col3, col1]).size().unstack()
    stacked_bar_df.plot(kind='bar', stacked=True)
    plt.title(title+'(total 180) 고려 여부(0=신경 안 씀, 1=신경 씀)')
    plt.ylabel("응답자 수")
    plt.xlabel("고려 여부(0=신경 안 씀, 1=신경 씀)")
    plt.xticks(rotation=30)
    plt.rcParams["figure.figsize"] = (15,12) 
    plt.savefig("3D "+ title+" 데이터 플롯.png")


# In[504]:


for col in cols:
    draw3dCatPlot(col, 'sex', 'age', col)


# In[292]:


pip install dataframe-image


# In[298]:


import dataframe_image as dfi


# In[506]:


def group3DTable(col):
    hey = df.groupby([col, 'age', 'sex', 'job']).size().unstack(fill_value=0)
    dfi.export(hey, col+' 데이터프레임.png')
    


# In[507]:


for col in cols:
    group3DTable(col)


# In[ ]:




