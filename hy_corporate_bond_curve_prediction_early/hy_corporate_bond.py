import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('D:/UIUC_courses/IE517/IE517_FY21_HW3/HY_Universe_corporate_bond.csv')
print(df.describe())

df=df.values


df[:,14]=df[:,14].astype(str)
df[:,12]=df[:,12].astype(str)
df[:,5:9]=df[:,5:9].astype(str)
df[:,15]=df[:,15].astype(np.float)
score=np.zeros(len(df[:,1]))
avg_score=np.zeros(len(df[:,1]))
count=np.zeros(len(df[:,1]))
for i in range(len(df[:,1])):
    
    for j in range(4):
        
        if df[i,j+5][0]=='A':
            df[i,j+5]=2
            count[i]=count[i]+1
            score[i]=score[i]+2;
            
        elif df[i,j+5][0]=='B':
            df[i,j+5]=1
            count[i]=count[i]+1
            score[i]=score[i]+1;
            
            
        elif df[i,j+5][0]=='C':
            df[i,j+5]=0
            count[i]=count[i]+1
            score[i]=score[i]+0;
            
        else:
            df[i,j+5]='NaN'
            
        
        avg_score[i]=(score[i])/count[i]
        
        
volume_traded_score_raw=np.transpose(np.array([avg_score,df[:,22]]))
volume_traded_score=[]
#volume_traded_score1=volume_traded_score[:, ~np.isnan(volume_traded_score[:,0].astype(np.float))]
for i in range(len(df[:,1])):
    if ~np.isnan(avg_score[i]):
        volume_traded_score.append([round(avg_score[i],2)*5,df[i,22]])
volume_traded_score_np=np.array(volume_traded_score)
volume_traded_score_df=pd.DataFrame(np.array(volume_traded_score),columns=['score','volume_traded'])
sns.scatterplot(data=volume_traded_score_df,x="volume_traded",y="score")
plt.xlabel("volume_traded")
plt.ylabel("Average rating between 0 and 10")
plt.show()

coupon_score_raw=np.transpose(np.array([avg_score,df[:,9]]))
coupon_score=[]
#coupon_score1=coupon_score[:, ~np.isnan(coupon_score[:,0].astype(np.float))]
for i in range(len(df[:,1])):
    if ~np.isnan(avg_score[i]):
        coupon_score.append([round(avg_score[i],2)*5,df[i,9]])
coupon_score_np=np.array(coupon_score)
coupon_score_df=pd.DataFrame(np.array(coupon_score),columns=['score','coupon'])
sns.scatterplot(data=coupon_score_df,x="coupon",y="score")
plt.xlabel("coupon")
plt.ylabel("Average rating between 0 and 10")
plt.show()
             
In_ETF_score_raw=np.transpose(np.array([avg_score,df[:,19]]))
In_ETF_score=[]
#In_ETF_score1=In_ETF_score[:, ~np.isnan(In_ETF_score[:,0].astype(np.float))]
for i in range(len(df[:,1])):
    if ~np.isnan(avg_score[i]):
        In_ETF_score.append([round(avg_score[i]*5,2),df[i,19]])
In_ETF_score_np=np.array(In_ETF_score)
In_ETF_score_df=pd.DataFrame(np.array(In_ETF_score),columns=['score','In_ETF'])
sns.swarmplot(data=In_ETF_score_df,x="In_ETF",y="score")
plt.xlabel("In_ETF")
plt.ylabel("Average rating between 0 and 10")
plt.show()

sector_score_raw=np.transpose(np.array([avg_score,df[:,14]]))
sector_score=[]
#sector_score1=sector_score[:, ~np.isnan(sector_score[:,0].astype(np.float))]
for i in range(len(df[:,1])):
    if ~np.isnan(avg_score[i]):
        sector_score.append([round(avg_score[i]*5,2),df[i,14]])
sector_score_np=np.array(sector_score)
sector_score_df=pd.DataFrame(np.array(sector_score),columns=['score','sector'])
g = sns.FacetGrid(sector_score_df, col="sector",col_wrap=8)
g.map(sns.histplot, "score")

plt.xlabel("sector")
plt.ylabel("Average rating between 0 and 10")
plt.show()