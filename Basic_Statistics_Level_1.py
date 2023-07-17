#==============================================================================

#Soln 9.a:
import pandas as pd 
df = pd.read_csv("Q9_a.csv")

df["speed"].skew() #Skew of Car Speed

df["speed"].kurt() #Kurtosis of Car Speed 

df["dist"].skew()  #Skew of Car Distance

df["dist"].kurt()  #Kurtosis of Car Distance 

#==============================================================================

#Soln 9.b:
import pandas as pd 
df = pd.read_csv("Q9_b.csv")

df["SP"].skew() #Skew of SP

df["SP"].kurt() #Kurtosis of SP 

df["WT"].skew()  #Skew of Car WT

df["WT"].kurt()  #Kurtosis of WT 

#==============================================================================

#Soln 11
from scipy import stats 

#94% confidence levels

a = stats.norm.interval(0.94,loc=200,scale=30)

#98% confidence levels

b = stats.norm.interval(0.98,loc=200,scale=30)

#96% confidence levels,i.e, alpha=1%

c = stats.norm.interval(0.96,loc=200,scale=30)

#==============================================================================

#Soln 20
import pandas as pd 
df = pd.read_csv("Cars.csv")

from scipy.stats import norm
nd=norm(df["MPG"].mean(),df["MPG"].std())
        #Mean of MPG,Standard dev. of MPG

p1=1-nd.cdf(38) #P(MPG>38)

p2=nd.cdf(39.99) #P(MPG<40)

p3=nd.cdf(50)-nd.cdf(20) #P(20<MPG<50)

#==============================================================================

#Soln 22
import scipy.stats as stats

#for 90% confidence interval
stats.norm.ppf(0.90).round(3) 

#for 94% confidence interval
stats.norm.ppf(.94).round(3)

#for 60% confidence interval
stats.norm.ppf(0.6).round(3)

#==============================================================================

#Soln 23
import scipy.stats as stats

#for 95% confidence interval
stats.t.ppf(0.95,24).round(3) #df=25-1=24

#for 96% confidence interval
stats.t.ppf(0.96,24).round(3)

#for 99% confidence interval
stats.t.ppf(0.99,24).round(3)

#==============================================================================

#Soln 21.a
import pandas as pd 
df=pd.read_csv("Cars.csv")

#normality test 
#Ho: Data is normal 
#H1: Data is not normal 

from scipy.stats import shapiro 

calc,p=shapiro(df["MPG"])

alpha=0.05

if(p<alpha):
    print("Ho is rejected and H1 is accepted.")
else:
    print("Ho is accepted and H1 is rejected.")

#Ho is accepted.

#==============================================================================

#Soln 21.b
import pandas as pd 
df=pd.read_csv("wc-at.csv")

#normality test 
#Ho: Data is normal 
#H1: Data is not normal 

from scipy.stats import shapiro 

calc,p=shapiro(df["AT"])

alpha=0.05
if(p<alpha):
    print("Ho is rejected and H1 is accepted.")
else:
    print("Ho is accepted and H1 is rejected.")
#H1 is accepted

#==============================================================================

#Soln 24
import scipy.stats as stats
import numpy as np

(260-270)/(90/np.sqrt(18)) #t=(xbar-mu)/(SD/sqrt(n))

stats.t.cdf(-0.471,17) #P(X<=260)


df = pd.read_csv("Q7.csv")

df["Weigh"].mode()

#==============================================================================