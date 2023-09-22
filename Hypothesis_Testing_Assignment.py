#=================================================================================
"""
Soln.1: The manager of the F&B wants to differentiate between the diameter of 
two separate units. Hence I will consider using the Two Sample Z test to draw 
conclusions for the same. 

Null Hypothesis: mu_A = mu_B
Alternative Hypothesis: mu_A != mu_B
"""

import pandas as pd
df = pd.read_csv("Cutlets.csv")

from scipy import stats #Calculating z-val and p-val using the 2 sample Ztest 
zcal,pval = stats.ttest_ind(df["Unit A"],df["Unit B"])

print(pval) #0.472

if(pval<0.05):
    print("Null Hypothesis is rejected,Alternative Hypothesis is accepted.")
else:
    print("Null Hypothesis is accepted,Alternative Hypothesis is rejected.")
    
"""    
O/P:Null Hypothesis is accepted,Alternative Hypothesis is rejected.

There is enough statistical significance to conclude that the diameter of the 
cutlet between two units are the same over the entire units A & B population.

"""
#=================================================================================
"""
Soln.2: The Hospital is determined to find whether there is a difference in the 
average of the Turn Around Time of the different Lab reports. Since we are to 
consider means of multiple(more than 2) samples, I will be using the ANOVA test
to draw my conclusions.

Null Hypothesis: mu_Lab1 = mu_Lab2 = mu_Lab3 = mu_Lab4
Alternative Hypothesis: mu_Lab1 != mu_Lab2 != mu_Lab3 != mu_Lab4 
"""

import pandas as pd 
df = pd.read_csv("LabTAT.csv")

df_long = pd.DataFrame() #Creatng new dataframe for storing in long format.

df_long = pd.melt(df) #Using melt function to convert from wide to long format.

df_long.columns = ["Labaratory","TAT"] #Naming columns of new dataframe.

from statsmodels.formula.api import ols
anova1 = ols("TAT~C(Labaratory)",data=df_long).fit() 

import statsmodels.api as sm 
table = sm.stats.anova_lm(anova1,type=1)

pv=float(table.iloc[0:1,4].values)

print(pv) # ~0.0000002

if(pv<0.05):
    print("Null Hypothesis is rejected,Alternative Hypothesis is accepted.")
else:
    print("Null Hypothesis is accepted,Alternative Hypothesis is rejected.")

"""
O/P: Null Hypothesis is rejected,Alternative Hypothesis is accepted.

There is enough statistical significance to conclude that the average TAT of 
the four Labaratories are not the same.

"""
#=================================================================================
"""
Soln.3: Sales of products in four different regions are tabulated for males and females
As I am asked to find out if the male-female buyer ratios across regions are similar to 
each other, I will be using the Chi-square test of independence to draw a conclusion.

Null Hypothesis: All proportions are equal.
Alternate Hypothesis: Not all proportions are equal. 
    
"""

import pandas as pd 
df = pd.read_csv("BuyerRatio.csv")

df.set_index("Observed Values",inplace=True)

from scipy.stats import chi2_contingency
chi2_stat,pval,dof,expected = chi2_contingency(df)

print("P-value:", pval) #0.66
print("Degrees of Freedom:", dof) #3

if (pval<0.05):
    print("Ho is rejected and H1 is accepted")
else:
    print("Ho is accepted and H1 is rejected")

"""
O/P: Null Hypothesis is accepted,Alternative Hypothesis is rejected.

There is enough statistical significance to conclude that the buyer ratios of both 
the genders are proportional across the regions 

"""
#=================================================================================
"""
Soln.4: Telecall uses 4 centres across the globe and the manager wants to check whether
the defective ratios vary by centre. Hence for this, I will be using the Chi-square test
to analyse and draw conclusions.

Null Hypothesis: All defective proportions are equal across centres.
Alternate Hypothesis: Not all defective proportions are equal across centres. 
       
"""

import pandas as pd
df=pd.read_csv("CostomerOrderForm.csv")
df

#Checking if the columns contain other values than "Error Free" and "Defective".
df["Phillippines"].value_counts()
df["Indonesia"].value_counts()
df["Malta"].value_counts()
df["India"].value_counts()

#Transforming the df in such a way that it supports the test we are about to conduct
contingency_table = pd.crosstab(df["Phillippines"], [df["Indonesia"], df["Malta"], df["India"]])

# Perform the chi-square test on the contingency table
from scipy.stats import chi2_contingency
chi2_stat,pval, dof, expected = chi2_contingency(contingency_table)

print("P-value:", pval) #0.68
print("Degrees of freedom:", dof) #5

if (pval < 0.05):
    print("Ho is rejected and H1 is accepted")
else:
    print("Ho is accepted and H1 is rejected")
    
"""
O/P: Null Hypothesis is accepted,Alternative Hypothesis is rejected.

There is enough statistical significance to conclude that the defective ratios of  
the centres are proportional across the globe 

"""
#=================================================================================
