import matplotlib.pyplot as plt
import matplotlib.cm
import seaborn as sns
from statannotations import Annotator
import numpy as np
import pandas as pd

df1 = pd.read_excel('glineargamma.xlsx')
df2 = pd.read_excel('blineargamma.xlsx')


dataframes = [df[['slopes']] for df in [df1,df2]]


combined_df = pd.concat(dataframes, axis=1)

combined_df.columns = ['slope' + str(i) for i in range(1, len(combined_df.columns) + 1)]


combined_df_melted = combined_df.melt(var_name='marker', value_name='value')



vplot= sns.violinplot(data=combined_df_melted, x='marker', y='value', inner='point')


plt.xlabel("marker")
plt.ylabel("score")
plt.title('gamma slopes')
plt.show() 