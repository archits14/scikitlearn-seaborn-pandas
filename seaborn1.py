import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Advertising.csv', index_col=0)
# sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7, kind='reg')
# plt.show()

