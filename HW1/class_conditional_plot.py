import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt


n_bins = 10
breast = arff.loadarff(r'breast.w.arff')
df = pd.DataFrame(breast[0])
df.dropna(inplace=True)
i = df['Class'][0]
condition1 = df['Class'] == i
condition2 = df['Class'] != i
df1 = df.loc[condition1]
df2 = df.loc[condition2]


x1 = [df1.Clump_Thickness, df2.Clump_Thickness]
x2 = [df1.Cell_Size_Uniformity, df2.Cell_Size_Uniformity]
x3 = [df1.Cell_Shape_Uniformity, df2.Cell_Shape_Uniformity]
x4 = [df1.Marginal_Adhesion, df2.Marginal_Adhesion]
x5 = [df1.Single_Epi_Cell_Size, df2.Single_Epi_Cell_Size]
x6 = [df1.Bare_Nuclei, df2.Bare_Nuclei]
x7= [df1.Bland_Chromatin, df2.Bland_Chromatin]
x8 = [df1.Normal_Nucleoli, df2.Normal_Nucleoli]
x9 = [df1.Mitoses, df2.Mitoses]

fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)) = plt.subplots(nrows=3, ncols=3)

colors = ['tan', 'red']
names = ['benign','malignant']
ax0.hist(x1, n_bins, histtype='barstacked', color=colors, label=names)
ax0.legend(prop={'size': 5})
ax0.set_title('Clump_Thickness')

ax1.hist(x2, n_bins, histtype='barstacked', color = colors, label = names)
ax1.legend(prop={'size': 5})
ax1.set_title('Cell_Size_Uniformity')

ax2.hist(x2, n_bins, histtype='barstacked', color = colors, label = names)
ax2.legend(prop={'size': 5})
ax2.set_title('Marginal_Adhesion')

ax8.hist(x8, n_bins, histtype='barstacked', color = colors, label = names)
ax8.legend(prop={'size': 5})
ax8.set_title('Single_Epi_Cell_Size')

ax4.hist(x4, n_bins, histtype='barstacked', color = colors, label = names)
ax4.legend(prop={'size': 5})
ax4.set_title('Bare_Nuclei')

ax5.hist(x5, n_bins, histtype='barstacked', color = colors, label = names)
ax5.legend(prop={'size': 5})
ax5.set_title('Bland_Chromatin')

ax6.hist(x6, n_bins, histtype='barstacked', color = colors, label = names)
ax6.legend(prop={'size': 5})
ax6.set_title('Normal_Nucleoli')

ax7.hist(x7, n_bins, histtype='barstacked', color = colors, label = names)
ax7.legend(prop={'size': 5})
ax7.set_title('Mitoses')

ax3.hist(x3, n_bins, histtype='barstacked', color = colors, label = names)
ax3.legend(prop={'size': 5})
ax3.set_title('Cell_Shape_Uniformity')


fig.tight_layout()
plt.show()