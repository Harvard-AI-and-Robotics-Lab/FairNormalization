import sys, os
import numpy as np
from scipy.stats import norm

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.legend_handler import HandlerLine2D
from matplotlib import rc
# rc('font', **{'family':'serif','serif':['Times New Roman'], 'size':12})
# rc('text', usetex=True)

import seaborn as sns

# https://seaborn.pydata.org/tutorial/color_palettes.html
sns.set_theme()
sns.set_style("darkgrid") # darkgrid
colors = sns.color_palette("husl", 8) # , n_colors=10
colors.reverse()



plt.figure(figsize=(12,7))
ax = plt.subplot()

y_loc = .1
output_svg = 'data4paper/gauss_dist_unit.pdf'

Mean = 0 # -1.3
SD = 1
x_axis = np.arange(-15+Mean, 15+Mean, 0.001)
plt.plot(x_axis, norm.pdf(x_axis,Mean,SD), color='red', linewidth=3, label=f'Unit Normal Distribution') # linestyle='dashed', 

# plt.plot(x_pos, np.zeros_like(x_pos) + val, 'o')
# plt.plot(x_neg, np.zeros_like(x_neg) + val, 'x')

# sns.set(font_scale=2)
# if x_range is not None:
# 	plt.xlim(x_range)

# bins = np.linspace(x_range[0], x_range[1], 100)

# if reversOrder:
# 	plt.hist(x_tp, bins, alpha=1, label='TP', color='red')
# 	plt.hist(x_fp, bins, alpha=1, label='FP', color='m')
# 	plt.hist(x_fn, bins, alpha=1, label='FN', color='c')
# 	plt.hist(x_tn, bins, alpha=1, label='TN', color='blue')
# else:
# 	plt.hist(x_tn, bins, alpha=1, label='TN', color='blue')
# 	plt.hist(x_fn, bins, alpha=1, label='FN', color='c')
# 	plt.hist(x_fp, bins, alpha=1, label='FP', color='m')
# 	plt.hist(x_tp, bins, alpha=1, label='TP', color='red')
# plt.axvline(neg_db, label='Neg/Pos Threshold', color='blue', linestyle='dashed', linewidth=3)
# # plt.axvline(pos_db, label='Positive threshold', color='red', linestyle='dashed', linewidth=3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(linestyle='--')

plt.ylabel('Probability Density', fontsize=38)
plt.xlabel(r'$z$', fontsize=45)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

loc = plticker.MultipleLocator(base=y_loc) # this locator puts ticks at regular intervals
ax.yaxis.set_major_locator(loc)
plt.xlim([-8, 9])
plt.ylim([0, .4])
# if show_legend:
# plt.legend(loc=0,fontsize=28)
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=28)

plt.savefig(output_svg, bbox_inches='tight', dpi=300)
plt.show()