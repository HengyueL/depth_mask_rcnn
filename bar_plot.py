"""
This Script Produces the Bar Plot of experiment results
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.style.use('seaborn-paper')

# avoid to use Type 3 font which violates IEEE PaperPlaza
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#################################################

# Bar related parames
alpha_ = 0.7
width = 0.16

# Experiment Name and Legend shown on the plot
appr = ['simulation', 'masked_sim', 'real', 'masked_real']
appr_legend = {'simulation': 'SIM (RAW)',
               'masked_sim': 'SIM (MASKED)',
               'real': 'REAL (RAW)',
               'masked_real': 'REAL (MASKED)'}

# Here add the experiment result o plot
objects = {}
objects['DQN'] = {
    'simulation': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'masked_sim': [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'real': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'masked_real': [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
}
objects['DQN-init'] = {
    'simulation': [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
    'masked_sim': [7,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'real': [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'masked_real': [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
}
objects['TO-DQN'] = {
    'simulation': [1,1,1,1,1,1,1,1,1,1,1,0,0,0,1],
    'masked_sim': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    'real': [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
    'masked_real': [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
}

import collections
objects = collections.OrderedDict(objects.items())


def evaluate(obj):
    """
    Calculat task successful rate
    """
    acc = {}
    for a in appr:
        acc[a] = []
    # for each object
    for o in obj:
        print(o)
        for a in appr:
            print(a)
            obj[o][a + '_acc'] = np.mean(obj[o][a], dtype=np.float64)*100 # accuracy in percent
            acc[a].append(obj[o][a + '_acc'])
    # for all objects
    # for a in appr:
    #     acc[a].append(np.mean(acc[a]))
    return acc

# Test evaluate fuction
acc = evaluate(objects)
print(acc)

# Initialize the figure handle
plt.figure(num=None, figsize=(5, 1.5), facecolor='w', edgecolor='k')
pos = list(range(len(objects)))  # the last one is for all objects
# print(pos)
for i, a in enumerate(appr):
    print(i,a)
    plt.bar([p + width*i for p in pos], acc[a], width, alpha=alpha_)
plt.ylabel(r'\small{\textbf{Task Success Rate}} (\%)')
obj_labels = list(objects.keys())
for idx, ol in enumerate(obj_labels):
    obj_labels[idx] = r'\small{\textbf{%s}}' % ol
plt.xticks([p + 1.5 * width for p in pos], obj_labels)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*5)
plt.ylim([0, 100])

# Some other baselines (parallel line references)
x_data = np.linspace(min(pos)-width, max(pos)+width*5)
y_data = 83.3 * np.ones_like(x_data)
p1 = plt.plot(x_data, y_data, 'm-.', label='HUMAN')

y_random = 10 * np.ones_like(x_data)
p2 = plt.plot(x_data, y_random, 'c-.', label='RANDOM')

# Adding the legend and showing the plot
lgd = plt.legend(['HUMAN', 'RANDOM'] + [appr_legend[x] for x in appr], bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
plt.grid(True)
plt.subplots_adjust(left=0.055)

# Either Show the plot for debugging or save it to file
# plt.show()
plt.savefig('figure_save.pdf', format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
