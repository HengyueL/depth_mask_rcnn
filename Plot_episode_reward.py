# This Script Plots the learning curve based on the training logs
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.style.use('seaborn-paper')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def main(root_dir):
    fig_0 = plt.figure(0, facecolor='w', edgecolor='k')
    ax = fig_0.add_subplot(1, 1, 1)

    name_list = ['KIDQN']
    rot_list = [4,7,10, 19]
    for rot_idx in rot_list:
        for name in name_list:
            case_reward_log = []
            for case in range(1, 3, 1):
                base_dir = root_dir + str(case)
                if name == 'KIDQN':
                    plot_name = 'TO-DQN'
                else:
                    plot_name = name
                log_dir = os.path.join(base_dir,
                                       '%s/rot%d' % (name, rot_idx))
                grasp_success_dir = os.path.join(log_dir,
                                                 'transitions/grasp-success')
                # grasp_success_lenth = len([f for f in os.listdir(grasp_success_dir)])
                grasp_success_lenth = 3300
                reward_log = []
                avg_reward_log = []
                x_axis = []
                action_count = 0
                for i in range(grasp_success_lenth):
                    if i < 5000:
                        data = np.load(os.path.join(grasp_success_dir,
                                                    '%d.npy' % i))
                        if data > 0.5:
                            reward_log.append(1)
                        else:
                            reward_log.append(0)
                        action_count += 1
                        if action_count > 100:
                            avg_reward_log.append(np.sum(reward_log) / action_count)
                            x_axis.append(action_count)
                case_reward_log.append(avg_reward_log)
            case_reward_log = np.asarray(case_reward_log)
            reward_mean = np.mean(case_reward_log, axis=0)
            reward_std = np.std(case_reward_log, axis=0)
            ax.plot(x_axis,
                    reward_mean,
                    label=r'\small{%s $%d^{\circ}$}' % (plot_name, (90 / (rot_idx - 1))))
            ax.fill_between(x_axis,
                            reward_mean-reward_std,
                            reward_mean+reward_std,
                            alpha=0.2)

    name_list = ['DQN_init', 'DQN']
    rot_list = [19]
    case_list = [1, 3]
    for rot_idx in rot_list:
        for name in name_list:
            case_reward_log = []
            for case in case_list:
                base_dir = root_dir + str(case)
                if name == 'KIDQN':
                    plot_name = 'TO-DQN'
                elif name == 'DQN_init':
                    plot_name = 'DQN-init'
                else:
                    plot_name = name
                log_dir = os.path.join(base_dir,
                                       '%s/rot%d' % (name, rot_idx))
                grasp_success_dir = os.path.join(log_dir,
                                                 'transitions/grasp-success')
                grasp_success_lenth = 3300
                reward_log = []
                avg_reward_log = []
                x_axis = []
                action_count = 0
                for i in range(grasp_success_lenth):
                    if i < 5000:
                        data = np.load(os.path.join(grasp_success_dir,
                                                    '%d.npy' % i))
                        if data > 0.5:
                            reward_log.append(1)
                        else:
                            reward_log.append(0)
                        action_count += 1
                        if action_count > 100:
                            avg_reward_log.append(np.sum(reward_log) / action_count)
                            x_axis.append(action_count)
                case_reward_log.append(avg_reward_log)
            case_reward_log = np.asarray(case_reward_log)
            reward_mean = np.mean(case_reward_log, axis=0)
            reward_std = np.std(case_reward_log, axis=0)
            ax.plot(x_axis,
                    reward_mean,
                    label=r'\small{%s $%d^{\circ}$}' % (plot_name, (90 / (rot_idx - 1))))
            ax.fill_between(x_axis,
                            reward_mean-reward_std,
                            reward_mean+reward_std,
                            alpha=0.2)
    # lgd = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    ax.legend(loc=4)
    ax.grid()
    ax.set_ylim([0, 0.185])
    ax.set_xlim([0, 3300])
    ax.set_xlabel(r'On-policy action number', fontsize=12)
    ax.set_ylabel(r'Avg reward per action', fontsize=12)
    plt.show()
    # plt.savefig(fig_0,
    #             'progress222.pdf',
    #             format='pdf')
    print()
    plt.close(fig_0)
    # plt.close(fig_1)
#


if __name__ == '__main__':
    root_dir = 'logs/train'
    main(root_dir)