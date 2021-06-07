import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
import csv
import time
import random

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

day = 1

alpha = 0.9956
beta_s = 0.99
beta_b = 1.01

scale = 0.1

p_SOH = 1

p_gb = np.ones(24*day)

q_max = 120
q_min = 0

x_max = 40
x_min = 40

max_iter = 200


# In[2]:


def get_PAR(x, xp, p):
    if x is None:
        x_act = np.zeros(24*day)
    else:
        x_act = np.sum(x[2], axis=0)
    if xp is None:
        x_pass = np.zeros(24*day)
    else:
        x_pass = np.sum(xp[2], axis=0)

    return np.max(x_act + x_pass)/np.average(x_act + x_pass)

def get_PAR_list(exp):
    return get_PAR(exp[0], exp[1], exp[2])
def get_EC(x, xp, p):
    return active_users_energy_cost_ret(x, xp, p)# + passive_users_energy_cost_ret(x, xp, p)

def get_total_EC(x, xp, p):
    return active_users_energy_cost_ret(x, xp, p) + passive_users_energy_cost_ret(x, xp, p)

def get_EC_list(exp):
    return get_EC(exp[0], exp[1], exp[2])
def get_SOH(x, xp, p, soh_coef=p_SOH):
    return users_soh_cost_ret(x, xp, p, soh_coef)
def get_SOH_list(exp, soh_coef=p_SOH):
    return get_SOH(exp[0], exp[1], exp[2], soh_coef)
def get_avg_SOH(x, xp, p, soh_coef=p_SOH):
    if (x is not None):
        return users_soh_cost_ret(x, xp, p, soh_coef) / len(x[0])
    else:
        return 0
def get_avg_SOH_list(exp, soh_coef=p_SOH):
    return get_avg_SOH(exp[0], exp[1], exp[2], soh_coef)
def get_TR(x, xp, p):
    return users_transaction_cost_ret(x, xp, p)
def get_TR_list(exp):
    return get_TR(exp[0], exp[1], exp[2])
def get_avg_active_users_utility(exp):
    return average_active_users_utility_ret(exp[0], exp[1], exp[2])
def users_transaction_sell_cost_ret(x, xp, p):
    if x is None:
        return 0
    else:
        active_users = len(x[0][:, 0])
    tr_sell = 0
    for i in range(active_users):
        tr_sell += np.sum(np.multiply(p[0][i, :], x[0][i, :]))
    return tr_sell
def users_transaction_buy_cost_ret(x, xp, p):
    if x is None:
        return 0
    else:
        active_users = len(x[0][:, 0])
    tr_buy = 0
    for i in range(active_users):
        tr_buy += np.sum(np.multiply(p[1][i, :], x[1][i, :]))
    return tr_buy
def users_transaction_cost_ret(x, xp, p):
    if x is None:
        active_users = 0
    else:
        active_users = len(x[0][:, 0])

    tr_cost = 0
    for i in range(active_users):
        tr_cost += np.sum(np.multiply(p[1][i, :], x[1][i, :]))
        tr_cost -= np.sum(np.multiply(p[0][i, :], x[0][i, :]))
    return tr_cost
def active_users_energy_cost_ret(x, xp, p):
    if x is None:
        return 0

    if xp is None:
        passive_load = np.zeros(24*day)
    else:
        passive_load = np.sum(xp[2], axis=0)
    return np.sum(np.multiply(np.sum(x[2], axis=0), np.sum(x[2], axis=0) + passive_load))

def average_active_users_energy_cost_ret(x, xp, p):
    if x is None:
        return 0
    else:
        active_users = len(x[0][:, 0])

    return active_users_energy_cost_ret(x, xp, p) / active_users
def passive_users_energy_cost_ret(x, xp, p):
    if xp is None:
        return 0

    if x is None:
        active_load = np.zeros(24 * day)
    else:
        active_load = np.sum(x[2], axis=0)
    return np.sum(np.multiply(np.sum(xp[2], axis=0), np.sum(xp[2], axis=0) + active_load))

def average_passive_users_energy_cost_ret(x, xp, p):
    if xp is None:
        return 0
    else:
        passive_users = int(len(xp[0][:, 0]))

    return passive_users_energy_cost_ret(x, xp, p) / passive_users
def users_soh_cost_ret(x, xp, p, soh_coef=p_SOH):
    if x is None:
        return 0
    else:
        active_users = len(x[0][:, 0])

    soh_cost = soh_coef * (np.sum(np.power(np.sum(x[0], axis=0), 2)) + np.sum(np.power(np.sum(x[1], axis=0), 2)))
    return soh_cost
def average_users_soh_cost_ret(x, xp, p, soh_coef=p_SOH):
    if x is None:
        return 0
    else:
        active_users = len(x[0][:, 0])

    return users_soh_cost_ret(x, xp, p, soh_coef) / active_users
def total_active_users_utility_ret(x, xp, p, soh_coef=p_SOH):
    return - users_transaction_cost_ret(x, xp, p) - active_users_energy_cost_ret(x, xp, p) - users_soh_cost_ret(x, xp,
                                                                                                                p,
                                                                                                                soh_coef)
def average_active_users_utility_ret(x, xp, p):
    if x is None:
        return 0
    else:
        active_users = len(x[0][:, 0])
        return total_active_users_utility_ret(x, xp, p) / active_users
def total_passive_users_utility_ret(x, xp, p):
    return -passive_users_energy_cost_ret(x, xp, p)
def average_passive_users_utility_ret(x, xp, p):
    if xp is None:
        return 0
    else:
        passive_users = len(xp[0][:, 0])
        return total_passive_users_utility_ret(x, xp, p) / passive_users
def user_utility(x, xp, p, soh_coef=p_SOH):
    if x is None:
        return np.zeros()
    else:
        active_users = len(x[0][:, 0])
    if xp is None:
        passive_users = 0
    else:
        passive_users = len(xp[0][:, 0])

    u = np.zeros(len(x[0][:, 0]))
    for i in range(len(x[0][:, 0])):
        tr_cost = np.sum(np.multiply(p[1][i, :], x[1][i, :])) - np.sum(np.multiply(p[0][i, :], x[0][i, :]))
        gb_sum = np.zeros((1, 24 * day))
        if (active_users != 0):
            gb_sum += np.sum(x[2], axis=0)
        if (passive_users != 0):
            gb_sum += np.sum(xp[2], axis=0)
        ec = np.sum(np.multiply(p_gb, np.multiply(gb_sum, x[2][i, :])))

        if (active_users != 0):
            soh_cost = soh_coef * (np.sum(np.multiply(x[0][i], np.sum(x[0], axis=0))) + np.sum(
                np.multiply(x[1][i], np.sum(x[1], axis=0))))
        else:
            soh_cost = 0
        u[i] = - tr_cost - ec - soh_cost
    return u
def passive_user_utility(x, xp, p, soh_coef=p_SOH):
    if x is None:
        active_users = 0
    else:
        active_users = len(x[0][:, 0])
    if xp is None:
        return np.zeros(3)
    else:
        passive_users = len(xp[0][:, 0])

    u = np.zeros(len(xp[0][:, 0]))
    for i in range(len(xp[0][:, 0])):
        # tr_cost = np.sum(np.multiply(p[1][i, :], x[1][i, :])) - np.sum(np.multiply(p[0][i, :], x[0][i, :]))
        gb_sum = np.zeros((1, 24 * day))
        if (active_users != 0):
            gb_sum += np.sum(x[2], axis=0)
        if (passive_users != 0):
            gb_sum += np.sum(xp[2], axis=0)
        ec = np.sum(np.multiply(p_gb, np.multiply(gb_sum, xp[2][i, :])))

        u[i] = - ec

    return u
def operator_electric_cost_ret(x, xp, p):
    if x is None:
        active_users = 0
    else:
        active_users = len(x[0][:, 0])

    if xp is None:
        passive_users = 0
    else:
        passive_users = len(xp[0][:, 0])

    if (passive_users != 0):
        gb_passive = np.sum(xp[2], axis=0)
    else:
        gb_passive = np.zeros(24 * day)
    if (active_users != 0):
        gb_active = np.sum(x[2], axis=0)
    else:
        gb_active = np.zeros(24 * day)
    generate = gb_active + gb_passive

    return np.sum(np.multiply(p_gb, np.power(generate, 2)))
def operator_tax_cost_ret(x, xp, p):
    if x is None:
        active_users = 0
    else:
        active_users = len(x[0][:, 0])
    if (active_users != 0):
        return scale * np.sum(np.power(p[0, :], 2) + np.power(p[1, :], 2))
    else:
        return 0
def operator_utility_ret(x, xp, p):
    return - operator_electric_cost_ret(x, xp, p) - operator_tax_cost_ret(x, xp, p)
def state_of_charge(x, xp, p):
    if x is None:
        active_users = 0
        q = np.zeros(24 * day + 1)
        q[0] = q_min
        for t in range(1, 24 * day + 1):
            q[t] = alpha * q[t - 1]
        return q
    else:
        active_users = int(len(x[:, 0]) / 4)

    q = np.zeros(24 * day + 1)
    q[0] = q_min
    for t in range(1, 24 * day + 1):
        q[t] = alpha * q[t - 1] + beta_b * np.sum(x[:active_users, t - 1]) - beta_s * np.sum(
            x[active_users:2 * active_users, t - 1])
    return q
def ess_selling_state(x, xp, p):
    if x is None:
        return np.zeros(24 * day)
    else:
        active_users = int(len(x[:, 0]) / 4)

    return np.sum(x[:active_users, :], axis=0)
def ess_buying_state(x, xp, p):
    if x is None:
        return np.zeros(24 * day)
    else:
        active_users = int(len(x[:, 0]) / 4)

    return np.sum(x[active_users:2 * active_users, :], axis=0)
def grid_selling_state(x, xp, p):
    if x is None:
        return np.zeros(24 * day)
    else:
        active_users = int(len(x[:, 0]) / 4)

    return np.sum(x[2 * active_users:3 * active_users, :], axis=0)
def grid_buying_state(x, xp, p):
    if x is None:
        return np.zeros(24 * day)
    else:
        active_users = int(len(x[:, 0]) / 4)

    return np.sum(x[3 * active_users:, :], axis=0)
def passive_grid_selling_state(x, xp, p):
    if xp is None:
        return np.zeros(24 * day)
    else:
        passive_users = int(len(xp[:, 0]) / 4)

    return np.sum(xp[2 * passive_users:3 * passive_users, :], axis=0)
def passive_grid_buying_state(x, xp, p):
    if x is None:
        return np.zeros(24 * day)
    else:
        passive_users = int(len(xp[:, 0]) / 4)

    return np.sum(xp[3 * passive_users:, :], axis=0)
def get_PAR_exps(result):
    return np.mean(list(map(get_PAR_list, result)))
def get_EC_exps(result):
    return np.mean(list(map(get_EC_list, result)))
def get_SOH_exps(result):
    return np.mean(list(map(get_SOH_list, result)))
def get_avg_SOH_exps(result):
    return np.mean(list(map(get_avg_SOH_list, result)))
def get_TR_exps(result):
    return np.mean(list(map(get_TR_list, result)))