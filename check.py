import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
import csv
import time
import random
from functions import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

total_user = 100  # max 100
day = 1  # max 5

pv_max = 1 / 3

alpha = 0.9956
beta_s = 0.99
beta_b = 1.01

tau_u = 0.1
tau_o = 0.1

ep = 0.00001
scale = 0.001

p_SOH = 1
solver = "ECOS"
p_gb_coef = 1

p_gb = p_gb_coef * np.ones(24*day)

E_PV = pv_max*np.load("E_PV.npy", allow_pickle=True)[:24*day]
load_123 = np.load("load_123.npy", allow_pickle=True)[:total_user, :24*day]
load_12 = np.load("load_12.npy", allow_pickle=True)[:total_user, :24*day]
load_13 = np.load("load_13.npy", allow_pickle=True)[:total_user, :24*day]
load_23 = np.load("load_23.npy", allow_pickle=True)[:total_user, :24*day]
#load_1 = np.load("load_1.npy", allow_pickle=True)[:total_user, :24*day]
load_error = np.load("load_error.npy", allow_pickle=True)[:total_user, :24*day]

load = load_123

q_max = 25
q_min = 0

x_max = 25
x_min = 25

#max_iter = 5000
max_iter = 2000
exp_nash = np.load("kkt_default_nash.npy", allow_pickle=True)

mult = 5
x_70, xp_70, p_70 = exp_nash[70]
a = -mult * p_70[1][0]+x_70[2][0]-p_SOH*x_70[1][0]
b = mult * p_70[0][0]-x_70[2][0]-p_SOH*x_70[0][0]
#p = np.random.uniform(0, 7, (24*day, 2))
#ps = np.tile(np.minimum(p[:,0], p[:,1]), (total_user, 1))
#pb = np.tile(np.maximum(p[:,0], p[:,1]), (total_user, 1))
#ps = np.minimum(p[:, :, 0], p[:, :, 1])
#pb = np.maximum(p[:, :, 0], p[:, :, 1])
#ps = np.tile(np.random.uniform(1, 3, 24*day), (total_user, 1))
#pb = np.tile(np.random.uniform(5, 7, 24*day), (total_user, 1))
def our_model(users, load, soh_coef=p_SOH, ps=None, pb=None, file_name = "default.npy"):
    if users != total_user:
        passive_load_matrix = load[users:]
        x_p = np.zeros((3, total_user - users, 24 * day))
        x_p[2] = np.maximum(passive_load_matrix - np.tile(E_PV, (total_user - users, 1)), np.zeros(24 * day))
    else:
        x_p = None

    if users != 0:
        load_matrix = load[:users]

        x = np.zeros((3, users, 24 * day, max_iter + 1))
        #x[:, :, :, 0] = x_70[:, :users, :]
        # initialize
        # ESS sell
        x[0][:, :, 0] = -np.minimum(load_matrix - np.tile(E_PV, (users, 1)), np.zeros(24 * day))

        # ESS buy
        # x[1] is already zeros

        # Grid buy
        x[2][:, :, 0] = np.maximum(load_matrix - np.tile(E_PV, (users, 1)), np.zeros(24 * day))
    else:
        x = None
        p = np.zeros((2, users, 24 * day))
        print("PAR", get_PAR(x, x_p, p))
        print("EC", get_EC(x, x_p, p))
        print("SOH", get_SOH(x, x_p, p, soh_coef))
        print("TR", get_TR(x, x_p, p))
        return x, x_p, p

    p = np.zeros((2, users, 24 * day, max_iter + 1))
    #p[:, :, :, 0] = mult * p_70[:, :users, :]
    if ps is not None:
        p[0][:, :, 0] = np.tile(ps, (users, 1))
    if pb is not None:
        p[1][:, :, 0] = np.tile(pb, (users, 1))

    c_1 = np.zeros((24 * day, max_iter + 1))
    c_2 = np.zeros((24 * day, max_iter + 1))
    #c_1[:, 0] = c1_70
    #c_2[:, 0] = c2_70
    for k in range(max_iter):

        # user

        c1 = cp.Variable((1, 24 * day))
        c2 = cp.Variable((1, 24 * day))
        # x_gb = cp.Variable((users, 24 * day))

        c1_mat = c1
        c2_mat = c2
        for i in range(users - 1):
            c1_mat = cp.vstack([c1_mat, c1])
            c2_mat = cp.vstack([c2_mat, c2])
        # x_s = (p[0][:, :, k] - p_gb_coef * x_gb - c2_mat)/soh_coef
        # x_b = (-p[1][:, :, k] + p_gb_coef * x_gb - c1_mat)/soh_coef

        a = (soh_coef + p_gb_coef) / (soh_coef * soh_coef + 2 * soh_coef * p_gb_coef)
        b = p_gb_coef / (soh_coef * soh_coef + 2 * soh_coef * p_gb_coef)
        c = p_gb_coef / (soh_coef + 2 * p_gb_coef)
        d = 1 / (soh_coef + 2 * p_gb_coef)
        e = soh_coef / (soh_coef + 2 * p_gb_coef)
        x_s = a * p[0][:, :, k] - b * p[1][:, :, k] - a * c2_mat - b * c1_mat - c * (
                    load_matrix - np.tile(E_PV, (users, 1)))
        x_b = b * p[0][:, :, k] - a * p[1][:, :, k] - b * c2_mat - a * c1_mat + c * (
                    load_matrix - np.tile(E_PV, (users, 1)))
        x_gb = d * p[0][:, :, k] + d * p[1][:, :, k] + d * c1_mat - d * c2_mat + e * (
                    load_matrix - np.tile(E_PV, (users, 1)))

        user_s = cp.sum(cp.multiply(p[0][:, :, k], x_s))
        user_b = cp.sum(cp.multiply(p[1][:, :, k], x_b))

        if users != total_user:
            passive_load = np.sum(x_p[2], axis=0)
        else:
            passive_load = np.zeros(24 * day)

        user_energy = p_gb_coef * cp.sum(
            cp.power(cp.sum(x_gb, axis=0), 2) + cp.multiply(cp.sum(x_gb, axis=0), passive_load))
        # user_energy = cp.sum(cp.multiply(cp.sum(x_gb, axis=0), cp.sum(x_gb, axis=0)+passive_load))

        user_ESS = soh_coef * cp.sum(cp.power(cp.sum(x_s, axis=0), 2) + cp.power(cp.sum(x_b, axis=0), 2))
        user_prox = tau_u * (cp.sum(cp.power(c1[0] - c_1[:, k], 2)) + cp.sum(cp.power(c2[0] - c_2[:, k], 2)))

        user_obj = user_s - user_b - user_energy - user_ESS - user_prox

        user_constraints = []
        # user_constraints += [x_gb == x_s - x_b + load_matrix - np.tile(E_PV, (users, 1))]
        user_constraints += [x_gb >= 0, x_s >= 0, x_b >= 0]

        ess_matrix = np.fromfunction(np.vectorize(lambda i, j: 0 if i < j else np.power(alpha, i - j)),
                                     (24 * day, 24 * day), dtype=float)
        q_ESS = q_min * np.fromfunction(np.vectorize(lambda i, j: np.power(alpha, i - j + 1)), (24 * day, 1),
                                        dtype=float) + beta_s * ess_matrix @ cp.sum(x_s,
                                                                                    axis=0).T - beta_b * ess_matrix @ cp.sum(
            x_b, axis=0).T

        user_constraints += [q_min <= q_ESS, q_ESS <= q_max]
        user_constraints += [cp.sum(x_s, axis=0) <= x_max]
        user_constraints += [cp.sum(x_b, axis=0) <= x_min]

        prob = cp.Problem(cp.Maximize(user_obj), user_constraints)
        result1 = prob.solve(solver=solver)

        x[0][:, :, k + 1] = x_s.value
        x[1][:, :, k + 1] = x_b.value
        x[2][:, :, k + 1] = x_gb.value
        c_1[:, k + 1] = c1.value
        c_2[:, k + 1] = c2.value

        # Operator
        p_s = cp.Variable((users, 24 * day))
        p_b = cp.Variable((users, 24 * day))

        a = (soh_coef + p_gb_coef) / (soh_coef * soh_coef + 2 * soh_coef * p_gb_coef)
        b = p_gb_coef / (soh_coef * soh_coef + 2 * soh_coef * p_gb_coef)
        c = p_gb_coef / (soh_coef + 2 * p_gb_coef)
        d = 1 / (soh_coef + 2 * p_gb_coef)
        e = soh_coef / (soh_coef + 2 * p_gb_coef)
        x_s = a * p_s - b * p_b - a * np.tile(c_2[:, k + 1], (users, 1)) - b * np.tile(c_1[:, k + 1],
                                                                                       (users, 1)) - c * (
                          load_matrix - np.tile(E_PV, (users, 1)))
        x_b = b * p_s - a * p_b - b * np.tile(c_2[:, k + 1], (users, 1)) - a * np.tile(c_1[:, k + 1],
                                                                                       (users, 1)) + c * (
                          load_matrix - np.tile(E_PV, (users, 1)))
        x_gb = d * p_s + d * p_b + d * np.tile(c_1[:, k + 1], (users, 1)) - d * np.tile(c_2[:, k + 1],
                                                                                        (users, 1)) + e * (
                           load_matrix - np.tile(E_PV, (users, 1)))

        if users != total_user:
            passive_load = np.sum(x_p[2], axis=0)
        else:
            passive_load = np.zeros(24 * day)

        operator_cost = cp.sum(cp.power(cp.sum(x_gb, axis=0) + passive_load, 2))
        operator_ESS = scale * cp.sum(cp.power(p_s, 2) + cp.power(p_b, 2))
        if k==0:
            operator_prox = 0
        else:
            operator_prox = tau_o * cp.sum(cp.power(p_s - p[0][:, :, k], 2) + cp.power(p_b - p[1][:, :, k], 2))

        operator_obj = - operator_cost - operator_ESS - operator_prox

        operator_constraints = []
        operator_constraints += [0 <= p_s, p_s <= p_b]
        operator_constraints += [x_gb >= 0, x_s >= 0, x_b >= 0]

        ess_matrix = np.fromfunction(np.vectorize(lambda i, j: 0 if i < j else np.power(alpha, i - j)),
                                     (24 * day, 24 * day), dtype=float)
        q_ESS = q_min * np.fromfunction(np.vectorize(lambda i, j: np.power(alpha, i - j + 1)), (24 * day, 1),
                                        dtype=float) + beta_s * ess_matrix @ cp.sum(x_s,
                                                                                    axis=0).T - beta_b * ess_matrix @ cp.sum(
            x_b, axis=0).T

        operator_constraints += [q_min <= q_ESS, q_ESS <= q_max]
        operator_constraints += [cp.sum(x_s, axis=0) <= x_max]
        operator_constraints += [cp.sum(x_b, axis=0) <= x_min]

        prob = cp.Problem(cp.Maximize(operator_obj), operator_constraints)
        result2 = prob.solve(solver=solver)
        # print(prob.status)
        p[0][:, :, k + 1] = p_s.value
        p[1][:, :, k + 1] = p_b.value
        x[0][:, :, k + 1] = x_s.value
        x[1][:, :, k + 1] = x_b.value
        x[2][:, :, k + 1] = x_gb.value

        pr = np.sum(
            np.power(np.abs(p[0][:, :, k] - p[0][:, :, k + 1]), 2) + np.power(np.abs(p[1][:, :, k] - p[1][:, :, k + 1]),
                                                                              2)) + np.sum(
            np.power(np.abs(c_1[:, k] - c_1[:, k + 1]), 2) + np.power(np.abs(c_2[:, k] - c_2[:, k + 1]), 2))

        np.save(file_name, [x[:, :, :, k + 1], x_p, p[:, :, :, k+1]])
        if k % 20 == 0:
            print("####################  Users : " + str(users) + "  Iter " + str(k + 1), "##################")
            print("PAR       :", get_PAR(x[:, :, :, k + 1], x_p, p[:, :, :, k+1]))
            print("EC        :", get_EC(x[:, :, :, k + 1], x_p, p[:, :, :, k+1]))
            print("Total EC  :", get_total_EC(x[:, :, :, k + 1], x_p, p[:, :, :, k+1]))
            print("SOH       :", get_SOH(x[:, :, :, k + 1], x_p, p[:, :, :, k+1], soh_coef))
            print("TR        :", get_TR(x[:, :, :, k + 1], x_p, p[:, :, :, k+1]))
            print("User obj  :", result1)
            print("Oper obj  :", result2)
            print("Oper tax  :", operator_ESS.value)
            print("Gap       :", pr)

        if k == 0 or k == 1:
            print("####################  Users : " + str(users) + "  Iter " + str(k + 1), "##################")
            print("s         :", user_s.value)
            print("b         :", user_b.value)
            print("ec        :", user_energy.value)
            print("soh       :", user_ESS.value)
            print("prox      :", user_prox.value)

            print("PAR       :", get_PAR(x[:, :, :, k + 1], x_p, p[:, :, :, k + 1]))
            print("EC        :", get_EC(x[:, :, :, k + 1], x_p, p[:, :, :, k + 1]))
            print("Total EC  :", get_total_EC(x[:, :, :, k + 1], x_p, p[:, :, :, k + 1]))
            print("SOH       :", get_SOH(x[:, :, :, k + 1], x_p, p[:, :, :, k + 1], soh_coef))
            print("User obj  :", result1)
            print("Oper obj  :", result2)
            print("Oper tax  :", operator_ESS.value)
            print("Gap       :", pr)

        if pr <= ep:
            break

    print("####################### CONVERGED #############################")
    print("PAR       :", get_PAR(x[:, :, :, k + 1], x_p, p[:, :, :, k + 1]))
    print("EC        :", get_EC(x[:, :, :, k + 1], x_p, p[:, :, :, k + 1]))
    print("Total EC  :", get_total_EC(x[:, :, :, k + 1], x_p, p[:, :, :, k + 1]))
    print("SOH       :", get_SOH(x[:, :, :, k + 1], x_p, p[:, :, :, k + 1], soh_coef))
    print("User obj  :", result1)
    print("Oper obj  :", result2)
    print("Oper tax  :", operator_ESS.value)
    print("Gap       :", pr)
    return x[:, :, :, k + 1], x_p, p[:, :, :, k + 1]


load_123 = np.load("load_123.npy", allow_pickle=True)[:,:24*day]
load_12 = np.load("load_12.npy", allow_pickle=True)[:,:24*day]
load_13 = np.load("load_13.npy", allow_pickle=True)[:,:24*day]
load_23 = np.load("load_23.npy", allow_pickle=True)[:,:24*day]

error = [0, 0.01, 0.02, 0.05, 0.1, 0.2]
load_error = np.load("load_error.npy", allow_pickle=True)

price_sell = np.load("ps.npy", allow_pickle=True)
price_buy = np.load("pb.npy", allow_pickle=True)
par = []
ec = []
exp_nash = []
#par_list = []
#ec_list = []
#exp_nash_list = []
for user in range(35, 71):
    for i in range(3, len(price_sell)):
        if user<=24 and i <=5:
            continue
        ps = price_sell[i]
        pb = price_buy[i]
        x, xp, p = our_model(user, load, p_SOH, ps, pb, "our_model_user_"+str(user)+"_price_"+str(i+1)+".npy")
        par += [get_PAR(x, xp, p)]
        ec += [get_EC(x, xp, p)]
        exp_nash += [[x, xp, p]]
    #par_list += [par]
    #ec_list += [ec]
    #exp_nash_list += [exp_nash]
    #np.save("par.npy", par_list)
    #np.save("ec.npy", ec_list)
    #np.save("our_model_initial.npy", exp_nash)

