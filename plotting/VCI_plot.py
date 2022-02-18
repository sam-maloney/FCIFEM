# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: samal
"""

import numpy as np
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
# Left and bottom borders and centre point constrained
# NY = NX, NQY=NY, NQX=1, quadType='gauss', massLumping=False
NX_array = np.array([  4,   8,  16,  32,  64, 128, 256])
start = int(np.log2(NX_array[0]))
stop = int(np.log2(NX_array[-1]))


# ##### Qord=2 #####

# # Sinusoidal mapping, uniform
# E_2_u_sin = np.array([5.65906087e-04, 2.12439915e-03, 7.24764050e-04, 1.81523111e-04,
#         4.65776364e-05, 1.15788587e-05, 2.88811650e-06])
# E_inf_u_sin = np.array([1.13181217e-03, 4.49000458e-03, 1.63010343e-03, 4.78277642e-04,
#         1.34469499e-04, 3.25764883e-05, 9.10318042e-06])
# t_setup_u_sin = np.array([7.70695601e-03, 2.44843290e-02, 8.09918420e-02, 3.12485005e-01,
#         1.21894376e+00, 4.79720513e+00, 1.90503340e+01])
# t_solve_u_sin = np.array([4.67006990e-04, 8.88426031e-04, 3.64110002e-03, 9.02129599e-03,
#         2.05598760e-02, 1.48443873e-01, 7.94564369e-01])

# # Sinusoidal mapping, 0.1 perturbation
# E_2_p1_sin = np.array([2.29568081e-03, 2.83979852e-03, 1.42301075e-03, 8.15506243e-04,
#         6.09308915e-04, 5.85970752e-04, 5.87676156e-04])
# E_inf_p1_sin = np.array([4.73774771e-03, 6.89905234e-03, 4.94754661e-03, 3.25690657e-03,
#         1.95384678e-03, 1.73365272e-03, 1.51457964e-03])
# t_setup_p1_sin = np.array([7.22425705e-03, 2.27039210e-02, 8.25599900e-02, 3.19918255e-01,
#         1.21794830e+00, 4.79542185e+00, 1.89186811e+01])
# t_solve_p1_sin = np.array([7.54970999e-04, 2.49404198e-03, 5.50760998e-03, 1.29038800e-02,
#         3.66370250e-02, 2.52634277e-01, 1.72728486e+00])

# # Sinusoidal mapping, 0.1 perturbation, Linear VCI
# E_2_p1_sin_L = np.array([3.56897796e-03, 2.29176314e-03, 1.20357939e-03, 6.55911094e-04,
#         4.65881879e-04, 4.35213098e-04, 4.40583573e-04])
# E_inf_p1_sin_L = np.array([1.09282783e-02, 6.28160870e-03, 4.17659691e-03, 2.57462330e-03,
#         1.67828742e-03, 1.16266382e-03, 1.14258424e-03])
# t_setup_p1_sin_L = np.array([7.51632301e-03, 2.56376730e-02, 8.67289380e-02, 3.47360178e-01,
#         1.29179624e+00, 5.19643203e+00, 2.00453248e+01])
# t_solve_p1_sin_L = np.array([7.36397982e-04, 2.58658401e-03, 5.62447903e-03, 1.31172700e-02,
#         3.77871740e-02, 2.29154345e-01, 1.68768613e+00])

# # Sinusoidal mapping, 0.1 perturbation, Quadratic VCI
# E_2_p1_sin_Q = np.array([3.80511436e-03, 2.35974784e-03, 6.50032068e-04, 2.34499251e-04,
#         8.79342241e-05, 4.09017791e-05, 3.62236534e-05])
# E_inf_p1_sin_Q = np.array([1.04012987e-02, 5.45828913e-03, 1.63360611e-03, 7.85188674e-04,
#         3.74168284e-04, 1.67395909e-04, 1.19366751e-04])
# t_setup_p1_sin_Q = np.array([1.99142660e-02, 7.21794970e-02, 3.60903704e-01, 1.09139127e+00,
#         4.27543800e+00, 1.69928375e+01, 6.74547598e+01])
# t_solve_p1_sin_Q = np.array([7.47944985e-04, 2.48315296e-03, 5.03371697e-03, 1.12878500e-02,
#         3.55030620e-02, 2.41591633e-01, 2.26252282e+00])

# # Sinusoidal mapping, 0.1 perturbation, ConservativePoint VCI
# E_2_p1_sin_CP = np.array([4.71828677e-03, 1.85512755e-03, 8.09928258e-04, 1.67244771e-04,
#         4.69070009e-05, 1.00442470e-05, 2.61654297e-06])
# E_inf_p1_sin_CP = np.array([1.36524395e-02, 4.39996767e-03, 2.36486184e-03, 6.73243780e-04,
#         1.94451683e-04, 6.82023029e-05, 1.37604238e-05])
# t_setup_p1_sin_CP = np.array([8.91726301e-03, 2.79221000e-02, 1.07774577e-01, 4.23878266e-01,
#         1.75276364e+00, 8.42446542e+00, 4.18039487e+01])
# t_solve_p1_sin_CP = np.array([7.72189000e-04, 2.43256899e-03, 5.20179200e-03, 1.09618860e-02,
#         3.36350870e-02, 2.13788053e-01, 1.21868699e+00])


##### Qord=3 #####

# Sinusoidal mapping, uniform
E_2_u_sin = np.array([2.32365060e-04, 2.31363124e-03, 5.69281741e-04, 1.59750346e-04,
        4.06278623e-05, 1.02224723e-05, 2.56121590e-06])
E_inf_u_sin = np.array([4.64730121e-04, 4.74666758e-03, 1.17420949e-03, 3.75900137e-04,
        1.06449355e-04, 2.83879579e-05, 7.61215061e-06])
t_setup_u_sin = np.array([1.31771790e-02, 4.60779430e-02, 1.81022431e-01, 7.06541914e-01,
        2.75732036e+00, 1.08622996e+01, 4.25364883e+01])
t_solve_u_sin = np.array([4.11210000e-04, 7.45905971e-04, 4.10119398e-03, 8.85574502e-03,
        2.09891450e-02, 3.49848579e-01, 8.28510807e-01])

# Sinusoidal mapping, 0.1 perturbation
E_2_p1_sin = np.array([2.36304941e-03, 2.24918006e-03, 7.88670096e-04, 3.33094204e-04,
        1.83642478e-04, 1.49196002e-04, 1.34220420e-04])
E_inf_p1_sin = np.array([6.92369577e-03, 5.32984077e-03, 2.23025933e-03, 1.96151098e-03,
        8.37637450e-04, 5.79428094e-04, 4.28699518e-04])
t_setup_p1_sin = np.array([1.35470160e-02, 4.68152690e-02, 1.79023350e-01, 6.84031013e-01,
        2.69265338e+00, 1.06143271e+01, 4.26205746e+01])
t_solve_p1_sin = np.array([8.45964008e-04, 2.62242701e-03, 5.30733803e-03, 1.14328330e-02,
        3.41176170e-02, 2.15202722e-01, 1.52689188e+00])

# Sinusoidal mapping, 0.1 perturbation, Linear VCI
E_2_p1_sin_L = np.array([2.24578349e-03, 2.05684766e-03, 7.38747735e-04, 2.85893160e-04,
        1.73666207e-04, 1.50146832e-04, 1.43279099e-04])
E_inf_p1_sin_L = np.array([4.43515213e-03, 5.77387201e-03, 2.31062375e-03, 1.07777191e-03,
        7.04285115e-04, 5.10240981e-04, 4.21492606e-04])
t_setup_p1_sin_L = np.array([1.38069520e-02, 4.85098010e-02, 1.87601476e-01, 7.12139480e-01,
        2.82149238e+00, 1.11472363e+01, 4.49303607e+01])
t_solve_p1_sin_L = np.array([8.48277006e-04, 2.61761196e-03, 5.39691403e-03, 1.09792550e-02,
        3.44774890e-02, 2.24973548e-01, 1.42814614e+00])

# Sinusoidal mapping, 0.1 perturbation, Quadratic VCI
E_2_p1_sin_Q = np.array([2.71513763e-03, 1.85170594e-03, 6.06365190e-04, 1.75243793e-04,
        5.78953309e-05, 2.73491294e-05, 2.22572623e-05])
E_inf_p1_sin_Q = np.array([5.39846259e-03, 4.67070383e-03, 1.45322807e-03, 5.15460207e-04,
        1.99803061e-04, 1.07477108e-04, 6.96378631e-05])
t_setup_p1_sin_Q = np.array([4.12836170e-02, 1.54654714e-01, 6.07982480e-01, 2.40523292e+00,
        9.62489600e+00, 3.84389026e+01, 1.49837853e+02])
t_solve_p1_sin_Q = np.array([8.08179029e-04, 2.44055601e-03, 5.14314201e-03, 1.06335820e-02,
        3.30339870e-02, 2.08540382e-01, 1.46179792e+00])

# Sinusoidal mapping, 0.1 perturbation, ConservativePoint VCI
E_2_p1_sin_CP = np.array([3.69029624e-03, 1.92980082e-03, 6.67047266e-04, 1.56867762e-04,
        3.92884774e-05, 9.84783314e-06, 2.31601938e-06])
E_inf_p1_sin_CP = np.array([9.98171754e-03, 4.67510179e-03, 2.08301164e-03, 4.94114246e-04,
        1.51247694e-04, 5.02897333e-05, 1.10893463e-05])
t_setup_p1_sin_CP = np.array([1.66634240e-02, 5.58684170e-02, 2.33342315e-01, 9.98959650e-01,
        3.99752330e+00, 1.76521769e+01, 8.61894637e+01])
t_solve_p1_sin_CP = np.array([8.22620001e-04, 2.51573796e-03, 5.18891303e-03, 1.07161680e-02,
        3.19106030e-02, 2.07477785e-01, 1.70623369e+00])


##### Qord=10 #####

# Sinusoidal mapping, 0.1 perturbation, Qord=10
E_2_p1_sin10 = np.array([1.97831030e-03, 1.67942706e-03, 5.07108806e-04, 1.45293429e-04,
       4.23171560e-05, 1.51363345e-05, 7.93534263e-06])
E_inf_p1_sin10 = np.array([4.22189624e-03, 3.94430507e-03, 1.39784711e-03, 5.15578145e-04,
       1.52409928e-04, 7.95138585e-05, 4.67892759e-05])
t_setup_p1_sin10 = np.array([1.19064700e-01, 4.72790800e-01, 1.84346810e+00, 7.38443345e+00,
       2.93813134e+01, 1.17864939e+02, 4.75000347e+02])
t_solve_p1_sin10 = np.array([7.80342962e-04, 2.45964801e-03, 4.99731995e-03, 1.12120030e-02,
       3.24666430e-02, 7.03826597e-01, 1.26428737e+00])


##### Begin Plotting Routines #####

solid_linewidth = 1.25
dashed_linewidth = 1.0

plt.rcParams['markers.fillstyle'] = 'full'
plt.rcParams['lines.markersize'] = 5.0
plt.rcParams['lines.linewidth'] = solid_linewidth
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = True
# fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
plt.rcParams['legend.fontsize'] = 'small'
# plt.rcParams['font.size'] = 'small'
# plt.rcParams['axes.titlesize'] = 'medium'
# plt.rcParams['axes.labelsize'] = 'medium'
# plt.rcParams['xtick.labelsize'] = 'small'
# plt.rcParams['ytick.labelsize'] = 'small'
# plt.rcParams['figure.titlesize'] = 'large'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
blue = colors[0]
orange = colors[1]
green = colors[2]
red = colors[3]
purple = colors[4]
brown = colors[5]
pink = colors[6]
grey = colors[7]
yellow = colors[8]
cyan = colors[9]

# clear the current figure, if opened, and set parameters
fig = plt.figure()
fig.clf()
fig.set_size_inches(7.75,3)
plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

axL1 = plt.subplot(121)
N_array = np.log2(NX_array**2).astype('int')
# plt.semilogy(N_array, E_2_u_str, '.-', label=r'Str/Uniform')
plt.semilogy(N_array, E_2_u_sin, 'o-', color=blue, label=r'Uniform')
# plt.semilogy(N_array, E_2_p1_str, '.-', label=r'Str/10% pert')
plt.semilogy(N_array, E_2_p1_sin, 's-', color=red, label=r'10\% perturbation')
plt.minorticks_off()
# plt.ylim(top=1.5e-2)
plt.xticks(N_array, N_array)
plt.xlabel(r'$\log_2(N_xN_y)$')
plt.ylabel(r'$|E_2|$', rotation=0, labelpad=10)
# plt.title('Uniform vs. Perturbed Grid')

# plot the intra-step order of convergence
plt.rcParams['lines.linewidth'] = dashed_linewidth
axL2 = axL1.twinx()
logN = np.log(NX_array)
plt.autoscale(False)
plt.plot(plt.xlim(), [2, 2], ':k', label='Expected ord')
# intraN = np.logspace(start+0.5, stop-0.5, num=stop-start, base=2.0)
intraN = np.arange(2*start + 1, 2*stop, 2)
# logE_str = np.log(E_2_u_str)
# order_str = (logE_str[0:-1] - logE_str[1:])/(logN[1:] - logN[0:-1])
# plt.plot(intraN, order_str, '.:', label=r'Str/Uniform order')
logE_sin = np.log(E_2_u_sin)
order_sin = (logE_sin[0:-1] - logE_sin[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_sin, 'o:', color=blue, label=r'Uniform order')
# logE_p1_str = np.log(E_2_p1_str)
# order_p1_str = (logE_p1_str[0:-1] - logE_p1_str[1:])/(logN[1:] - logN[0:-1])
# plt.plot(intraN, order_p1_str, '.:', label=r'Str/10% pert order')
logE_p1_sin = np.log(E_2_p1_sin)
order_p1_sin = (logE_p1_sin[0:-1] - logE_p1_sin[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_p1_sin, 's:', color=red, label=r'10\% pert order')

ordb = 0
ordt = 2.5
plt.ylim(ordb, ordt)
# plt.yticks(np.linspace(ordb, ordt, ordt - ordb + 1)) # unit spacing
plt.yticks(np.linspace(ordb, ordt, int((ordt - ordb)*2 + 1))) # 0.5 spacing
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = axL1.get_legend_handles_labels()
lines2, labels2 = axL2.get_legend_handles_labels()
leg = axL2.legend(lines, labels, loc='lower left')
# leg = axL2.legend(lines + lines2, labels + labels2, loc='best')
# leg = axL2.legend(lines + [lines2[-1]], labels + [labels2[-1]], loc='lower left')
plt.margins(0,0)

# plot the error convergence
plt.rcParams['lines.linewidth'] = solid_linewidth
axR1 = plt.subplot(122)
plt.autoscale(True)
# plt.semilogy(N_array, E_2_p1_sin, '.-', label=r'Q3')
plt.semilogy(N_array, E_2_p1_sin10, 'o-', color=blue, label=r'Q10')
plt.semilogy(N_array, E_2_p1_sin_L, 's-', color=red, label=r'Q3 VC1')
plt.semilogy(N_array, E_2_p1_sin_Q, '^-', color=orange, label=r'Q3 VC2')
plt.semilogy(N_array, E_2_p1_sin_CP, 'd-k', label=r'Q3 VC1-C')
plt.minorticks_off()
# plt.ylim(top=1.5e-2)
plt.xticks(N_array, N_array)
plt.xlabel(r'$\log_2(N_xN_y)$')
plt.ylabel(r'$|E_2|$', rotation=0, labelpad=10)
# plt.title(r'Improved Quadrature')

# plot the intra-step order of convergence
plt.rcParams['lines.linewidth'] = dashed_linewidth
axR2 = axR1.twinx()
logN = np.log(NX_array)
plt.plot(plt.xlim(), [2, 2], ':k', label='Expected ord')
# logE_p1_sin = np.log(E_2_p1_sin)
# order_p1_sin = (logE_p1_sin[0:-1] - logE_p1_sin[1:])/(logN[1:] - logN[0:-1])
# plt.plot(intraN, order_p1_sin, '.:', label=r'Q3 order')
logE_sin10 = np.log(E_2_p1_sin10)
order_sin10 = (logE_sin10[0:-1] - logE_sin10[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_sin10, 'o:', color=blue, label=r'Q10 order')
logE_p1_sin_L = np.log(E_2_p1_sin_L)
order_p1_sin_L = (logE_p1_sin_L[0:-1] - logE_p1_sin_L[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_p1_sin_L, 's:', color=red, label=r'Q3 VC1 order')
logE_p1_sin_Q = np.log(E_2_p1_sin_Q)
order_p1_sin_Q = (logE_p1_sin_Q[0:-1] - logE_p1_sin_Q[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_p1_sin_Q, '^:', color=orange, label=r'Q3 VC2 order')
logE_p1_sin_CP = np.log(E_2_p1_sin_CP)
order_p1_sin_CP = (logE_p1_sin_CP[0:-1] - logE_p1_sin_CP[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_p1_sin_CP, 'd:k', label=r'Q3 VC1-C order')

ordb = 0
ordt = 2.5
plt.ylim(ordb, ordt)
# plt.yticks(np.linspace(ordb, ordt, ordt - ordb + 1))
plt.yticks(np.linspace(ordb, ordt, int((ordt - ordb)*2 + 1)))
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = axR1.get_legend_handles_labels()
lines2, labels2 = axR2.get_legend_handles_labels()
leg = axR2.legend(lines, labels, loc='lower left')
# leg = axR2.legend(lines + lines2, labels + labels2, loc='best')
# leg = axR2.legend(lines + [lines2[-1]], labels + [labels2[-1]], loc='lower left')

ylimL = axL1.get_ylim()
ylimR = axR1.get_ylim()
axL1.set_ylim(min(ylimL[0],ylimR[0]), max(ylimL[1],ylimR[1]))
axR1.set_ylim(min(ylimL[0],ylimR[0]), max(ylimL[1],ylimR[1]))

plt.margins(0,0)
plt.savefig("Poisson_conv_all_VCI.pdf", bbox_inches = 'tight', pad_inches = 0)