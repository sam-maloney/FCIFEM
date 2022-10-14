# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: Samuel A. Maloney
"""

import numpy as np
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
# u0(x,y) = sin(2*pi*x - 2*pi*y)
# dt = 0.01, t_final = 1, theta = 0.7853981633974483
# Omega = (0,1) X (0,1), Periodic BCs
# NQY = NY, quadType = 'gauss', massLumping = False

E_2_L = []
E_inf_L = []
t_setup_L = []
t_solve_L = []
labels_L = []
NX_L = []
NY_L = []

E_2_R = []
E_inf_R = []
t_setup_R = []
t_solve_R = []
labels_R = []
NX_R = []
NY_R = []


#%% Qord = 2

##### Uniform Grid, no VCI #####

# StraightMapping()
# NQX = 2
E_2_L.append(np.array([7.07102499e-01, 6.32402419e-01, 2.86940209e-01, 8.47400108e-02,
       2.21045970e-02, 5.58543215e-03, 1.40009207e-03]))
E_inf_L.append(np.array([9.99993944e-01, 8.94352078e-01, 4.05794735e-01, 1.19840473e-01,
       3.12606208e-02, 7.89899390e-03, 1.98002924e-03]))
t_setup_L.append(np.array([8.39306402e-03, 2.77188430e-02, 1.12893830e-01, 4.43844597e-01,
       1.70832985e+00, 6.86146949e+00, 5.85081655e+01]))
t_solve_L.append(np.array([1.01042930e-02, 1.55548900e-02, 1.99466010e-02, 3.45256270e-02,
       1.07203592e-01, 3.98279348e-01, 3.69627561e+00]))
labels_L.append('unaligned 1:1')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# StraightMapping()
# NQX = 2
E_2_L.append(np.array([7.07106781e-01, 7.05811181e-01, 4.91497907e-01, 1.70622808e-01,
       4.63469303e-02, 1.18294095e-02, 2.97270987e-03]))
E_inf_L.append(np.array([1.00000000e+00, 9.98167745e-01, 6.95083006e-01, 2.41297090e-01,
       6.55444575e-02, 1.67293114e-02, 4.20404672e-03]))
t_setup_L.append(np.array([8.22972599e-03, 2.91037130e-02, 3.33058056e-01, 1.33388678e+00,
       5.31589151e+00, 2.12669501e+01, 8.54309218e+01]))
t_solve_L.append(np.array([7.39359600e-03, 1.49197310e-02, 4.19275480e-02, 6.55330370e-02,
       1.92617814e-01, 7.25883293e-01, 2.65749576e+00]))
labels_L.append('unaligned 1:4')
NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
NY_L.append(4)

# # StraightMapping()
# # NQX = 2*NY/NX = 8
# E_2_L.append(np.array([7.07106781e-01, 7.05811181e-01, 4.91497907e-01, 1.70622808e-01,
#        4.63469303e-02, 1.18294095e-02, 2.97270992e-03]))
# E_inf_L.append(np.array([1.00000000e+00, 9.98167745e-01, 6.95083006e-01, 2.41297090e-01,
#        6.55444575e-02, 1.67293114e-02, 4.20404680e-03]))
# t_setup_L.append(np.array([2.77759810e-02, 1.12366065e-01, 4.23069426e-01, 1.69134054e+00,
#        6.80023618e+00, 5.96505151e+01, 1.81200726e+02]))
# t_solve_L.append(np.array([7.45609001e-03, 1.52141900e-02, 1.86658920e-02, 3.35279490e-02,
#        1.02366033e-01, 7.07833079e-01, 2.63570252e+00]))
# labels_L.append('unaligned 1:4')
# NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_L.append(4)

# # LinearMapping(1.0)
# # NQX = 2
# E_2_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# E_inf_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# t_setup_L.append(np.array([9.25346697e-03, 3.21748269e-02, 1.23727462e-01, 4.81422153e-01,
#        1.93050537e+00, 7.58449797e+00, 3.05494134e+01]))
# t_solve_L.append(np.array([2.85823800e-03, 2.90348398e-03, 3.37723800e-03, 4.23491898e-03,
#        7.99175602e-03, 2.37594850e-02, 1.13397781e-01]))
# labels_L.append('aligned 1:1')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)

# # LinearMapping(1.0)
# # NQX = 2
# E_2_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00]))
# E_inf_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00]))
# t_setup_L.append(np.array([3.00081340e-02, 1.19100415e-01, 4.83360463e-01, 1.89216008e+00,
#        7.61443644e+00, 7.47863485e+01]))
# t_solve_L.append(np.array([2.74835201e-03, 3.11105500e-03, 4.04266600e-03, 7.38527800e-03,
#        2.49185030e-02, 1.68670821e-01]))
# labels_L.append('aligned 1:16')
# NX_L.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY_L.append(16)


# ##### px = py = 0.1, seed = 42, no VCI #####

# StraightMapping()
# NQX = 2
E_2_R.append(np.array([7.07448695e-01, 6.35011387e-01, 2.90646457e-01, 8.61810504e-02,
        2.24596145e-02, 5.67433074e-03, 1.42485852e-03]))
E_inf_R.append(np.array([1.00477395e+00, 9.00660009e-01, 4.15355527e-01, 1.23879062e-01,
        3.30582064e-02, 8.39822500e-03, 2.12766492e-03]))
t_setup_R.append(np.array([8.43639701e-03, 2.86128280e-02, 1.13110589e-01, 4.45229557e-01,
        1.71987595e+00, 6.85899247e+00, 8.57916602e+01]))
t_solve_R.append(np.array([1.03429280e-02, 1.57817300e-02, 1.91249180e-02, 3.43449010e-02,
        1.13200130e-01, 4.64121109e-01, 3.86005894e+00]))
labels_R.append('unaligned 1:1')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_R.append(1)

# # StraightMapping()
# # NQX = 2
# E_2_R.append(np.array([7.06071506e-01, 7.09956287e-01, 4.96651081e-01, 1.73343753e-01,
#        4.71739292e-02, 1.20118549e-02, 3.01627462e-03]))
# E_inf_R.append(np.array([1.00005657e+00, 9.98244195e-01, 7.02332762e-01, 2.46612275e-01,
#        6.71152084e-02, 1.71686670e-02, 4.31216026e-03]))
# t_setup_R.append(np.array([8.33054195e-03, 2.88201780e-02, 1.15684686e-01, 1.09445508e+00,
#        5.36132829e+00, 2.16960538e+01, 4.49775794e+01]))
# t_solve_R.append(np.array([1.17213990e-02, 1.70038120e-02, 1.94246850e-02, 6.72728960e-02,
#        1.99776603e-01, 6.31660898e-01, 3.50252752e+00]))
# labels_R.append('unaligned 1:4')
# NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_R.append(4)

# # StraightMapping()
# # NQX = 2*NY/NX = 8
# E_2_R.append(np.array([7.06071506e-01, 7.09956287e-01, 4.96651081e-01, 1.73343753e-01,
#         4.71739292e-02, 1.20118549e-02, 3.01627462e-03]))
# E_inf_R.append(np.array([1.00005657e+00, 9.98244195e-01, 7.02332762e-01, 2.46612275e-01,
#         6.71152084e-02, 1.71686670e-02, 4.31216026e-03]))
# t_setup_R.append(np.array([2.84658810e-02, 1.05656812e-01, 8.80536200e-01, 5.30077956e+00,
#         2.13091930e+01, 3.84704133e+01, 1.98987728e+02]))
# t_solve_R.append(np.array([1.09323960e-02, 1.52622050e-02, 4.33308770e-02, 6.72776140e-02,
#         1.99617540e-01, 7.14287854e-01, 3.43150723e+00]))
# labels_R.append('unaligned 1:4')
# NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_R.append(4)

# LinearMapping(1.0)
# NQX = 2
E_2_R.append(np.array([5.44660098e-02, 1.43405612e-02, 6.89535567e-03, 2.43109177e-03,
        7.31317721e-04, 2.19958333e-04, 5.82577843e-05]))
E_inf_R.append(np.array([8.85325196e-02, 3.60005603e-02, 1.61937650e-02, 7.26873331e-03,
        2.19637150e-03, 8.95937918e-04, 2.71687751e-04]))
t_setup_R.append(np.array([2.56886540e-02, 9.40227450e-02, 3.75932368e-01, 1.49201452e+00,
        5.90069044e+00, 2.36700608e+01, 6.84904408e+01]))
t_solve_R.append(np.array([2.82979180e-02, 3.06785430e-02, 3.56860510e-02, 5.96461380e-02,
        2.18346108e-01, 8.16068900e-01, 3.45866489e+00]))
labels_R.append('aligned 1:1')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_R.append(1)

# # LinearMapping(1.0)
# # NQX = 2
# E_2_R.append(np.array([1.16690317e-02, 3.37918842e-03, 1.25809264e-03, 3.11231707e-04,
#        9.68315044e-05, 2.69354992e-05, 7.40313957e-06]))
# E_inf_R.append(np.array([1.84503695e-02, 9.17559015e-03, 3.85301481e-03, 1.49602980e-03,
#        4.47418033e-04, 1.37362408e-04, 3.76058163e-05]))
# t_setup_R.append(np.array([2.50602480e-02, 9.34238260e-02, 3.71478654e-01, 1.46576713e+00,
#        5.84375629e+00, 2.34483538e+01, 9.35721318e+01]))
# t_solve_R.append(np.array([2.33382650e-02, 3.00430400e-02, 3.52520950e-02, 5.72499690e-02,
#        1.66368956e-01, 6.05174458e-01, 2.86908873e+00]))
# labels_R.append('aligned 1:4')
# NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_R.append(4)

# # LinearMapping(1.0)
# # NQX = 2
# E_2_R.append(np.array([6.62261699e-04, 2.52437230e-04, 6.49630157e-05, 1.75872829e-05,
#        4.98119124e-06, 1.32066564e-06]))
# E_inf_R.append(np.array([1.69021758e-03, 8.29575228e-04, 2.52457064e-04, 7.76755926e-05,
#        2.22228807e-05, 7.13334665e-06]))
# t_setup_R.append(np.array([9.41028950e-02, 3.73975909e-01, 1.48622124e+00, 5.87716088e+00,
#        2.34903765e+01, 4.09797032e+01]))
# t_solve_R.append(np.array([3.01620550e-02, 2.99753310e-02, 4.32885930e-02, 1.17833754e-01,
#        4.68606211e-01, 2.49566663e+00]))
# labels_R.append('aligned 1:16')
# NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY_R.append(16)

# LinearMapping(1.0)
# NQX = 2*NY/NX = 8
E_2_R.append(np.array([9.45561263e-03, 2.97389170e-03, 1.13526445e-03, 2.92348290e-04,
        8.21824901e-05, 2.46466598e-05, 6.88023053e-06]))
E_inf_R.append(np.array([1.66144277e-02, 8.39359371e-03, 3.69067658e-03, 1.20180476e-03,
        3.45918281e-04, 1.29176652e-04, 3.38074281e-05]))
t_setup_R.append(np.array([3.06043700e-02, 1.24677755e-01, 4.83355725e-01, 1.87676842e+00,
        7.58027188e+00, 3.02691679e+01, 2.85467679e+02]))
t_solve_R.append(np.array([1.25539050e-02, 1.37132530e-02, 1.60676281e-02, 2.76841620e-02,
        8.95892230e-02, 3.37857407e-01, 2.62958947e+00]))
labels_R.append('aligned 1:4')
NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
NY_R.append(4)

# LinearMapping(1.0)
# NQX = 2*NY/NX = 32
E_2_R.append(np.array([6.94071548e-04, 2.44589567e-04, 6.24443417e-05, 1.68693305e-05,
        4.69617007e-06, 1.26475717e-06]))
E_inf_R.append(np.array([1.73686746e-03, 6.53717870e-04, 2.47900868e-04, 7.29518015e-05,
        2.18310093e-05, 7.05301583e-06]))
t_setup_R.append(np.array([1.45468265e+00, 5.81184738e+00, 2.34448912e+01, 9.31427256e+01,
        2.81053673e+02, 1.20709391e+03]))
t_solve_R.append(np.array([2.53560630e-02, 3.13328060e-02, 4.33115539e-02, 1.14914145e-01,
        2.62726201e-01, 2.54636588e+00]))
labels_R.append('aligned 1:16')
NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
NY_R.append(16)


##### px = py = 0.1, seed = 42 #####

# # StraightMapping()
# # VCI-C (slice-by-slice) using ssqr.min2norm
# # NQX = 2
# E_2_R.append(np.array([7.07460738e-01, 6.28733266e-01, 2.88468236e-01, 8.39029104e-02,
#        2.18154189e-02, 5.47695188e-03, 1.37450982e-03]))
# E_inf_R.append(np.array([1.00225239e+00, 8.90015233e-01, 4.15749818e-01, 1.22727131e-01,
#        3.28053395e-02, 8.29429217e-03, 2.13508324e-03]))
# t_setup_R.append(np.array([1.52368690e-02, 5.34187540e-02, 2.08331038e-01, 8.58873133e-01,
#        8.18670468e+00, 3.54357894e+01, 1.44208128e+02]))
# t_solve_R.append(np.array([1.01175490e-02, 1.62076520e-02, 1.89295400e-02, 3.45800980e-02,
#        1.99944312e-01, 5.28063695e-01, 3.77378788e+00]))
# labels_R.append('unaligned 1:1')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)

# # StraightMapping()
# # VCI-C (whole domain) using ssqr.min2norm
# # NQX = 2
# E_2_R.append(np.array([7.07522489e-01, 6.33604215e-01, 2.87585815e-01, 8.50195268e-02,
#        2.21426770e-02, 5.59294990e-03, 1.40414756e-03]))
# E_inf_R.append(np.array([1.00014181e+00, 9.00877794e-01, 4.07694481e-01, 1.21739957e-01,
#        3.21316153e-02, 8.14508358e-03, 2.04921265e-03]))
# t_setup_R.append(np.array([1.44033150e-02, 4.95031730e-02, 2.13672003e-01, 8.25592303e-01,
#        4.51149219e+00, 3.11490825e+01, 1.08382166e+02]))
# t_solve_R.append(np.array([1.03412060e-02, 1.53460940e-02, 1.95661860e-02, 3.53527070e-02,
#        1.08814274e-01, 4.09430433e-01, 4.65426143e+00]))
# labels_R.append('unaligned 1:1')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)

# # StraightMapping()
# # VC1-C (whole domain) using ssqr.min2norm
# # NQX = 2
# E_2_R.append(np.array([7.06222353e-01, 7.09962346e-01, 4.95729635e-01, 1.72800843e-01,
#         4.70054906e-02, 1.19676031e-02, 3.00503588e-03]))
# E_inf_R.append(np.array([1.00587660e+00, 9.98881107e-01, 7.01605157e-01, 2.45671528e-01,
#         6.71134676e-02, 1.70650254e-02, 4.30388644e-03]))
# t_setup_R.append(np.array([1.37381360e-02, 4.99025170e-02, 1.91920213e-01, 8.01411791e-01,
#         7.21032738e+00, 3.20181782e+01, 9.47690390e+01]))
# t_solve_R.append(np.array([1.17648750e-02, 1.60609750e-02, 1.95374970e-02, 3.38323560e-02,
#         2.00324333e-01, 7.10507357e-01, 2.94556064e+00]))
# labels_R.append('unaligned 1:4')
# NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_R.append(4)

# # StraightMapping()
# # VC1-C (whole domain) using ssqr.min2norm
# # NQX = 2*NY/NX = 8
# E_2_R.append(np.array([7.06222353e-01, 7.09962346e-01, 4.95729635e-01, 1.72800843e-01,
#        4.70054906e-02, 1.19676031e-02, 3.00503588e-03]))
# E_inf_R.append(np.array([1.00587660e+00, 9.98881107e-01, 7.01605157e-01, 2.45671528e-01,
#        6.71134676e-02, 1.70650254e-02, 4.30388644e-03]))
# t_setup_R.append(np.array([1.58250278e-01, 2.66735378e-01, 7.95538807e-01, 3.14460427e+00,
#        2.86907347e+01, 9.68201580e+01, 3.44822057e+02]))
# t_solve_R.append(np.array([2.69431040e-02, 1.57664799e-02, 2.08054330e-02, 3.48960420e-02,
#        1.97102968e-01, 4.26821031e-01, 3.31158523e+00]))
# labels_R.append('unaligned 1:4')
# NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_R.append(4)

# # LinearMapping(1.0)
# # VC1-C (whole domain) using ssqr.min2norm
# # NQX = 2
# E_2_R.append(np.array([5.43432006e-02, 1.41517366e-02, 6.83758078e-03, 2.37358588e-03,
#        7.27359517e-04, 2.16833668e-04, 5.77301327e-05]))
# E_inf_R.append(np.array([9.03365151e-02, 3.42703949e-02, 1.54744165e-02, 7.10849879e-03,
#        2.21268240e-03, 8.52686763e-04, 2.74716678e-04]))
# t_setup_R.append(np.array([4.24074560e-02, 1.64051813e-01, 6.51828743e-01, 2.61073537e+00,
#        1.05224015e+01, 4.21282873e+01, 1.17776135e+02]))
# t_solve_R.append(np.array([2.85940690e-02, 3.14335740e-02, 3.56146390e-02, 5.94880830e-02,
#        2.14317297e-01, 8.48394292e-01, 3.14011060e+00]))
# labels_R.append('aligned 1:1')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)

# # LinearMapping(1.0)
# # VC1-C (whole domain) using ssqr.min2norm
# # NQX = 2
# E_2_R.append(np.array([1.20931509e-02, 3.36896452e-03, 1.24254965e-03, 3.09487775e-04,
#         9.66039571e-05, 2.69229948e-05, 7.37607780e-06]))
# E_inf_R.append(np.array([1.94802532e-02, 9.56401207e-03, 4.00237403e-03, 1.44547117e-03,
#         4.42193774e-04, 1.34704939e-04, 3.76399404e-05]))
# t_setup_R.append(np.array([4.30399200e-02, 1.63480293e-01, 6.56427033e-01, 2.59995278e+00,
#         1.04359472e+01, 3.86310914e+01, 1.22221516e+02]))
# t_solve_R.append(np.array([2.46115810e-02, 3.00631661e-02, 3.67770600e-02, 5.71617510e-02,
#         1.66950847e-01, 6.22991323e-01, 2.66621995e+00]))
# labels_R.append('aligned 1:4')
# NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_R.append(4)

# # LinearMapping(1.0)
# # VC1-C (whole domain) using ssqr.min2norm
# # NQX = 2
# E_2_R.append(np.array([6.51453117e-04, 2.45880142e-04, 6.53957520e-05, 1.76877574e-05,
#         4.94722153e-06, 1.30997324e-06]))
# E_inf_R.append(np.array([1.66271771e-03, 8.07198754e-04, 2.48851469e-04, 7.71454962e-05,
#         2.32594996e-05, 7.57807287e-06]))
# t_setup_R.append(np.array([5.22033330e-02, 2.05726801e-01, 1.87465076e+00, 1.03867671e+01,
#         4.17962471e+01, 7.01081067e+01]))
# t_solve_R.append(np.array([1.28145330e-02, 1.32781340e-02, 4.31080370e-02, 1.19834167e-01,
#         4.86013095e-01, 2.52412617e+00]))
# labels_R.append('aligned 1:16')
# NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY_R.append(16)

# # LinearMapping(1.0)
# # VC1-C (whole domain) using ssqr.min2norm
# # NQX = 2*NY/NX = 8
# E_2_R.append(np.array([9.66731177e-03, 2.97293353e-03, 1.13373312e-03, 2.92289566e-04,
#        8.33910700e-05, 2.47570413e-05, 6.90596904e-06]))
# E_inf_R.append(np.array([1.68781927e-02, 8.39707875e-03, 3.73873626e-03, 1.19879233e-03,
#        3.55490886e-04, 1.29895424e-04, 3.43475127e-05]))
# t_setup_R.append(np.array([7.96547560e-02, 2.13992186e-01, 8.65733649e-01, 3.47149343e+00,
#        3.05461182e+01, 1.08276821e+02, 3.97265040e+02]))
# t_solve_R.append(np.array([1.26781120e-02, 1.40505460e-02, 1.59700990e-02, 2.69987430e-02,
#        1.61389295e-01, 5.96038088e-01, 2.79138275e+00]))
# labels_R.append('aligned 1:4')
# NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_R.append(4)

# # LinearMapping(1.0)
# # VC1-C (whole domain) using ssqr.min2norm
# # NQX = 2*NY/NX = 32
# E_2_R.append(np.array([6.94439311e-04, 2.45136821e-04, 6.24750122e-05, 1.69096225e-05,
#        4.70588755e-06]))
# E_inf_R.append(np.array([1.72762126e-03, 6.57458808e-04, 2.46825551e-04, 7.29210697e-05,
#        2.18167867e-05]))
# t_setup_R.append(np.array([8.38270232e-01, 9.48508263e+00, 2.85601204e+01, 7.65827894e+01,
#        5.64696045e+02]))
# t_solve_R.append(np.array([1.09737940e-02, 3.12066320e-02, 2.13892010e-02, 6.58833050e-02,
#        4.73511064e-01]))
# labels_R.append('aligned 1:16')
# NX_R.append(np.array([ 2,  4,  8, 16, 32]))
# NY_R.append(16)

# # LinearMapping(1.0)
# # VCI-C (slice-by-slice) using scipy.sparse.linalg.lsqr
# # NQX = 2*NY/NX = 32
# E_2_R.append(np.array([6.94979966e-04, 2.43980284e-04, 6.24798602e-05, 1.67586907e-05,
#        4.68673088e-06, 1.25796171e-06]))
# E_inf_R.append(np.array([1.73399685e-03, 6.51807058e-04, 2.48844725e-04, 7.12552248e-05,
#        2.18097377e-05, 7.09999369e-06]))
# t_setup_R.append(np.array([1.87893247e+00, 7.32306180e+00, 7.90894622e+01, 5.13684013e+02,
#        3.46966044e+03, 2.84592882e+04]))
# t_solve_R.append(np.array([1.17172050e-02, 1.48357880e-02, 4.33917010e-02, 6.36274470e-02,
#        2.68959171e-01, 2.25124838e+00]))
# labels_R.append('aligned 1:16')
# NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY_R.append(16)


#%% Qord = 3

##### Uniform grid, VCI-C (slice-by-slice) using ssqr.min2norm #####

# # StraightMapping()
# # NQX = 1
# E_2_L.append(np.array([7.07102499e-01, 6.32402419e-01, 2.86940209e-01, 8.47400108e-02,
#         2.21045970e-02, 5.58543215e-03, 1.40009207e-03]))
# E_inf_L.append(np.array([9.99993944e-01, 8.94352078e-01, 4.05794735e-01, 1.19840473e-01,
#        3.12606208e-02, 7.89899390e-03, 1.98002923e-03]))
# t_setup_L.append(np.array([4.70260930e-02, 1.81292197e-01, 6.96564461e-01, 1.93102438e+00,
#        7.44859410e+00, 3.60649786e+01, 1.48865582e+02]))
# t_solve_L.append(np.array([2.30894950e-02, 3.65550540e-02, 4.17699400e-02, 3.36732130e-02,
#        1.08683866e-01, 5.07320221e-01, 4.05322921e+00]))
# labels_L.append('unaligned 1:1')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)

# # StraightMapping()
# # NQX = NY/NX = 4
# E_2_L.append(np.array([7.07106781e-01, 7.05811181e-01, 4.91497907e-01, 1.70622808e-01,
#        4.63469303e-02, 1.18294095e-02, 2.97270992e-03]))
# E_inf_L.append(np.array([1.00000000e+00, 9.98167745e-01, 6.95083006e-01, 2.41297090e-01,
#        6.55444575e-02, 1.67293114e-02, 4.20404680e-03]))
# t_setup_L.append(np.array([5.93563080e-02, 2.31001661e-01, 9.21408252e-01, 3.59988035e+00,
#        2.83620386e+01, 1.25452350e+02, 5.52517975e+02]))
# t_solve_L.append(np.array([7.45416700e-03, 1.57737530e-02, 1.94822680e-02, 3.32286810e-02,
#        1.92044193e-01, 3.80585466e-01, 3.04666002e+00]))
# labels_L.append('aligned 1:4')
# NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_L.append(4)

# # LinearMapping(1.0)
# # NQX = 1
# E_2_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# E_inf_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# t_setup_L.append(np.array([4.88730690e-02, 1.86418051e-01, 7.25021679e-01, 2.90234456e+00,
#        1.17377452e+01, 3.99592889e+01, 1.64007416e+02]))
# t_solve_L.append(np.array([6.50436699e-03, 6.97252101e-03, 7.31036499e-03, 1.00214330e-02,
#        2.03012190e-02, 2.47333640e-02, 2.10771298e-01]))
# labels_L.append('aligned 1:1')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)

# # LinearMapping(1.0)
# # NQX = 1
# E_2_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# E_inf_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# t_setup_L.append(np.array([4.93491310e-02, 1.84336010e-01, 7.29488004e-01, 2.93562295e+00,
#        9.31869497e+00, 3.79475151e+01, 1.47018570e+02]))
# t_solve_L.append(np.array([6.50308900e-03, 6.69645800e-03, 7.20020701e-03, 8.57191700e-03,
#        7.63390699e-03, 2.74969240e-02, 8.70193770e-02]))
# labels_L.append('aligned 1:4')
# NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_L.append(4)


# ##### Uniform grid, no VCI #####

# # StraightMapping()
# # NQX = 1
# E_2_L.append(np.array([7.07102499e-01, 6.32402419e-01, 2.86940209e-01, 8.47400108e-02,
#        2.21045970e-02, 5.58543215e-03, 1.40009207e-03]))
# E_inf_L.append(np.array([9.99993944e-01, 8.94352078e-01, 4.05794735e-01, 1.19840473e-01,
#        3.12606208e-02, 7.89899390e-03, 1.98002922e-03]))
# t_setup_L.append(np.array([2.59653440e-02, 9.60321950e-02, 3.72480096e-01, 1.51662188e+00,
#        5.96279402e+00, 2.38240262e+01, 9.32769962e+01]))
# t_solve_L.append(np.array([2.44251360e-02, 3.58391850e-02, 4.18269810e-02, 6.73743070e-02,
#        1.85792999e-01, 6.89515132e-01, 3.60605242e+00]))
# labels_L.append('unaligned 1:1')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)

# # StraightMapping()
# # NQX = 4
# E_2_L.append(np.array([7.07106781e-01, 7.05811181e-01, 4.91497907e-01, 1.70622808e-01,
#        4.63469304e-02, 1.18294095e-02, 2.97270993e-03]))
# E_inf_L.append(np.array([1.00000000e+00, 9.98167745e-01, 6.95083006e-01, 2.41297090e-01,
#        6.55444575e-02, 1.67293114e-02, 4.20404681e-03]))
# t_setup_L.append(np.array([3.04938000e-02, 1.15963849e-01, 4.74950790e-01, 1.89672524e+00,
#        7.51895720e+00, 6.01483583e+01, 3.08996843e+02]))
# t_solve_L.append(np.array([7.36359500e-03, 1.51947810e-02, 1.91468110e-02, 3.27766000e-02,
#        9.90197870e-02, 6.96034517e-01, 2.68265982e+00]))
# labels_L.append('unaligned 1:4')
# NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_L.append(4)

# # LinearMapping(1.0)
# # NQX = 1
# E_2_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# E_inf_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# t_setup_L.append(np.array([2.73262700e-02, 1.01799461e-01, 4.06831057e-01, 1.62005424e+00,
#        6.42052850e+00, 2.57952718e+01, 6.04735972e+01]))
# t_solve_L.append(np.array([6.54689601e-03, 7.07215800e-03, 7.35113501e-03, 9.95931601e-03,
#        2.04455920e-02, 7.57735950e-02, 2.65873768e-01]))
# labels_L.append('aligned 1:1')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)

# # LinearMapping(1.0)
# # NQX = 
# E_2_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# E_inf_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# t_setup_L.append(np.array([2.73053050e-02, 1.02500677e-01, 4.06507570e-01, 1.62117526e+00,
#        6.49584998e+00, 2.59842743e+01, 9.79202378e+01]))
# t_solve_L.append(np.array([6.44853500e-03, 6.62292600e-03, 7.28711300e-03, 8.65784100e-03,
#        1.61171100e-02, 4.25822420e-02, 1.72924669e-01]))
# labels_L.append('aligned 1:4')
# NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_L.append(4)


# ##### px = py = 0.1, seed = 42, no VCI #####

# # StraightMapping()
# # NQX = 1
# E_2_R.append(np.array([7.07448695e-01, 6.35011387e-01, 2.90646457e-01, 8.61810504e-02,
#        2.24596145e-02, 5.67433074e-03, 1.42485852e-03]))
# E_inf_R.append(np.array([1.00477395e+00, 9.00660009e-01, 4.15355527e-01, 1.23879062e-01,
#        3.30582064e-02, 8.39822500e-03, 2.12766492e-03]))
# t_setup_R.append(np.array([1.72929108e-01, 9.51024580e-02, 3.74753928e-01, 1.50016260e+00,
#        6.10599585e+00, 2.40968839e+01, 9.49774581e+01]))
# t_solve_R.append(np.array([8.93239460e-02, 5.54534370e-02, 4.34385540e-02, 7.00512000e-02,
#        2.03555422e-01, 7.72443960e-01, 3.62231795e+00]))
# labels_R.append('unaligned 1:1')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)

# # LinearMapping(1.0)
# # NQX = 1
# E_2_R.append(np.array([4.17828953e-02, 1.16959925e-02, 4.84938436e-03, 1.83840883e-03,
#        5.69563649e-04, 1.65625100e-04, 4.48447058e-05]))
# E_inf_R.append(np.array([7.90334238e-02, 2.81271867e-02, 1.38725295e-02, 5.99212924e-03,
#        1.98861723e-03, 6.45249417e-04, 2.11720070e-04]))
# t_setup_R.append(np.array([2.78115400e-02, 1.04515585e-01, 4.06629675e-01, 1.62032077e+00,
#        6.50896160e+00, 2.60520799e+01, 9.71830533e+01]))
# t_solve_R.append(np.array([2.87081700e-02, 3.01700220e-02, 3.62924990e-02, 6.13144450e-02,
#        2.27313199e-01, 8.35398510e-01, 2.95626823e+00]))
# labels_R.append('aligned 1:1')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)

# # LinearMapping(1.0)
# # NQX = 4
# E_2_R.append(np.array([8.64329058e-03, 2.75222621e-03, 1.05897604e-03, 2.54197731e-04,
#        6.54424224e-05, 2.04380058e-05, 5.53069160e-06]))
# E_inf_R.append(np.array([1.51876258e-02, 7.95694310e-03, 3.70298310e-03, 9.74585857e-04,
#        2.96034005e-04, 1.13593687e-04, 3.05105083e-05]))
# t_setup_R.append(np.array([9.99896600e-02, 3.97597781e-01, 7.65955867e-01, 2.04888877e+00,
#        8.44476181e+00, 3.34125533e+01, 2.72867091e+02]))
# t_solve_R.append(np.array([2.83708200e-02, 3.03293850e-02, 1.63050570e-02, 2.78716960e-02,
#        9.28489360e-02, 3.32388050e-01, 2.83546340e+00]))
# labels_R.append('aligned 1:4')
# NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_R.append(4)

# # LinearMapping(1.0)
# # NQX = 16
# E_2_R.append(np.array([7.17585490e-04, 2.39463363e-04, 6.26711350e-05, 1.63379027e-05,
#        4.73689815e-06, 1.23219168e-06]))
# E_inf_R.append(np.array([1.78151024e-03, 6.40128599e-04, 2.51058561e-04, 6.38675550e-05,
#        2.01964677e-05, 6.80074294e-06]))
# t_setup_R.append(np.array([5.24847981e-01, 2.09586884e+00, 1.83216352e+01, 7.19638908e+01,
#        3.49286947e+02, 1.40086923e+03]))
# t_solve_R.append(np.array([1.11414680e-02, 1.46026940e-02, 2.11085850e-02, 6.27990880e-02,
#        3.80353359e-01, 2.02687558e+00]))
# labels_R.append('aligned 1:16')
# NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY_R.append(16)


# ##### px = py = 0.1, seed = 42, VC1-C (whole domain) using ssqr.min2norm #####

# # StraightMapping()
# # NQX = 1
# E_2_R.append(np.array([7.07524521e-01, 6.33525202e-01, 2.87448749e-01, 8.49683404e-02,
#        2.21307106e-02, 5.58993718e-03, 1.40338321e-03]))
# E_inf_R.append(np.array([1.00002761e+00, 9.00786545e-01, 4.07545401e-01, 1.21710495e-01,
#        3.21437603e-02, 8.14469363e-03, 2.04940777e-03]))
# t_setup_R.append(np.array([1.55646440e-02, 5.53931360e-02, 2.32634438e-01, 2.05574035e+00,
#        9.34149883e+00, 4.46373785e+01, 1.36979815e+02]))
# t_solve_R.append(np.array([1.02515560e-02, 1.57257540e-02, 2.06208690e-02, 6.84402180e-02,
#        1.97079202e-01, 7.34847391e-01, 2.89280845e+00]))
# labels_R.append('unaligned 1:1')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)

# # LinearMapping(1.0)
# # NQX = 1
# E_2_R.append(np.array([4.20588066e-02, 1.20344639e-02, 4.62066701e-03, 1.81373936e-03,
#        5.69312090e-04, 1.69265068e-04, 4.56623421e-05]))
# E_inf_R.append(np.array([8.03316995e-02, 2.93828184e-02, 1.36057581e-02, 6.23217764e-03,
#        2.08326721e-03, 7.01409572e-04, 2.23593090e-04]))
# t_setup_R.append(np.array([1.71393950e-02, 6.11791060e-02, 2.55736097e-01, 9.90431428e-01,
#        3.91592057e+00, 1.75384460e+01, 8.55581658e+01]))
# t_solve_R.append(np.array([1.30258670e-02, 1.38500570e-02, 1.72923100e-02, 3.24152890e-02,
#        1.26708600e-01, 8.32324660e-01, 3.83949834e+00]))
# labels_R.append('aligned 1:1')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)

# # LinearMapping(1.0)
# # NQX = 1
# E_2_R.append(np.array([8.42483119e-03, 2.79679089e-03, 1.03154046e-03, 2.54227843e-04,
#        6.35208213e-05, 2.05660374e-05, 5.49806304e-06]))
# E_inf_R.append(np.array([1.54179560e-02, 7.93045625e-03, 3.66067429e-03, 9.91423646e-04,
#        2.83100128e-04, 1.20429956e-04, 2.96728190e-05]))
# t_setup_R.append(np.array([6.19423580e-02, 5.24986616e-01, 2.88658111e+00, 1.16108029e+01,
#        4.73325182e+01, 1.30713728e+02, 3.17532813e+02]))
# t_solve_R.append(np.array([1.24272890e-02, 3.07168760e-02, 3.54864830e-02, 5.71163310e-02,
#        1.70405363e-01, 3.56370156e-01, 2.24756884e+00]))
# labels_R.append('aligned 1:4')
# NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_R.append(4)

# # LinearMapping(1.0)
# # NQX = NY/NX = 16
# E_2_R.append(np.array([7.16175566e-04, 2.39188298e-04, 6.28228565e-05, 1.62826519e-05,
#         4.72266736e-06]))
# E_inf_R.append(np.array([1.78190802e-03, 6.40973740e-04, 2.52334466e-04, 6.35779325e-05,
#         2.02935252e-05]))
# t_setup_R.append(np.array([ 2.84478084e+00,  1.14115362e+01,  2.55662289e+01,  1.72343341e+02,
#         5.09364385e+02]))
# t_solve_R.append(np.array([ 2.56719650e-02,  3.20457310e-02,  4.32489410e-02,  6.20164860e-02,
#         2.70904149e-01]))
# labels_R.append('aligned 1:16')
# NX_R.append(np.array([ 2,  4,  8, 16, 32]))
# NY_R.append(16)

# # VC1-C (slice-by-slice) using ssqr.min2norm
# # LinearMapping(1.0)
# # NQX = NY/NX = 16
# E_2_R.append(np.array([7.23275854e-04, 2.40099354e-04, 6.25889898e-05, 1.63201448e-05,
#        4.73955648e-06, 1.23180266e-06]))
# E_inf_R.append(np.array([1.79888294e-03, 6.46969448e-04, 2.49035940e-04, 6.66908661e-05,
#        2.02669928e-05, 6.74787023e-06]))
# t_setup_R.append(np.array([2.57224262e+00, 1.21039871e+01, 2.33930650e+01, 1.23564061e+02,
#        3.33406841e+02, 1.38262239e+03]))
# t_solve_R.append(np.array([2.48715700e-02, 3.09756570e-02, 2.01477720e-02, 6.27698010e-02,
#        2.67013017e-01, 2.46739756e+00]))
# labels_R.append('aligned 1:16')
# NX_R.append(np.array([  2,  4,  8, 16, 32, 64]))
# NY_R.append(16)


#%% Plotting

solid_linewidth = 1.25
dashed_linewidth = 1.0

plt.rc('markers', fillstyle='full')
plt.rc('lines', markersize=5.0)
plt.rc('pdf', fonttype=42)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
# plt.rc('font', size='small')
plt.rc('legend', fontsize='small')
# plt.rc('axes', titlesize='medium', labelsize='medium')
# plt.rc('xtick', labelsize='small')
# plt.rc('ytick', labelsize='small')
# plt.rc('figure', titlesize='large')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'
purple = '#9467bd'
brown = '#8c564b'
pink = '#e377c2'
grey = '#7f7f7f'
yellow = '#bcbd22'
cyan = '#17becf'
black = '#000000'

# Two plots
fig = plt.figure(figsize=(3.5, 3))
axR1 = fig.subplots(1, 1)

# # Two plots
# fig = plt.figure(figsize=(7.75, 3))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# axL1, axR1 = fig.subplots(1, 2)

# if len(E_2_L) == 1:
#     cycler = plt.cycler(color=[black], marker=['d'])
# elif len(E_2_L) < 4: # 2 and 3
#     cycler = plt.cycler(color=[blue, red, black], marker=['o', 's', 'd'])
# else: # 4 or more
#     cycler = plt.cycler(color=[blue, red, orange, black, green] + colors[4:],
#         marker=['o', 's', '^', 'd', 'x', '+', 'v', '<', '>', '*', 'p'])

# axL1.set_prop_cycle(cycler)

# N_L = []
# inds_L = []
# for i, error in enumerate(E_2_L):
#     N_L.append(np.log2(NY_L[i]*NX_L[i]**2).astype('int'))
#     inds_L.append(N_L[i] >= 2)
#     axL1.semilogy(N_L[i][inds_L[i]], error[inds_L[i]]/(2*np.pi), label=labels_L[i],
#                   linewidth=solid_linewidth)
# # axL1.minorticks_off()
# Nmin = min([min(N_L[i]) for i in range(len(N_L))])
# Nmax = max([max(N_L[i]) for i in range(len(N_L))])
# Nstep = 2
# axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
# axL1.set_title(r'Uniform Grid')
# axL1.set_xlabel(r'$\log_2(N_xN_y)$')
# axL1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
# axL1.legend(loc='lower left')
# xlim = axL1.get_xlim()

if len(E_2_R) == 1:
    cycler = plt.cycler(color=[black], marker=['d'])
elif len(E_2_R) < 4: # 2 and 3
    cycler = plt.cycler(color=[blue, red, black], marker=['o', 's', 'd'])
else: # 4 or more
    cycler = plt.cycler(color=[blue, red, orange, black, green] + colors[4:],
        marker=['o', 's', '^', 'd', 'x', '+', 'v', '<', '>', '*', 'p'])

axR1.set_prop_cycle(cycler)

N_R = []
inds_R = []
for i, error in enumerate(E_2_R):
    N_R.append(np.log2(NY_R[i]*NX_R[i]**2).astype('int'))
    inds_R.append(N_R[i] >= 2)
    axR1.semilogy(N_R[i][inds_R[i]], error[inds_R[i]]/(2*np.pi), label=labels_R[i],
                  linewidth=solid_linewidth)
Nmin = min([min(N_R[i]) for i in range(len(N_R))])
Nmax = max([max(N_R[i]) for i in range(len(N_R))])
Nstep = 2
axR1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
# axR1.set_xlim(xlim)
# axR1.set_title(r'Perturbed Grid (up to 10\%)')
axR1.set_xlabel(r'$\log_2(N_xN_y)$')
axR1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axR1.legend(loc='lower left')

# ordb = 0
# ordt = 3
# ordstep = 0.5
# axR1.set_ylim(ordb, ordt)
# axR1.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))
# lines, labels = axR1.get_legend_handles_labels()
# axR1.legend(lines[1:], labels[1:], loc='lower right')

fig.savefig("AD_conv.pdf", bbox_inches = 'tight', pad_inches = 0)