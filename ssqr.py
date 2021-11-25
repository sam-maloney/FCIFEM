# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:35:11 2021

@author: Samuel A. Maloney

"""

import numpy as np
from scipy.sparse import csc_matrix, spmatrix

import cppyy
import cppyy.ll

cppyy.cppdef(r"""
              #define SUITESPARSE_GPU_EXTERN_ON
              #define CHOLMOD_H
              """)
              
try:
    cppyy.c_include('suitesparse\\SuiteSparseQR_definitions.h')
    prefix = 'suitesparse\\'
except(ImportError):
    cppyy.c_include('SuiteSparseQR_definitions.h')
    prefix = ''    

cppyy.c_include(prefix + 'SuiteSparse_config.h')
suffixes = ['io64', 'config', 'core', 'check', 'cholesky', 'matrixops',
            'modify', 'camd', 'partition', 'supernodal']
for suffix in suffixes:
    cppyy.c_include(prefix + 'cholmod_' + suffix + '.h')
cppyy.include(prefix + 'SuiteSparseQR.hpp')
cppyy.include(prefix + 'SuiteSparseQR_C.h')
cppyy.load_library('cholmod')
cppyy.load_library('spqr')
## Initialize cholmod common
cc = cppyy.gbl.cholmod_common()
cppyy.gbl.cholmod_l_start(cc)
## Set up cholmod common deinit to run when Python exits
def _deinit():
    '''Deinitialize the CHOLMOD library.'''
    cppyy.gbl.cholmod_l_finish( cc )
import atexit
atexit.register(_deinit)

##### cholmod defines from cholmod_core.h #####

# itype defines the types of integer used:
CHOLMOD_INT     = 0 # all integer arrays are int
CHOLMOD_INTLONG = 1 # most are int, some are SuiteSparse_long
CHOLMOD_LONG    = 2 # all integer arrays are SuiteSparse_long

# dtype defines what the numerical type is (double or float):
CHOLMOD_DOUBLE = 0 # all numerical values are double
CHOLMOD_SINGLE = 1 # all numerical values are float

# xtype defines the kind of numerical values used:
CHOLMOD_PATTERN = 0 # pattern only, no numerical values
CHOLMOD_REAL    = 1 # real matrix
CHOLMOD_COMPLEX = 2 # a complex matrix (ANSI C99 compatible)
CHOLMOD_ZOMPLEX = 3 # a complex matrix (MATLAB compatible)

##### SPQR defines from SuiteSparseQR_definitions.h #####

# ordering options
SPQR_ORDERING_FIXED   = 0
SPQR_ORDERING_NATURAL = 1
SPQR_ORDERING_COLAMD  = 2
SPQR_ORDERING_GIVEN   = 3 # only used for C/C++ interface
SPQR_ORDERING_CHOLMOD = 4 # CHOLMOD best-effort (COLAMD, METIS,...)
SPQR_ORDERING_AMD     = 5 # AMD(A'*A)
SPQR_ORDERING_METIS   = 6 # metis(A'*A)
SPQR_ORDERING_DEFAULT = 7 # SuiteSparseQR default ordering
SPQR_ORDERING_BEST    = 8 # try COLAMD, AMD, and METIS; pick best
SPQR_ORDERING_BESTAMD = 9 # try COLAMD and AMD; pick best

# tol options
SPQR_DEFAULT_TOL = -2.0 # if tol <= -2, the default tol is used
SPQR_NO_TOL      = -1.0 # if -2 < tol < 0, then no tol is used

# for qmult, method can be 0,1,2,3:
SPQR_QTX = 0
SPQR_QX  = 1
SPQR_XQT = 2
SPQR_XQ  = 3

# system can be 0,1,2,3:  Given Q*R=A*E from SuiteSparseQR_factorize:
SPQR_RX_EQUALS_B    = 0 # solve R*X=B      or X = R\B
SPQR_RETX_EQUALS_B  = 1 # solve R*E'*X=B   or X = E*(R\B)
SPQR_RTX_EQUALS_B   = 2 # solve R'*X=B     or X = R'\B
SPQR_RTX_EQUALS_ETB = 3 # solve R'*X=E'*B  or X = R'\(E'*B)


# helper function for getting a pointer for int64 array outputs
try:
    cppyy.cppdef(r"""
    SuiteSparse_long** create_SSL_pointer() {
        return new SuiteSparse_long*;
    }
    """)
except: # if it's already been defined, ignore error
    pass


class QRfactorization:
    def __init__(self, Zs, Zd, R, E, H, HPinv, HTau):
        self.Zs = Zs
        self.Zd = Zd
        self.R = R
        self.E = E
        self.H = H
        self.HPinv = HPinv
        self.HTau = HTau
    
    def free(self):
        cppyy.gbl.cholmod_l_free_sparse(self.Zs, cc)
        cppyy.gbl.cholmod_l_free_dense(self.Zd, cc)
        cppyy.gbl.cholmod_l_free_sparse(self.R, cc)
        cppyy.gbl.cholmod_l_free_sparse(self.H, cc)
        cppyy.gbl.cholmod_l_free_dense(self.HTau, cc)
        
    def __del__(self):
        self.free()


def free(A):
    try:
        cppyy.gbl.cholmod_l_free_sparse(A, cc)
    except:
        pass
    try:
        cppyy.gbl.cholmod_l_free_dense(A, cc)
    except:
        pass
    return

def scipyCscToCholmodSparse(A):
    if not isinstance(A, csc_matrix):
        raise TypeError("Input matrix must be of type scipy.sparse.csc_matrix")
    if A.indptr.dtype != 'int64':
        raise TypeError("Input matrix must have indices of type int64")
    nrow, ncol = A.shape
    return cppyy.gbl.cholmod_sparse(
        int(nrow), # the matrix is nrow-by-ncol
        int(ncol),
        A.nnz, # nzmax; maximum number of entries in the matrix
        # pointers to int or SuiteSparse_long (int64):
        A.indptr, # *p; [0..ncol], the column pointers
        A.indices, # *i; [0..nzmax-1], the row indices
        cppyy.nullptr, # *nz; for unpacked matrices only
        # pointers to double or float:
        A.data, # *x; size nzmax or 2*nzmax (complex), if present
        cppyy.nullptr, # *z; size nzmax, if present (zomplex)
        0, # stype, Describes what parts of the matrix are considered, 0="unsymmetric": use both upper and lower triangular parts
        CHOLMOD_LONG, # itype; p, i, and nz are int or long
        CHOLMOD_REAL, # xtype; pattern, real, complex, or zomplex
        CHOLMOD_DOUBLE, # dtype; x and z are double or float
        A.has_sorted_indices, # sorted; TRUE if columns are sorted, FALSE otherwise
        True # packed; TRUE if packed (nz ignored), FALSE if unpacked (nz is required)
        )

def cholmodSparseToScipyCsc(chol_A):
    A = csc_matrix((np.iinfo('int32').max + 1, 1)) # forces idx_type = 'int64'
    A.data = np.frombuffer(chol_A.x, dtype=np.float64, count=chol_A.nzmax)
    A.indptr = np.frombuffer(chol_A.p, dtype=np.int64, count=chol_A.ncol+1)
    A.indices = np.frombuffer(chol_A.i, dtype=np.int64, count=chol_A.nzmax)
    A._shape = (chol_A.nrow, chol_A.ncol)
    return A

def checkMatrixEqual(A, chol_A):
    try:
        Ax = np.frombuffer(chol_A.x, dtype=np.float64, count=chol_A.nzmax)
        Ap = np.frombuffer(chol_A.p, dtype=np.int64, count=chol_A.ncol+1)
        Ai = np.frombuffer(chol_A.i, dtype=np.int64, count=chol_A.nzmax)
    except ReferenceError:
        print("Error: cholmod matrix uninitialized")
        return False
    try:
        assert np.allclose(Ax, A.data)
        assert np.array_equal(Ap, A.indptr)
        assert np.array_equal(Ai, A.indices)
    except AssertionError:
        return False
    return True

def QR_C(chol_A, tol=SPQR_DEFAULT_TOL, econ=None):
    if isinstance(chol_A, spmatrix):
        chol_A = scipyCscToCholmodSparse(chol_A)
    if econ is None:
        econ = chol_A.nrow
        
    Zs = cppyy.bind_object(cppyy.nullptr, 'cholmod_sparse')
    Zd = cppyy.bind_object(cppyy.nullptr, 'cholmod_dense')
    R = cppyy.bind_object(cppyy.nullptr, 'cholmod_sparse')
    H = cppyy.bind_object(cppyy.nullptr, 'cholmod_sparse')
    HTau = cppyy.bind_object(cppyy.nullptr, 'cholmod_dense')
    E = cppyy.gbl.create_SSL_pointer()
    HPinv = cppyy.gbl.create_SSL_pointer()
    
    r = cppyy.gbl.SuiteSparseQR_C(
        # inputs, not modified
        SPQR_ORDERING_DEFAULT, # ordering; all, except 3:given treated as 0:fixed
        tol, # tol; only accept singletons above tol
        econ, # econ; number of rows of C and R to return; a value less
              # than the rank r of A is treated as r, and a value greater
              # than m is treated as m.
        0, # getCTX; if 0: return Z = C of size econ-by-bncols
                   # if 1: return Z = C' of size bncols-by-econ
                   # if 2: return Z = X of size econ-by-bncols
        chol_A, # *A; m-by-n sparse matrix
        # B is either sparse or dense.  If Bsparse is non-NULL, B is sparse
        # and Bdense is ignored.  If Bsparse is NULL and Bdense is non-
        # NULL, then B is dense.  B is not present if both are NULL.
        cppyy.nullptr, # *Bsparse
        cppyy.nullptr, # *Bdense
        # output arrays, neither allocated nor defined on input.
        # Z is the matrix C, C', or X
        Zs, # **Zsparse
        Zd, # **Zdense
        R, # **R; the R factor
        E, # **E; size n, fill-reducing ordering of A.
        H, # **H; the Householder vectors (m-by-nh)
        HPinv, # **HPinv; size m, row permutation for H
        HTau, # **HTau; size nh, Householder coefficients
        # workspace and parameters
        cc
        )
    QR = QRfactorization(Zs, Zd, R, E, H, HPinv, HTau)
    return QR, r

def qmult(QR, x, method=SPQR_QX):
    nrow = len(x)
    chol_x = cppyy.gbl.cholmod_dense(
        nrow, 1, # column vector, nrow-by-1
        nrow, # nzmax; maximum number of entries in the matrix
        nrow, # d; leading dimension (d >= nrow must hold)
        x.data, # *x; size nzmax or 2*nzmax (complex), if present
        cppyy.nullptr, # *z; size nzmax, if present (zomplex)
        CHOLMOD_REAL, # xtype; pattern, real, complex, or zomplex
        CHOLMOD_DOUBLE # dtype; x and z double or float
        )
    chol_Qx = cppyy.gbl.SuiteSparseQR_qmult['double'](
        # inputs, no modified
        method, # method; 0,1,2,3
        QR.H, # *H; either m-by-nh or n-by-nh
        QR.HTau, # *HTau; size 1-by-nh
        QR.HPinv[0], # *HPinv; size mh
        chol_x, # *Xdense; size m-by-n
        # workspace and parameters
        cc
        )
    return np.frombuffer(chol_Qx.x, dtype=np.float64, count=chol_Qx.nzmax)

# chol_Q = cppyy.bind_object(cppyy.nullptr, 'cholmod_sparse')
# chol_R = cppyy.bind_object(cppyy.nullptr, 'cholmod_sparse')
# chol_E = cppyy.gbl.create_SSL_pointer()        
# r = cppyy.gbl.SuiteSparseQR_C_QR( # returns rank(A) est., (-1) if failure
#     # inputs:
#     SPQR_ORDERING_DEFAULT, # ordering; all, except 3:given treated as 0:fixed
#     SPQR_DEFAULT_TOL, # tol; columns with 2-norm <= tol treated as 0
#     nrow, # econ; e = max(min(m,econ),rank(A))
#     chol_A, # *A; m-by-n sparse matrix to factorize
#     # outputs:
#     chol_Q, # **Q, m-by-e sparse matrix
#     chol_R, # **R, e-by-n sparse matrix
#     chol_E, # **E, size n column perm, NULL if identity
#     cc # workspace and parameters
#     )
# cppyy.gbl.cholmod_l_free_sparse(chol_Q, cc)
# cppyy.gbl.cholmod_l_free_sparse(chol_R, cc)