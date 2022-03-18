# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:35:11 2021

@author: Samuel A. Maloney

"""

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix, isspmatrix_csc, SparseEfficiencyWarning
from warnings import warn

import cppyy
import cppyy.ll

cppyy.cppdef(r"""
              #define SUITESPARSE_GPU_EXTERN_ON
              #define CHOLMOD_H
              """)

try:
    cppyy.c_include('SuiteSparseQR_definitions.h')
    prefix = ''
except(ImportError):
    try:
        cppyy.c_include('suitesparse/SuiteSparseQR_definitions.h')
        prefix = 'suitesparse/'
    except(ImportError):
        cppyy.c_include('suitesparse\\SuiteSparseQR_definitions.h')
        prefix = 'suitesparse\\'

cppyy.c_include(prefix + 'SuiteSparse_config.h')
suffixes = ['io64', 'config', 'core', 'check', 'cholesky', 'matrixops',
            'modify', 'camd', 'partition', 'supernodal']
for suffix in suffixes:
    cppyy.c_include(prefix + 'cholmod_' + suffix + '.h')
cppyy.include(prefix + 'SuiteSparseQR.hpp')
cppyy.include(prefix + 'SuiteSparseQR_C.h')
cppyy.load_library('cholmod') # requires cudatoolkit (libcudart.so)
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
    def __init__(self, Zs, Zd, R, E, H, HPinv, HTau, r):
        self.Zs = Zs
        self.Zd = Zd
        self.R = R
        self.E = E
        self.H = H
        self.HPinv = HPinv
        self.HTau = HTau
        self.r = r

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

def scipyCscToCholmodSparse(X, Z=None, forceInt64=False):
    if not isspmatrix(X):
        raise TypeError('Input matrix must be a sublcass of scipy.sparse '
                        'spmatrix')
    if not isspmatrix_csc(X):
        warn('Input matrix should be of type scipy.sparse.csc_matrix to avoid '
             'conversion', SparseEfficiencyWarning)
        X = X.tocsc()
    nrow, ncol = X.shape
    indptr = X.indptr
    indices = X.indices
    if forceInt64:
        if (X.indptr.dtype != 'int64') or (X.indices.dtype != 'int64'):
            warn('Input matrix should have all indices of type int64 to avoid '
                 'copying index arrays', SparseEfficiencyWarning)
        indptr = X.indptr.astype('int64', copy=False)
        indices = X.indices.astype('int64', copy=False)
        itype = CHOLMOD_LONG
    else:
        if X.indptr.dtype == 'int32':
            if X.indices.dtype == 'int32':
                itype = CHOLMOD_INT
            else: # X.indices.dtype == 'int64'
                warn('If indices.dtype of input matrix is int64 then '
                     'indptr.dtype should also be int64 to avoid copying '
                     'indptr array', SparseEfficiencyWarning)
                indptr = X.indptr.astype('int64')
                itype = CHOLMOD_LONG
        else: # X.indptr.dtype == 'int64'
            if X.indices.dtype == 'int32':
                itype = CHOLMOD_INTLONG
            else: # X.indices.dtype == 'int64'
                itype = CHOLMOD_LONG
    if Z is None:
        _Z = cppyy.nullptr
        if np.iscomplexobj(X):
            xtype = CHOLMOD_COMPLEX
        else:
            xtype = CHOLMOD_REAL
    else: # Z is not None
        if X.shape != Z.shape:
            raise TypeError('Real and imaginary matrices must have same shape')
        if X.dtype != Z.dtype:
            raise TypeError('Real and imaginary matrices must have same dtype')
        _Z = Z.data
        xtype = CHOLMOD_ZOMPLEX
    if (X.dtype == 'float64') or (X.dtype == 'complex128'):
        dtype = CHOLMOD_DOUBLE
    elif (X.dtype == 'float32') or (X.dtype == 'complex64'):
        dtype = CHOLMOD_SINGLE
    else:
        raise TypeError('Input matrix must have floating point or complex '
                        'dtype')
    return cppyy.gbl.cholmod_sparse(
        int(nrow), # the matrix is nrow-by-ncol
        int(ncol),
        int(X.nnz), # nzmax; maximum number of entries in the matrix
        # pointers to int or SuiteSparse_long (int64):
        indptr, # *p; [0..ncol], the column pointers
        indices, # *i; [0..nzmax-1], the row indices
        cppyy.nullptr, # *nz; for unpacked matrices only
        # pointers to double or float:
        X.data, # *x; size nzmax or 2*nzmax (complex), if present
        _Z, # *z; size nzmax, if present (zomplex)
        0, # stype, Describes what parts of the matrix are considered:
           # 0:  matrix is "unsymmetric": use both upper and lower triangular parts
           #     the matrix may actually be symmetric in pattern and value, but
 	       #     both parts are explicitly stored and used).  May be square or
           #     rectangular.
           # >0: matrix is square and symmetric, use upper triangular part.
           #     Entries in the lower triangular part are ignored.
           # <0: matrix is square and symmetric, use lower triangular part.
           #     Entries in the upper triangular part are ignored.
        itype, # itype; CHOLMOD_INT:     p, i, and nz are int.
			   #        CHOLMOD_INTLONG: p is SuiteSparse_long, i and nz are int.
			   #        CHOLMOD_LONG:    p, i, and nz are SuiteSparse_long
        xtype, # xtype; pattern, real, complex, or zomplex
        dtype, # dtype; x and z are double or float
        X.has_sorted_indices, # sorted; TRUE if columns are sorted, FALSE otherwise
        True # packed; TRUE if packed (nz ignored), FALSE if unpacked (nz is required)
        )

def cholmodSparseToScipyCsc(chol_A):
    if not isinstance(chol_A, cppyy.gbl.cholmod_sparse_struct):
        raise TypeError('Input must be a cholmod_sparse_struct')
    A = csc_matrix((np.iinfo('int32').max + 1, 1)) # forces idx_type = 'int64'
    A.data = np.frombuffer(chol_A.x, dtype=np.float64, count=chol_A.nzmax)
    A.indptr = np.frombuffer(chol_A.p, dtype=np.int64, count=chol_A.ncol+1)
    A.indices = np.frombuffer(chol_A.i, dtype=np.int64, count=chol_A.nzmax)
    A._shape = (chol_A.nrow, chol_A.ncol)
    return A

def checkMatrixEqual(A, B):
    if isspmatrix_csc(A):
        Ax = A.data
        Ap = A.indptr
        Ai = A.indices
    else:
        try:
            Ax = np.frombuffer(A.x, dtype=np.float64, count=A.nzmax)
            Ap = np.frombuffer(A.p, dtype=np.int64, count=A.ncol+1)
            Ai = np.frombuffer(A.i, dtype=np.int64, count=A.nzmax)
        except ReferenceError:
            print('Error: Input must be of type scipy.sparse.csc_matrix or a '
                  'cholmod_sparse_struct')
            return False
    if isspmatrix_csc(B):
        Bx = B.data
        Bp = B.indptr
        Bi = B.indices
    else:
        try:
            Bx = np.frombuffer(B.x, dtype=np.float64, count=B.nzmax)
            Bp = np.frombuffer(B.p, dtype=np.int64, count=B.ncol+1)
            Bi = np.frombuffer(B.i, dtype=np.int64, count=B.nzmax)
        except ReferenceError:
            print('Error: Input must be of type scipy.sparse.csc_matrix or a '
                  'cholmod_sparse_struct')
            return False
    try:
        assert np.allclose(Ax, Bx)
        assert np.array_equal(Ap, Bp)
        assert np.array_equal(Ai, Bi)
    except AssertionError:
        return False
    return True

def numpyArrayToCholmodDense(x, z=None):
    if x.ndim > 2:
        raise TypeError('Input array cannot have more than 2 dimensions')
    elif x.ndim == 1:
        # Assumes column vector; specify (1,n) shape to get row vector
        nrow = len(x)
        ncol = 1
    else: # x.ndim == 2
        nrow, ncol = x.shape
    if z is None:
        _z = cppyy.nullptr
        if np.iscomplexobj(x):
            xtype = CHOLMOD_COMPLEX
        else:
            xtype = CHOLMOD_REAL
    else: # z is not None
        if x.shape != z.shape:
            raise TypeError('Real and imaginary arrays must have same shape')
        if x.dtype != z.dtype:
            raise TypeError('Real and imaginary arrays must have same dtype')
        _z = z.data
        xtype = CHOLMOD_ZOMPLEX
    if (x.dtype == 'float64') or (x.dtype == 'complex128'):
        dtype = CHOLMOD_DOUBLE
    elif (x.dtype == 'float32') or (x.dtype == 'complex64'):
        dtype = CHOLMOD_SINGLE
    else:
        raise TypeError('Input array(s) must have floating point or complex '
                        'dtype')
    return cppyy.gbl.cholmod_dense(
        nrow,   # the matrix is nrow-by-ncol
        ncol,
        x.size, # nzmax; maximum number of entries in the matrix
        nrow,   # d;     leading dimension (d >= nrow must hold)
        x.data, # *x;    size nzmax or 2*nzmax (complex), if present
        _z,     # *z;    size nzmax, if present (zomplex)
        xtype,  # xtype; pattern, real, complex, or zomplex
        dtype   # dtype; x and z double or float
        )

def cholmodDenseToNumpyArray(chol_x):
    if not isinstance(chol_x, cppyy.gbl.cholmod_dense_struct):
        raise TypeError('Input must be a cholmod_dense_struct')
    z = None
    if chol_x.xtype == CHOLMOD_REAL:
        if chol_x.dtype == CHOLMOD_DOUBLE:
            x = np.frombuffer(chol_x.x, dtype=np.float64, count=chol_x.nzmax)
        else: # chol_x.dtype == CHOLMOD_SINGLE
            x = np.frombuffer(chol_x.x, dtype=np.float32, count=chol_x.nzmax)
    elif chol_x.xtype == CHOLMOD_COMPLEX:
        if chol_x.dtype == CHOLMOD_DOUBLE:
            x = np.frombuffer(chol_x.x, dtype=np.complex128, count=chol_x.nzmax)
        else: # chol_x.dtype == CHOLMOD_SINGLE
            x = np.frombuffer(chol_x.x, dtype=np.complex64, count=chol_x.nzmax)
    if chol_x.xtype == CHOLMOD_ZOMPLEX:
        if chol_x.dtype == CHOLMOD_DOUBLE:
            x = np.frombuffer(chol_x.x, dtype=np.float64, count=chol_x.nzmax)
            z = np.frombuffer(chol_x.z, dtype=np.float64, count=chol_x.nzmax)
        else: # chol_x.dtype == CHOLMOD_SINGLE
            x = np.frombuffer(chol_x.x, dtype=np.float32, count=chol_x.nzmax)
            z = np.frombuffer(chol_x.z, dtype=np.float32, count=chol_x.nzmax)
    x.resize(chol_x.nrow, chol_x.ncol)
    if z is None:
        return x
    else:
        z.resize(chol_x.nrow, chol_x.ncol)
        return x, z

def QR_C(A, ordering=SPQR_ORDERING_DEFAULT, tol=SPQR_DEFAULT_TOL, econ=None):
    if isinstance(A, cppyy.gbl.cholmod_sparse_struct):
        chol_A = A
    else:
        chol_A = scipyCscToCholmodSparse(A)
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
        ordering, # ordering; all, except 3:given treated as 0:fixed
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
        cc # workspace and parameters
        )
    QR = QRfactorization(Zs, Zd, R, E, H, HPinv, HTau, r)
    return QR

def qmult(QR, x, method=SPQR_QX):
    returnNumpy = False
    if ( isinstance(x, cppyy.gbl.cholmod_dense_struct) or
         isinstance(x, cppyy.gbl.cholmod_sparse_struct) ):
        chol_x = x
    elif isinstance(x, np.ndarray):
        chol_x = numpyArrayToCholmodDense(x)
        returnNumpy = True
    else:
        chol_x = scipyCscToCholmodSparse(x)
        returnNumpy = True

    if chol_x.dtype == CHOLMOD_DOUBLE:
        dtype = 'double'
    else:
        dtype = 'float'

    chol_Qx = cppyy.gbl.SuiteSparseQR_qmult[dtype](
        # inputs, no modified
        method, # method; 0,1,2,3
        QR.H, # *H; either m-by-nh or n-by-nh
        QR.HTau, # *HTau; size 1-by-nh
        QR.HPinv[0], # *HPinv; size mh
        chol_x, # *Xdense; size m-by-n
        cc # workspace and parameters
        )

    if returnNumpy:
        if isinstance(chol_Qx, cppyy.gbl.cholmod_dense_struct):
            return cholmodDenseToNumpyArray(chol_Qx)
        else: # isinstance(chol_Qx, cppyy.gbl.cholmod_sparse_struct)
            return cholmodSparseToScipyCsc(chol_Qx)
    else:
        return chol_Qx

def min2norm(A, b, ordering=SPQR_ORDERING_DEFAULT, tol=SPQR_DEFAULT_TOL):
    returnNumpy = False
    if isinstance(A, cppyy.gbl.cholmod_sparse_struct):
        chol_A = A
    else:
        chol_A = scipyCscToCholmodSparse(A, forceInt64=True)
        returnNumpy = True

    if ( isinstance(b, cppyy.gbl.cholmod_dense_struct) or
         isinstance(b, cppyy.gbl.cholmod_sparse_struct) ):
        chol_b = b
    elif isinstance(b, np.ndarray):
        chol_b = numpyArrayToCholmodDense(b)
        returnNumpy = True
    else:
        chol_b = scipyCscToCholmodSparse(b)
        returnNumpy = True

    if chol_A.dtype != chol_b.dtype:
        raise TypeError('Both inputs must have the same dtype')

    if chol_A.dtype == CHOLMOD_DOUBLE:
        dtype = 'double'
    else:
        dtype = 'float'

    chol_x = cppyy.gbl.SuiteSparseQR_min2norm[dtype](
        ordering, # ordering; all, except 3:given treated as 0:fixed
        tol, # tol; only accept singletons above tol
        chol_A,
        chol_b,
        cc # workspace and parameters
        )

    if returnNumpy:
        if isinstance(chol_x, cppyy.gbl.cholmod_dense_struct):
            return cholmodDenseToNumpyArray(chol_x)
        else: # isinstance(chol_x, cppyy.gbl.cholmod_sparse_struct)
            return cholmodSparseToScipyCsc(chol_x)
    else:
        return chol_x

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
