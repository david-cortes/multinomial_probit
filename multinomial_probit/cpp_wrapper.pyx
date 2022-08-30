### BLAS and LAPACK
from scipy.linalg.cython_blas cimport (
    ddot, dcopy, daxpy, dsyr, dgemv, dgemm, dtrmm, dtrsm
)
from scipy.linalg.cython_lapack cimport (
    dlacpy, dpotri, dpotrf
)

ctypedef double (*ddot__)(const int*, const double*, const int*, const double*, const int*) nogil
ctypedef void (*dcopy__)(const int*, const double*, const int*, double*, const int*) nogil
ctypedef void (*daxpy__)(const int*, const double*, const double*, const int*, double*, const int*) nogil
ctypedef void (*dsyr__)(const char*, const int*, const double*, const double*, const int*, double*, const int*) nogil
ctypedef void (*dgemv__)(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*) nogil
ctypedef void (*dgemm__)(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*) nogil
ctypedef void (*dtrmm__)(const char*,const char*,const char*,const char*,const int*,const int*,const double*,const double*,const int*,double*,const int*) nogil
ctypedef void (*dtrsm__)(const char*,const char*,const char*,const char*,const int*,const int*,const double*,const double*,const int*,double*,const int*) nogil


ctypedef void (*dlacpy__)(const char*, const int*, const int*, const double*, const int*, double*, const int*) nogil
ctypedef void (*dpotri__)(const char*,const int*,double*,const int*,const int*) nogil
ctypedef void (*dpotrf__)(const char*, const int*, double*, const int*, const int*) nogil



cdef public double ddot_(const int* a1, const double* a2, const int* a3, const double* a4, const int* a5) nogil:
    return (<ddot__>ddot)(a1, a2, a3, a4, a5)

cdef public void dcopy_(const int* a1, const double* a2, const int* a3, double* a4, const int* a5) nogil:
    (<dcopy__>dcopy)(a1, a2, a3, a4, a5)

cdef public void daxpy_(const int* a1, const double* a2, const double* a3, const int* a4, double* a5, const int* a6) nogil:
    (<daxpy__>daxpy)(a1, a2, a3, a4, a5, a6)

cdef public void dsyr_(const char* a1, const int* a2, const double* a3, const double* a4, const int* a5, double* a6, const int* a7) nogil:
    (<dsyr__>dsyr)(a1, a2, a3, a4, a5, a6, a7)

cdef public void dgemv_(const char* a1, const int* a2, const int* a3, const double* a4, const double* a5, const int* a6, const double* a7, const int* a8, const double* a9, double* a10, const int* a11) nogil:
    (<dgemv__>dgemv)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)

cdef public void dgemm_(const char* a1, const char* a2, const int* a3, const int* a4, const int* a5, const double* a6, const double* a7, const int* a8, const double* a9, const int* a10, const double* a11, double* a12, const int* a13) nogil:
    (<dgemm__>dgemm)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13)

cdef public void dtrmm_(const char* a1,const char* a2,const char* a3,const char* a4,const int* a5,const int* a6,const double* a7,const double* a8,const int* a9,double* a10,const int* a11) nogil:
    (<dtrmm__>dtrmm)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)

cdef public void dtrsm_(const char* a1,const char* a2,const char* a3,const char* a4,const int* a5,const int* a6,const double* a7,const double* a8,const int* a9,double* a10,const int* a11) nogil:
    (<dtrsm__>dtrsm)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)

cdef public void dlacpy_(const char* a1, const int* a2, const int* a3, const double* a4, const int* a5, double* a6, const int* a7) nogil:
    (<dlacpy__>dlacpy)(a1, a2, a3, a4, a5, a6, a7)

cdef public void dpotri_(const char* a1,const int* a2,double* a3,const int* a4,const int* a5) nogil:
    (<dpotri__>dpotri)(a1, a2, a3, a4, a5)

cdef public void dpotrf_(const char* a1, const int* a2, double* a3, const int* a4, const int* a5) nogil:
    (<dpotrf__>dpotrf)(a1, a2, a3, a4, a5)


### The acutal library
import numpy as np
cimport numpy as np
from libcpp cimport bool as bool
import ctypes

cdef extern from "multinomial_probit.h":
    int get_num_mnp_opt_vars(const int k, const int n)
    
    void get_mnp_starting_point(double optvars[], const int k, const int n)

    void get_mnp_prediction_matrices(
        const int k,
        const double * Lflat,
        double * L,
        double * class_Mats,
        double * class_Rhos,
        double * class_vars,
        int * class_check_Rho
    )
    
    double mnp_likelihood(
        const int m, const int k,
        int nthreads,
        const int * y,
        const double * pred,
        const double * Lflat,
        const double * weights
    )

    void mnp_classpred(
        const int m, const int k,
        int nthreads,
        double * out,
        const double * pred,
        const double * class_Mats,
        const double * class_Rhos,
        const double * class_vars,
        const int * class_check_Rho,
        bool logp
    )

    double mnp_fun_grad(
        const int m, const int k,
        int nthreads,
        const bool only_x,
        double * gradX,
        double * gradL,
        const int * y,
        const double * pred,
        const double * Lflat,
        const double * weights
    )

def wrapped_mnp_num_vars(k, n):
    return get_num_mnp_opt_vars(k, n)

def wrapped_mnp_starting_point(k, n):
    cdef np.ndarray[double, ndim=1] optvars = np.empty(get_num_mnp_opt_vars(k, n))
    get_mnp_starting_point(&optvars[0], k, n)
    return optvars

def wrapped_get_mnp_prediction_matrices(np.ndarray[double, ndim=1] Lflat, int k):
    cdef np.ndarray[double, ndim=2] L = np.empty((k,k))
    cdef np.ndarray[double, ndim=3] class_Mats = np.empty((k,k-1,k))
    cdef np.ndarray[double, ndim=3] class_Rhos = np.empty((k,k-1,k-1))
    cdef np.ndarray[double, ndim=2] class_vars = np.empty((k,k-1))
    cdef np.ndarray[int, ndim=1] class_check_Rho = np.empty(k, dtype=ctypes.c_int)
    get_mnp_prediction_matrices(
        k,
        &Lflat[0],
        &L[0,0],
        &class_Mats[0,0,0],
        &class_Rhos[0,0,0],
        &class_vars[0,0],
        &class_check_Rho[0]
    )
    return L, class_Mats, class_Rhos, class_Rhos, class_vars, class_check_Rho

def wrapped_mnp_fun_grad(
    np.ndarray[int, ndim=1] y,
    np.ndarray[double, ndim=2] pred,
    np.ndarray[double, ndim=1] Lflat,
    np.ndarray[double, ndim=1] weights,
    int nthreads,
    bool only_x
):
    cdef int m = y.shape[0]
    cdef int k = pred.shape[1] + 1
    cdef np.ndarray[double, ndim=2] gradX = np.zeros((m,k-1))
    cdef np.ndarray[double, ndim=1] gradL = np.zeros(k+int(k*(k-1)/2) - 1)
    cdef double * weights_ptr = NULL
    if weights.shape[0]:
        weights_ptr = &weights[0]
    cdef double ll = mnp_fun_grad(
        m, k,
        nthreads,
        only_x,
        &gradX[0,0],
        &gradL[0],
        &y[0],
        &pred[0,0],
        &Lflat[0],
        weights_ptr
    )
    return ll, gradX, gradL

def wrapped_mnp_fun(
    np.ndarray[int, ndim=1] y,
    np.ndarray[double, ndim=2] pred,
    np.ndarray[double, ndim=1] Lflat,
    np.ndarray[double, ndim=1] weights,
    int nthreads
):
    cdef int m = y.shape[0]
    cdef int k = pred.shape[1] + 1
    cdef double * weights_ptr = NULL
    if weights.shape[0]:
        weights_ptr = &weights[0]
    return -mnp_likelihood(
        m, k,
        nthreads,
        &y[0],
        &pred[0,0],
        &Lflat[0],
        weights_ptr
    )

def wrapped_mnp_classpred(
    int m, int k, int nthreads,
    np.ndarray[double, ndim=2] pred,
    np.ndarray[double, ndim=3] class_Mats,
    np.ndarray[double, ndim=3] class_Rhos,
    np.ndarray[double, ndim=2] class_vars,
    np.ndarray[int, ndim=1] class_check_Rho,
    bool logp
):
    cdef np.ndarray[double, ndim=2] out = np.empty((m,k))
    mnp_classpred(
        m, k,
        nthreads,
        &out[0,0],
        &pred[0,0],
        &class_Mats[0,0,0],
        &class_Rhos[0,0,0],
        &class_vars[0,0],
        &class_check_Rho[0],
        logp
    )
    return out
