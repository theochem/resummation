import numpy as np
import itertools as it
from scipy.linalg import toeplitz

class Pade(object):
    """A Pade approximant P(M,N) of a truncated Taylor series.

    Attributes
    ----------
    coeffs : np.ndarray
        Expansion coefficients of the divergent serie.

    M : int
        Order of the numerator.

    N : int
        Order of the denominator.

    a : np.ndarray
        Expansion coefficients of the numerator.

    b : np.ndarray
        Expansion coefficients of the dinominator.

    tol: float
        Tolerance of the pade method.

    Methods
    -------
    __init__(self, M, N, coeffs, a, b)
        Initilize the instance.

    __call__(self, x)
        Call the pade function.

    Properties
    ----------
    poles(self)
        Singularity of the pade function.

    zeros(self)
        A zero of a meromorphic function f is a complex number z such that f(z) = 0.

    residues(self)
        Estimate of residues.

    ClassMethods
    ------------
    plain(cls, M, N, coeffs, tol=0)
        Plain Pade approximant.

    robust(cls, M, N, coeffs, tol=10e-14)
        Robust Pade approximant.
    """
    def __init__(self, coeffs, M, N, a, b, tol):
        """Initilize the instance.

        Parameters
        ----------
        coeffs : np.ndarray
            Expansion coefficients of the divergent serie.

        M : int
            Order of the numerator.

        N : int
            Order of the denominator.

        a : np.ndarray
            Expansion coefficients of the numerator.

        b : np.ndarray
            Expansion coefficients of the dinominator.
        
        tol: float
            Tolerance of the pade method.
        """
        self.coeffs = coeffs
        self.M = M
        self.N = N
        self.a = a
        self.b = b
        self.tol = tol

    def __call__(self, z):
        """Call the pade function.

        Parameters
        ----------
        z : (int, float)
            Variable of the pade function.

        Raises
        ------
        TypeError
            If z is not int or float.

        Returns
        -------
        func
            Pade functiion.
        """
        if not (isinstance(z, (int, float))):
            raise TypeError("Parameter z must be int or float.")
        a, b = self.a, self.b
        a = np.array([*reversed(a)])
        b = np.array([*reversed(b)])
        func = np.polyval(a, z) / np.polyval(b, z)
        return func

    @classmethod
    def plain(cls, coeffs, M, N, tol=0):
        """Approximant Pade coefficients from Taylor series coefficients.

        Parameters
        ----------
        coeffs : np.ndarray
            Expansion coefficients of the divergent serie.

        M : int
            Order of the numerator.

        N : int
            Order of the denominator.

        Raises
        ------
        TypeError
            If `coefficients` is not a 1-dimensional `numpy` array with `dtype` float.
            If `order` is not an integer.

        ValueError
            If dimension of `coefficients` is less than N + 1. Nth order Meijer-G requires N + 1
            coefficients of the serie.        

        Returns
        -------
        pade : Pade
            Instance of Pade class.
        """
        if not (isinstance(coeffs, np.ndarray)
            and coeffs.ndim == 1
            and coeffs.dtype == float
        ):
            raise TypeError("Coefficients must be a one-dimensional `numpy` array with `dtype` float.")
        if not (isinstance(M, int) and isinstance(N, int)):
            raise TypeError("Order M and N given must be integer.")
        if coeffs.shape[0] < (M+N+1):
            raise ValueError("Number of coefficients must be higher than order of approximant + 1.")

        if M <= N:
            C = np.zeros((N, N))
            for i, j in it.product(range(N), range(N)):
                k = M + i - j
                if k >= 0:
                    C[i, j] = coeffs[k]

            c = np.zeros(N)
            for k in range(N):
                c[k] = -coeffs[M + k + 1]

            b = np.linalg.solve(C, c)
            b = np.array([1, *b.tolist()])
            a = np.zeros(M + 1)
            for m in range(M + 1):
                for k in range(m + 1):
                    a[m] += b[m - k] * coeffs[k]

            a = [*a.tolist()]
            b = [*b.tolist()]
        else:
            C = np.zeros((M + 1, M + 1))
            for i, j in it.product(range(M + 1), range(M + 1)):
                if j < M - N + 1:
                    if i == j:
                        C[i, j] = -1
                else:
                    C[i, j] = coeffs[M + i - j]

            c = np.zeros(M + 1)
            for k in range(M + 1):
                c[k] = -coeffs[N + k]

            w = np.linalg.solve(C, c)
            a = np.zeros(M + 1)
            b = np.zeros(N + 1)
            b[0] = 1
            for k in range(M + 1):
                if k < M - N + 1:
                    a[k + N] = w[k]
                else:
                    b[k - M + N] = w[k]

            for m in range(N):
                for k in range(min(m, N) + 1):
                    a[m] += b[m - k] * coeffs[k]
                    
            a = a.tolist()
            b = b.tolist()

        pade = cls(coeffs, M, N, a, b, tol)
        return pade

    @classmethod
    def robust(cls, coeffs, M, N, tol=1e-14):
        """Approximant Pade coefficients from Taylor series coefficients.

        Parameters
        ----------
        coeffs : np.ndarray
            Expansion coefficients of the divergent serie.

        M : int
            Order of the numerator.

        N : int
            Order of the denominator.

        Raises
        ------
        TypeError
            If `coefficients` is not a 1-dimensional `numpy` array with `dtype` float.
            If `order` is not an integer.

        ValueError
            If dimension of `coefficients` is less than N + 1. Nth order Meijer-G requires N + 1
            coefficients of the serie.        

        Returns
        -------
        pade : Pade
            Instance of Pade class.
        """
        if not (isinstance(coeffs, np.ndarray)
            and coeffs.ndim == 1
            and coeffs.dtype == float
        ):
            raise TypeError("Coefficients must be a one-dimensional `numpy` array with `dtype` float.")
        if not (isinstance(M, int) and isinstance(N, int)):
            raise TypeError("Order M and N given must be integer.")
        if coeffs.shape[0] < (M+N+1):
            raise ValueError("Number of coefficients must be higher than order of approximant + 1.")

        eps = np.finfo(np.float).eps

        def discard_trailing_epsilon(v):
            if np.abs(v)[-1] <= tol:
                return v[ : np.where( np.abs(v) > tol )[0][-1] + 1 ]
            else:
                return v
    
        #  Make sure c is long enough but not longer than necessary.
        c = coeffs[:M + N + 1]
        if len(c) < M + N + 1:
            c = np.pad(coeffs, (0, M + N + 1 - len(c)), 'constant')
    
        #  Compute absolute tolerance.
        ts = tol*np.linalg.norm(c)
    
        ## Compute the Pade approximation.

        # check for the special case r = 0
        if ( np.linalg.norm(c[0:M+1], np.inf) <= tol*np.linalg.norm(c, np.inf) ):
            a = 0
            b = 1
            mu = -np.inf
            nu = 0
        else:
            ## the general case
    
            #  First row/column of Toeplitz matrix.
            row = np.zeros(N+1)
            row[0] = c[0]
            col = c
    
            #  Do diagonal hopping across block.
            while True:
                #  Special case n == 0.
                if N == 0:
                    a = c[0:M+1]
                    b = 1
                    break
    
                #  Form Toeplitz matrix.
                Z = toeplitz(col[0:M+N+1], row[0:N+1])
    
                #  Compute numerical rank.
                C = Z[M+1:M+N+1,:]
                rho = np.sum(np.linalg.svd(C, compute_uv=False ) > ts)
                if rho == N:
                    break
    
                #  Decrease mn, n if rank-deficient.
                M = M - (N - rho)
                N = rho
    
            #  Hopping finished. Now compute b and a.
            if N > 0:
                U, S, V = np.linalg.svd(C)

                #  Null vector gives b.
                b = V[:,N]
    
                #  Do final computation via reweighted QR for better zero preservation.
                D = np.diag(np.abs(b) + np.sqrt(eps))
                Q, R = np.linalg.qr( np.transpose( np.matmul(C,D) ), mode='complete' )
    
                #  Compensate for reweighting.
                b = np.matmul(D,Q)[:,N]
                b = b/np.linalg.norm(b)
    
                #  Multiplying gives a.
                a = np.dot( Z[0:M+1,0:N+1], b )
    
                #  Count leading zeros of b.
                lam = np.argmax( np.abs(b) > tol )
    
                #  Discard leading zeros of b and a.
                b = b[lam:]
                a = a[lam:]
    
                #  Discard trailing zeros of b.
                b = discard_trailing_epsilon(b)
    
            #  Discard trailing zero coefficients in a.
            a = discard_trailing_epsilon(a)
    
            #  Normalize.
            a = a/b[0]
            b = b/b[0]
    
            #  Exact numerator, denominator degrees.
            mu = len(a)
            nu = len(b)

        pade = cls(coeffs, mu, nu, a, b, tol)
        return pade

    @property
    def poles(self):
        """Singularity of the pade function.

        Returns
        -------
        poles : np.1darray
            Poles of the Pade function.
        """
        b = self.b
        b = np.array([*reversed(b)])
        poles = np.roots(b)
        return poles

    @property
    def zeros(self):
        """A zero of a meromorphic function f is a complex number z such that f(z) = 0.

        Returns
        -------
        zeros : np.1darray
            Zeros of the Pade function.
        """
        a = self.a
        a = np.array([*reversed(a)])
        zeros = np.roots(a)
        return zeros

    @property
    def residues(self):
        """Estimate of residues.

        Returns
        -------
        residues : np.1darray
            Residues of the Pade function.
        """
        poles = self.poles()
        # Perturbation for residue estimate
        t = max(self.tol,1e-7)
        residues = []
        for pole in poles:
            residues.append(t*(self(pole+t)-self(pole-t))/2)
        residues = np.array(residues)
        return residues