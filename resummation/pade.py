import numpy as np
import mpmath as mpm
import itertools as it

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
    def build(cls, input, M, N, tol=10e-8):
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
        from types import LambdaType
        if not (isinstance(M, int) and isinstance(N, int)):
            raise TypeError("Order M and N given must be integer.")
        elif isinstance(input, LambdaType):
            coeffs = mpm.taylor(input,tol,M+N+1)
            coeffs = np.array([float(coeff) for coeff in coeffs])
        elif (isinstance(input, np.ndarray)
            and input.ndim == 1
            and input.dtype == float
        ):
            coeffs = input
        elif coeffs.shape[0] < (M+N+1):
            raise ValueError("Number of coefficients must be higher than order of approximant + 1.")
        else:
            raise TypeError("Input must be LambdaType function or a one-dimensional `numpy` array with `dtype` float.")

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
        poles = self.poles
        # Perturbation for residue estimate
        t = max(self.tol,1e-7)
        residues = []
        for pole in poles:
            residues.append(t*(self(pole+t)-self(pole-t))/2)
        residues = np.array(residues)
        return residues
