import numpy as np
import mpmath as mpm
import copy

from resummation.pade import Pade

class BorelPade(object):
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
        import scipy.integrate as integrate
        if not (isinstance(z, (int, float))):
            raise TypeError("Parameter z must be int or float.")
        a, b = self.a, self.b
        a = np.array([*reversed(a)])
        b = np.array([*reversed(b)])
        func = lambda t : np.exp(-t)*np.polyval(a, z*t) / np.polyval(b, z*t)
        return integrate.quad(func, 0, np.inf)

    @classmethod
    def build(cls, coeffs, M, N, tol=0):
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

        tmp_coeffs = copy.deepcopy(coeffs)
        for i, coeff in enumerate(coeffs):
            tmp_coeffs[i] = coeff/mpm.factorial(i)
        pade = Pade.plain(tmp_coeffs, M, N, tol)
        a = pade.a
        b = pade.b
        borel_pade = BorelPade(coeffs, M, N, a, b, tol)
        
        return borel_pade