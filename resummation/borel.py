import numpy as np
import mpmath as mpm
import copy

class BorelPade(object):
    """A Borel Pade approximant BP(M,N) of a function or truncated Taylor series.

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
    __init__(self, coeffs, M, N, a, b, tol)
        Initilize the instance.

    __call__(self, z)
        Call the borel pade function.

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
    build(cls, input, M, N, tol=10e-8)
        Build the Borel Pade instance.
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
            Tolerance of the borel pade method.
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
        from scipy.integrate import quad
        if not (isinstance(z, (int, float, complex))):
            raise TypeError("Parameter z must be int float or compelex numbr.")
        a, b = self.a, self.b
        a = [*reversed(a)]
        b = [*reversed(b)]
        func = lambda t : np.exp(-t)*self.pade(z*t)
        val, err = quad(func, 0,np.inf)
        result = complex(float(val),float(err))
        return result

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
        bp : BorelPade
            Instance of BorelPade class.
        """
        from types import LambdaType
        from resummation.pade import Pade

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

        # The Borel transform of original polynoimal is a degree (K − 1) polynomial.
        borel_coeffs = copy.deepcopy(coeffs)
        for k, coeff in enumerate(coeffs):
            borel_coeffs[k] = coeff/mpm.factorial(k)
            
        pade = Pade.build(borel_coeffs, M, N, tol)
        a,b = pade.a, pade.b
        a = np.array([float(i) for i in a])
        b = np.array([float(i) for i in b])
        bp = cls(coeffs, M, N, a, b, tol)
        
        return bp

    @property
    def pade(self):
        """Pade summation in the Borel plane

        Parameters
        ----------
        z : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        TypeError
            _description_
        """
        a, b = self.a, self.b
        a = [*reversed(a)]
        b = [*reversed(b)]
        func = lambda t : mpm.polyval(a, t) / mpm.polyval(b, t)        
        return func

    @property
    def poles(self):
        """Singularity of the borel pade function.

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