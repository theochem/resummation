import numpy as np
import mpmath as mpm
import math

class LinearBorel(object):
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
    __init__(self, a, b, tol)
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
    def __init__(self, a, b, tol):
        """Initilize the instance.

        Parameters
        ----------
        a : np.ndarray
            Expansion coefficients of the numerator.

        b : np.ndarray
            Expansion coefficients of the dinominator.
        
        tol: float
            Tolerance of the borel pade method.
        """
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
            Pade function.
        """
        mpm.mp.dps = 15; mpm.mp.pretty = True
        from scipy.integrate import quad
        if not (isinstance(z, (int, float, complex))):
            raise TypeError("Parameter z must be int float or compelex numbr.")
        a, b = self.a, self.b
        a = [*reversed(a)]
        b = [*reversed(b)]
        pade = lambda t : -mpm.polyval(a,t)/mpm.polyval(b,t)
        func = lambda t : mpm.exp(-t)* pade(z*t)
        val = mpm.quad(func, [0, np.inf])
        return val

    @classmethod
    def build(cls, coeffs, M, N, tol=10e-8):
        """Approximant Pade coefficients from Taylor series coefficients.

        Parameters
        ----------
        coeffs : np.ndarray
            Expansion coefficients of the divergent serie.

        M : int
            Order of the numerator.

        N : int
            Order of the denominator.

        Returns
        -------
        bp : BorelPade
            Instance of BorelPade class.
        """
        from resummation.linear_pade import LinearPade
        from types import LambdaType

        if not isinstance(all((M,N)), int):
            raise TypeError("The given order M and N must be integer.")

        if hasattr(coeffs, "__iter__"):
            if len(coeffs) < (M+N+1):
                raise ValueError("Number of coefficients must be higher than order of M+N+1.")
            else:
                coeffs = np.asarray(coeffs)
                coeffs = coeffs[:(M+N+1)]
        elif isinstance(coeffs, LambdaType):
            import mpmath as mpm
            coeffs = mpm.taylor(coeffs,tol,M+N+1)
            coeffs = np.array([float(coeff) for coeff in coeffs])
        else:
            raise TypeError("Input coefficients must be iterator or LambdaType function.")

        # The Borel transform of original polynoimal is a degree (K − 1) polynomial.
        borel_coeffs = []
        for k, coeff in enumerate(coeffs):
            borel_coeffs.append(coeff/math.factorial(k))
            
        pade = LinearPade.build(borel_coeffs, M, N, tol)
        a,b = pade.a, pade.b
        bp = cls(a, b, tol)
        
        return bp

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
