import numpy as np
import cmath
import mpmath as mpm
import copy

class QuadraticBorel(object):
    """A quadratic Borel Pade approximant QB(L, M, N) of a function or truncated Taylor series.

    Attributes
    ----------
    coeffs : np.ndarray
        Expansion coefficients of the divergent serie.

    L : int
        Order of the numerator.

    M : int
        Order of the numerator.

    N : int
        Order of the denominator.

    p : np.ndarray
        Expansion coefficients of the numerator.

    q : np.ndarray
        Expansion coefficients of the dinominator.

    r : np.ndarray
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
    def __init__(self, p, q, r, tol):
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
        self.p = p
        self.q = q
        self.r = r
        self.tol = tol

    def __call__(self, z, func='plus'):
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
        import scipy as sp
        if not (isinstance(z, (int, float, complex))):
            raise TypeError("Parameter z must be int float or compelex numbr.")
        p, q, r = self.p, self.q, self.r
        p = [*reversed(p)]
        q = [*reversed(q)]
        r = [*reversed(r)]
        PL = lambda t : np.polyval(p,t)
        QM = lambda t : np.polyval(q,t)
        RN = lambda t : np.polyval(r,t)
        if func=='plus':
            pade = lambda t : (-QM(t) + np.sqrt(QM(t)**2 - 4*PL(t)*RN(t)))/(2*RN(t))
            func = lambda t : np.exp(-t)*pade(z*t)
            val,error = sp.integrate.quad(func, 0, np.inf, complex_func=True)
        elif func=='minus':
            pade = lambda t : (-QM(t) - np.sqrt(QM(t)**2 - 4*PL(t)*RN(t)))/(2*RN(t))
            func = lambda t : np.exp(-t)*pade(z*t)
            val,error = sp.integrate.quad(func, 0, np.inf, complex_func=True)
        else:
            raise ValueError("Function parameter must be plus or minus.")
        
        return val, error.real

    @classmethod
    def build(cls, input, L, M, N, tol=10e-8):
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
        from resummation.quadratic_pade import QuadraticPade

        if not isinstance(all((L,M,N)),int):
            raise TypeError("The given Order L, M and N must be integer.")
        elif isinstance(input, LambdaType):
            coeffs = mpm.taylor(input,tol,L+M+N+2)
            coeffs = np.array([float(coeff) for coeff in coeffs])
        elif hasattr(input,'__iter__'):
            coeffs = input
            if len(coeffs) < (L+M+N+2):
                raise ValueError("Number of coefficients must be higher than order of L+M+N+2.")
        else:
            raise TypeError("Input must be LambdaType function or a one-dimensional `numpy` array with `dtype` float.")

        # The Borel transform of original polynoimal is a degree (K − 1) polynomial.
        borel_coeffs = copy.deepcopy(coeffs)
        for k, coeff in enumerate(coeffs):
            borel_coeffs[k] = coeff/mpm.factorial(k)
            
        qpade = QuadraticPade.build(borel_coeffs, L, M, N, tol)
        p,q,r = qpade.p, qpade.q, qpade.r
        bqp = cls(p, q, r, tol)
        
        return bqp

    @property
    def zeros(self):
        """Zeros of the pade function.

        Returns
        -------
        poles : np.1darray
            Poles of the Pade function.
        """
        p = self.p
        p = np.array([*reversed(p)])
        zeros = np.roots(p)
        return zeros

    @property
    def poles(self):
        """Singularity of the pade function.

        Returns
        -------
        poles : np.1darray
            Poles of the Pade function.
        """
        r = self.r
        r = np.array([*reversed(r)])
        poles = np.roots(r)
        return poles

    @property
    def branchs(self):
        """Square root branch of the.

        Returns
        -------
        poles : np.1darray
            Poles of the Pade function.
        """
        p = self.p
        q = self.q
        r = self.r
        p = np.array([*reversed(p)])
        q = np.array([*reversed(q)])
        r = np.array([*reversed(r)])
        d = q**2-4*p*r
        branchs = np.roots(d)
        return branchs
