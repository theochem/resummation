import numpy as np
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
        mpm.mp.dps = 15; mpm.mp.pretty = True
        if not (isinstance(z, (int, float, complex))):
            raise TypeError("Parameter z must be int float or compelex numbr.")
        p, q, r = self.p, self.q, self.r
        p = [*reversed(p)]
        q = [*reversed(q)]
        r = [*reversed(r)]
        PL = lambda t : mpm.polyval(p,t)
        QM = lambda t : mpm.polyval(q,t)
        RN = lambda t : mpm.polyval(r,t)
        pade_plus = lambda t : (-QM(t) + np.sqrt(QM(t)**2 - 4*PL(t)*RN(t)))/(2*RN(t))
        func_plus = lambda t : mpm.exp(-t)*pade_plus(z*t)
        pade_minus = lambda t : (-QM(t) - np.sqrt(QM(t)**2 - 4*PL(t)*RN(t)))/(2*RN(t))
        func_minus = lambda t : mpm.exp(-t)*pade_minus(z*t)
        val_plus = mpm.quad(func_plus, [0, np.inf])
        val_minus = mpm.quad(func_minus, [0, np.inf])
        return val_plus, val_minus

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



