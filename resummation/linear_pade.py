import numpy as np
import scipy as sp

class LinearPade(object):
    """A linear Pade approximant of order L, M for a truncated Taylor series.

    Attributes
    ----------
    coeffs : np.ndarray
        Expansion coefficients of the divergent serie.

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
    build(cls, M, N, coeffs, tol=0)
        Plain Pade approximant.
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
            Tolerance of the pade method.
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
            Pade functiion.
        """
        if not (isinstance(z, (int, float))):
            raise TypeError("Parameter z must be int or float.")
        a, b = self.a, self.b
        a = np.array([*reversed(a)])
        b = np.array([*reversed(b)])
        func = -np.polyval(a, z) / np.polyval(b, z)
        return func

    @classmethod
    def build(cls, coeffs, L, M, tol=10e-8):
        """Approximant Pade coefficients from Taylor series coefficients.

        Parameters
        ----------
        coeffs : np.ndarray
            Expansion coefficients of the divergent serie.

        L : int
            Order of the numerator.

        M : int
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

        if not isinstance(all((L,M)), int):
            raise TypeError("The given order L and M must be integer.")

        if hasattr(coeffs, "__iter__"):
            if len(coeffs) < (L+M+1):
                raise ValueError("Number of coefficients must be higher than order of L+M+1.")
            else:
                coeffs = np.asarray(coeffs)
                coeffs = coeffs[:(L+M+1)]
        elif isinstance(coeffs, LambdaType):
            import mpmath as mpm
            coeffs = mpm.taylor(coeffs,tol,L+M+1)
            coeffs = np.array([float(coeff) for coeff in coeffs])
        else:
            raise TypeError("Input coefficients must be iterator or LambdaType function.")

        # Build the diagonal-constant matrix
        C = sp.linalg.toeplitz(coeffs, r=np.zeros_like(coeffs))
        C_linear = C[:, :M+1]

        I = np.zeros((L+M+1,L+1))
        I[:L+1,:L+1] = np.eye(L+1)

        matrix = np.concatenate((I, C_linear), axis=-1)

        coeff = np.copy(matrix[:,L+1])
        mat_f = np.delete(matrix, L+1,1)
        coeff = np.array([coeff])
        coeff = -coeff.transpose()
        solution = np.linalg.solve(mat_f, coeff)

        a = solution[0:(L+1)]
        b = solution[L+1:(L+M+1)]
        a = a.transpose()
        a = a[0].tolist()
        b = b.transpose()
        b = b[0].tolist()
        b = [1] + b

        pade = cls(a, b, tol)
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
