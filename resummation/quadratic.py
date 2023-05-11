import numpy as np
import mpmath as mpm
import itertools as it
from scipy.linalg import matmul_toeplitz, qr, solve_toeplitz, toeplitz


class QuadraticPade(object):
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
    def __init__(self, coeffs, L, M, N, p, q, r, tol):
        """Initilize the instance.

        Parameters
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
        """
        self.coeffs = coeffs
        self.L = L
        self.M = M
        self.N = N
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
        if not (isinstance(z, (int, float))):
            raise TypeError("Parameter z must be int or float.")
        p,q,r = self.p,self.q,self.r        
        p = np.array([*reversed(p)])
        q = np.array([*reversed(q)])
        r = np.array([*reversed(r)])
        PL = np.polyval(p, z) 
        QM = np.polyval(q, z)
        RN = np.polyval(r, z)
        plus = (-QM + np.sqrt(QM**2 - 4*PL*RN))/(2*RN)
        minus = (-QM - np.sqrt(QM**2 - 4*PL*RN))/(2*RN)
        return plus,minus

    @classmethod
    def build(cls, an, p_deg, q_deg, r_deg, tol=10e-8):
        """Approximant Pade coefficients from Taylor series coefficients.

        Parameters
        ----------
        coeffs : np.ndarray
            Expansion coefficients of the divergent serie.

        L : int
            Order of the numerator.

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
        an = np.asarray(an)
        assert an.ndim == 1
        l_max = r_deg + q_deg + p_deg + 2
        if an.size < l_max:
            raise ValueError("Order of r+q+p (r_deg+q_deg+p_deg) must be smaller than len(an).")
        an = an[:l_max]
        full_amat = toeplitz(an, r=np.zeros_like(an))
        amat2 = (full_amat@full_amat[:, :r_deg+1])
        amat = full_amat[:, :q_deg+1]
        lower = np.concatenate((amat[p_deg+1:, :], amat2[p_deg+1:, :]), axis=-1)
        q_, __ = qr(lower.conj().T, mode='full')
        qrcoeff = q_[:, -1]
        assert qrcoeff.size == r_deg + q_deg + 2
        upper = np.concatenate((amat[:p_deg+1, :], amat2[:p_deg+1, :]), axis=-1)
        pcoeff = -upper@qrcoeff

        p = pcoeff 
        q = qrcoeff[:q_deg+1] 
        r = qrcoeff[q_deg+1:]
        q_pade = cls(an, p_deg, q_deg, r_deg, p, q, r, tol)
        return q_pade


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

