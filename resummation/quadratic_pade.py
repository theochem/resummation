import numpy as np
import scipy as sp

class QuadraticPade(object):
    """A quadratic Pade approximant for a truncated Taylor series.

    Attributes
    ----------
    p : np.ndarray
        Expansion coefficients of the polynomials P(z).

    q : np.ndarray
        Expansion coefficients of the polynomials Q(z).

    r : np.ndarray
        Expansion coefficients of the polynomials R(z).

    Methods
    -------
    __init__(self, p, q, r)
        Initilize the instance.

    __call__(self, z)
        Call the quadratic pade function.

    ClassMethods
    ------------
    build(cls, coeffs, L, M, N, tol)
        Approximant quadratic Pade coefficients from Taylor series coefficients.
    """
    def __init__(self, p, q, r):
        """Initilize the instance.

        Parameters
        ----------
        p : np.ndarray
            Expansion coefficients of polynomials P(z).

        q : np.ndarray
            Expansion coefficients of polynomials Q(z).

        r : np.ndarray
            Expansion coefficients of polynomials R(z).
        """
        self.p = p
        self.q = q
        self.r = r

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
        plus : float
            Quadratic Pade approximants Pade F+.

        minus : float
            Quadratic Pade approximants Pade F-.
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
        return plus, minus

    @classmethod
    def build(cls, coeffs, L, M, N, tol=1e-8):
        """Approximant quadratic Pade coefficients from Taylor series coefficients.

        Parameters
        ----------
        coeffs : iterator
            Expansion coefficients of the serie.

        L : int
            Order of the polynomials P(z).

        M : int
            Order of the polynomials Q(z).

        N : int
            Order of the polynomials R(z).

        tol: float
            Tolerance of the pade method.

        Returns
        -------
        q_pade : QuadraticPade
            Instance of QuadraticPade class.
        """
        from types import LambdaType

        if not isinstance(all((L,M,N)), int):
            raise TypeError("The given order L, M and N must be integer.")

        if hasattr(coeffs, "__iter__"):
            if len(coeffs) < (L+M+N+2):
                raise ValueError("Number of coefficients must be higher than order of L+M+N+2.")
            else:
                coeffs = np.asarray(coeffs)
                coeffs = coeffs[:(L+M+N+2)]
        elif isinstance(coeffs, LambdaType):
            import mpmath as mpm
            coeffs = mpm.taylor(coeffs,tol,L+M+N+2)
            coeffs = np.array([float(coeff) for coeff in coeffs])
        else:
            raise TypeError("Input coefficients must be iterator or LambdaType function.")

        # Build the diagonal-constant matrix
        C = sp.linalg.toeplitz(coeffs, r=np.zeros_like(coeffs))
        C_square = np.dot(C, C[:, :N+1])
        C_linear = C[:, :M+1]

        # lower-right subblock
        lower_right = np.concatenate((C_linear[L+1:, :], C_square[L+1:, :]), axis=-1)
        # upper-left subblock
        upper_left = np.concatenate((C_linear[:L+1, :], C_square[:L+1, :]), axis=-1)

        eig_values, eig_vectors = sp.linalg.qr(lower_right.conj().T, mode='full')
        qr = eig_values[:, -1]
        p = -np.dot(upper_left,qr)
        q = qr[:M+1] 
        r = qr[M+1:]

        q_pade = cls(p, q, r)
        return q_pade


