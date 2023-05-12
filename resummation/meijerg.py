import mpmath as mpm
import numpy as np
from scipy.linalg import lstsq

class MeijerG(object):
    """Given a serie, generate the Nth order Meijer-G approximant. Input coefficients, and order.
    """
    def __init__(self, coeffs, order, x, y, p, q):
        """Initialize the instance.

        Parameters
        ----------
        coeffs : np.ndarray(M)
            Expansion coefficients of the divergent serie. M must be at least N + 1.

        order : int
            Order N of the Meijer-G approximant.
        """    
        self.coeffs = coeffs
        self.order = order
        self.x = x
        self.y = y
        self.p = p
        self.q = q

    def __call__(self, g):
        """Call the meijerg function.

        Parameters
        ----------
        g : (int, float)
            Variable or the function.

        Returns
        -------
        approximant : mpc (mpmath complex float)
            Nth order Meijer-G approximant evaluated at g. 
        """
        if not (isinstance(g, (int, float))):
            raise TypeError("Parameter g must be a int or float.")

        coeffs = self.coeffs
        order = self.order
        x = self.x
        y = self.y
        p = self.p
        q = self.q
        l = len(x)

        # Meijer-G approximant: Generate the approximant by using a Laplace transform.
        approximant = 1
        for i in range(l):
            approximant *= mpm.gamma(-y[i])/mpm.gamma(-x[i])
        # transform arrays to list for meijerg function, given m=l+2, n=1, p=l+1, and q=l+2
        # a = [[1], -y_1, ..., -y_l]]
        a = [[1], np.ndarray.tolist(-y)]
        # b = [[1, 1, -x_1, ..., -x_l],[]]
        x = np.insert(-x, 0, [1, 1])
        b = [np.ndarray.tolist(x), []]
        z = -q[l]/(p[l] * g)

        approximant *= mpm.meijerg(a, b, z, series=2)

        # Undo normalization and change of even order to odd order
        if order % 2 == 0:
            approximant *= coeffs[1] * g
            approximant += coeffs[0]
        else:
            approximant *= coeffs[0]

        return approximant

    @classmethod
    def build(cls, coeffs, order, tol=10e-8):
        """Evaluate the MeijerG Nth order approximant of a serie Z(g) \sim \sum_{n=0}^{\infty}.

        Parameters
        ----------
        coeffs : np.ndarray(M)
            Expansion coefficients of the divergent serie. M must be at least N + 1.

        order : int
            Order N of the Meijer-G approximant.

        Returns
        -------
        meijerg : MeijerG 
            Instance of MeijerG class.
        """
        from types import LambdaType

        if not isinstance(order, int):
            raise TypeError("The given order must be integer.")

        if hasattr(coeffs, "__iter__"):
            if len(coeffs) < (order+1):
                raise ValueError("Number of coefficients must be higher than order+1.")
            else:
                coeffs = np.asarray(coeffs)
                coeffs = coeffs[:(order+1)]
        elif isinstance(coeffs, LambdaType):
            coeffs = mpm.taylor(coeffs,tol,order+1)
            coeffs = np.array([float(coeff) for coeff in coeffs])
        else:
            raise TypeError("Input coefficients must be iterator or LambdaType function.")

        # For odd orders
        tmp_coeffs = coeffs.copy()
        dim = order 
        # For even orders, the first term is subtracted to make it odd
        if order % 2 == 0:
            tmp_coeffs = coeffs[1:].copy()
            dim -= 1
        # Normalize the coefficients, by dividing all terms by the first term
        tmp_coeffs /= tmp_coeffs[0]
        l = int((dim - 1)/2)

        # Test that inputs have the correct shape and type
        # Borel transform: Divide coefficients of serie by n!
        # b_n = \frac{z_{n}}{n!}
        r = np.zeros(dim)
        for position in range(dim + 1):
            tmp_coeffs[position] /= mpm.factorial(position)
            # generate array of ratios r_n = b_n+1 / b_n
            if position > 0:
                r[position - 1] = tmp_coeffs[position] / tmp_coeffs[position - 1]
    
        # Hypergeometric ansatz: define a set of N equations from equation the ratios between 
        # consecutive Borel-transformed coefficients with a rational function of n.
        # Solve for p_n and q_n.
        # generate array of (0, 1, ..., 2l)

        exp = np.zeros(1)
        exp = np.append(exp, np.linspace(1, l, l, dtype=float))
        exp = np.append(exp, np.linspace(1, l, l, dtype=float))
        A = np.linspace(0, dim - 1, dim, dtype=float)
        A = A[:, np.newaxis] ** exp
        A[:, (l + 1):] *= -r[:, np.newaxis]

        # solve system of equations A*sol = r
        sol, res, rnk, s = lstsq(A, r, lapack_driver = 'gelss')
        # get p values from the first l + 1 terms of solution
        p = sol[:(l + 1)]
        # assign q_0 = 1
        q = np.ones(1)
        # add the remaining values of q_1 to q_N into q array
        q = np.append(q, sol[(l + 1) :])
    
        # Hypergeometric approximants in the Borel plane: Solve for the roots x, and y to parametrize
        # the hypergeometric approximant.
        # must reverse array because of how numpy roots reads array as polynomial
        x = np.roots(p[::-1])
        y = np.roots(q[::-1])

        meijerg = cls(coeffs, order, x, y, p, q)
        return meijerg

    @property
    def hyper(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        l = len(self.x)
        a = np.insert(-self.x,0,1)
        b = -self.y
        p = self.p[l]
        q = self.q[l]
        func = lambda x: mpm.hyper(a.tolist(), b.tolist(), p/q*x)
        return func
