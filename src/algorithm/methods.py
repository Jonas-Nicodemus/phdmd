from algorithm.dmd import iodmd, operator_inference
from algorithm.phdmd import phdmd
from utils.system import to_lti, to_phlti


class Method:
    """
    Base class for all methods.
    """
    def __init__(self, name):
        self.name = name

    def __call__(self, X, Y, U, delta_t, H):
        pass


class IODMDMethod(Method):
    def __init__(self):
        super().__init__('DMD')

    def __call__(self, X, Y, U, delta_t, H):
        n = X.shape[0]
        AA_dmd, e = iodmd(X, Y, U)
        lti_dmd = to_lti(AA_dmd, n=n, no_feedtrough=True, sampling_time=delta_t)
        return lti_dmd


class OIMethod(Method):
    def __init__(self):
        super().__init__('OI')

    def __call__(self, X, Y, U, delta_t, H):
        n = X.shape[0]
        AA_dmd, e = operator_inference(X, Y, U, delta_t, H)
        lti_dmd = to_lti(AA_dmd, n=n, E=H, no_feedtrough=True)

        return lti_dmd


class PHDMDMethod(Method):
    def __init__(self):
        super().__init__('pHDMD')

    def __call__(self, X, Y, U, delta_t, H):
        n = X.shape[0]
        J, R, e = phdmd(X, Y, U, delta_t=delta_t, H=H)
        phlti = to_phlti(J, R, n=n, E=H, no_feedtrough=True)

        return phlti


methods = [IODMDMethod(), OIMethod(), PHDMDMethod()]
