import numpy as np
from general_model import PCNNModel


class ClassicalPCNN(PCNNModel):
    """
    Classical PCNN:
    - given by:
        1) F_ij[n] = f * F_ij[n-1] + Vf * SumOf{M_kl * Y_kl[n-1]} + S_ij
        2) L_ij[n] = h * L_ij[n-1] + Vl * SumOf{W_kl * Y_kl[n-1]}
        3) U_ij[n] = F_ij[n] *(1 + beta * L_ij[n])
        4) Theta_ij[n] = g * Theta_ij[n-1] - Vtheta * Y_ij[n-1]
        5) Y_ij[n] = 1 if U_ij[n] > Theta_ij[n] else 0

    """
    def __init__(self, shape, kernel_type):
        super(ClassicalPCNN, self).__init__(shape)
        '''
        '''
        self.v_theta = 50
        self.Theta = np.ones(shape) * self.v_theta

        self.M = self.choose_kernel(kernel_type)
        self.W = self.choose_kernel(kernel_type)
        self.W = self.choose_kernel(kernel_type)

    def iterate(self, S):
        """
        1) F_ij[n] = f * F_ij[n-1] + Vf * SumOf{M_kl * Y_kl[n-1]} + S_ij
        2) L_ij[n] = h * L_ij[n-1] + Vl * SumOf{W_kl * Y_kl[n-1]}
        3) U_ij[n] = F_ij[n] *(1 + beta * L_ij[n])
        4) Theta_ij[n] = g * Theta_ij[n] - Vtheta * Y_ij[n-1]
        5) Y_ij[n] = 1 if U_ij[n] > Theta_ij[n] else 0
        """
        self.F = self.f * self.F + self.v_f * self.convolve(self.Y, self.M) + S
        self.L = self.h * self.L + self.v_l * self.convolve(self.Y, self.W)
        self.U = self.F * (1 + self.beta*self.L)
        self.Theta = self.g * self.Theta + self.v_theta * self.Y
        self.Y = np.where(self.U > self.Theta, 1, 0)

        return self.Y


