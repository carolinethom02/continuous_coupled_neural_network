import numpy as np
from ccnn_general_model import CCNNModel


class ClassicalCCNN(CCNNModel):
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
        super(ClassicalCCNN, self).__init__(shape)
        '''
        '''
        self.v_theta = 50
        # self.v_theta = np.exp(-self.alpha_f) + 1 + (6 * self.beta * self.v_l)
        self.Theta = np.ones(shape) * self.v_theta

        self.M = self.choose_kernel(kernel_type)
        self.W = self.choose_kernel(kernel_type)

        # # matrix parameters
        # # feeding synaptic weight
        # self.M = np.array([[0.5, 1, 0.5], 
        #           [1, 0, 1],
        #           [0.5, 1, 0.5]])
        # # linking synaptic weight
        # self.W = np.array([[0.5, 1, 0.5], 
        #           [1, 0, 1],
        #           [0.5, 1, 0.5]])
       

    def iterate(self, S):
        """
        1) F_ij[n] = f * F_ij[n-1] + Vf * SumOf{M_kl * Y_kl[n-1]} + S_ij >>>>>>>check that f is the correct term
        2) L_ij[n] = h * L_ij[n-1] + Vl * SumOf{W_kl * Y_kl[n-1]} >>>>>>>>>>check that h is the correct term
        3) U_ij[n] = k * U_ij[n-1] + F_ij[n] *(1 + beta * L_ij[n]) >>>>>>>>change
        4) Theta_ij[n] = g * Theta_ij[n-1] + Vtheta * Y_ij[n-1] >>>>>>>>check the added constant
        5) Y_ij[n] = 1 / (1 + e^-(U_ij[n] - E_ij[n])) >>>>>>right one  
        """
        # self.F = self.f * self.F + self.v_f * self.convolve(self.Y, self.M) + S
        # self.L = self.h * self.L + self.v_l * self.convolve(self.Y, self.W)
        # self.U = self.F * (1 + self.beta*self.L)
        # self.Theta = self.g * self.Theta + self.v_theta * self.Y
        # self.Y = np.where(self.U > self.Theta, 1, 0)

        self.F = (np.exp(-self.alpha_f) * self.F) + (self.v_f * self.convolve(self.Y, self.M)) + S
        self.L = (np.exp(-self.alpha_l) * self.L) + (self.v_l * self.convolve(self.Y, self.W))
        self.U = (np.exp(-self.alpha_f) * self.U) + (self.F * (1 + self.beta * self.L))
        # self.Y = 1 / (1 + (np.exp(-(self.U - self.Theta))))
        self.Theta = (np.exp(-self.alpha_theta) * self.Theta) + (self.v_theta * self.Y)
        self.Y = 1 / (1 + (np.exp(-(self.U - self.Theta))))
        

        return self.Y


