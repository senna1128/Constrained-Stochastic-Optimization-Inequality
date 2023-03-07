
AdapNewton = Parameter.AdapNewtonParams(true,
                    5000,       # Max_Iter
                    1e-7,      # EPS_Step
                    1e-5,      # EPS_Res
                    1e-1,      # epsilon
                    1,         # delta
                    1e-3,      # eta
                    20,        # Rep
                    0.3,       # beta
                    1.5,       # alpha_max
                    2,         # rho
                    1,         # kap_grad
                    0.05,      # kap_f
                    1,         # chi_grad
                    1,         # chi_f
                    0.1,       # prob of gradient
                    0.1,       # prob of f
                    2,         # constant of Kappa
                    1,         # constant of Chi_err
                    2,         # constant of gradient
                    10,        # dimension of x
                    5,         # dimension of constraint
                    [1e-8,1e-4,1e-2],   # Gaussian variance
                    [10,100,10000])             # Exp parameter


AdapGD = Parameter.AdapGDParams(true,
                    5000,       # Max_Iter
                    1e-7,      # EPS_Step
                    1e-5,      # EPS_Res
                    1e-1,      # epsilon
                    1,         # delta
                    1e-3,      # eta
                    20,        # Rep
                    0.3,       # beta
                    1.5,       # alpha_max
                    2,         # rho
                    1,         # kap_grad
                    0.05,      # kap_f
                    1,         # chi_grad
                    1,         # chi_f
                    0.1,       # prob of gradient
                    0.1,       # prob of f
                    2,         # constant of Kappa
                    1,         # constant of Chi_err
                    2,         # constant of gradient
                    10,        # dimension of x
                    5,         # dimension of constraint
                    [1e-8,1e-4,1e-2],   # Gaussian variance
                    [10,100,10000])             # Exp parameter
