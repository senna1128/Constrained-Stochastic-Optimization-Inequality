
AdapNewton = Parameter.AdapNewtonParams(true,
                    2e3,       # Max_Iter
                    1e-7,      # EPS_Step
                    1e-5,      # EPS_Res
                    1e-1,      # epsilon
                    1,         # delta
                    1e-3,      # eta
                    50,        # Rep
                    0.3,       # beta
                    1.5,       # alpha_max
                    2,         # rho
                    1,         # kap_grad
                    0.05,      # kap_f
                    1,         # chi_grad
                    1,         # chi_f
                    0.1,       # prob of gradient
                    0.1,       # prob of f
                    [2,8,64],        # constant of Kappa
                    [1,10,100],      # constant of Chi_err
                    [2,8,64],        # constant of gradient
                    [1e-8,1e-4,1e-2,1e-1])   # Sigma


AdapGD = Parameter.AdapGDParams(true,
                    2e3,      # Max_Iter
                    1e-7,      # EPS_Step
                    1e-5,      # EPS_Res
                    1e-1,      # epsilon
                    1,         # delta
                    1e-3,      # eta
                    50,        # Rep
                    0.3,       # beta
                    1.5,       # alpha_max
                    2,         # rho
                    1,         # kap_grad
                    0.05,      # kap_f
                    1,         # chi_grad
                    1,         # chi_f
                    0.1,       # prob of gradient
                    0.1,       # prob of f
                    [2,8,64],        # constant of Kappa
                    [1,10,100],      # constant of Chi_err
                    [2,8,64],        # constant of gradient
                    [1e-8,1e-4,1e-2,1e-1])   # Sigma
