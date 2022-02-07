
AdapNewton = Parameter.AdapNewtonParams(true,
                    10000,    # Max_Iter
                    1e-6,     # EPS_Step
                    1e-5,     # EPS_Res
                    1,        # epsilon
                    0.001,    # eta
                    1,        # delta
                    5,        # Rep
                    0.3,      # beta
                    1.5,      # alpha_max
                    2,        # rho
                    1,        # kap_grad
                    0.05,     # kap_f
                    0.1,      # prob of gradient
                    0.1,      # prob of f
                    [1,5,10,50],        # constant of gradient
                    [1e-8, 1e-4, 1e-2, 1e-1, 1])   # Sigma


AdapGD = Parameter.AdapGDParams(true,
                    10000,    # Max_Iter
                    1e-6,     # EPS_Step
                    1e-5,     # EPS_Res
                    1,        # epsilon
                    0.001,    # eta
                    1,        # delta
                    5,        # Rep
                    0.3,      # beta
                    1.5,      # alpha_max
                    2,        # rho
                    1,        # kap_grad
                    0.05,     # kap_f
                    0.1,      # prob of gradient
                    0.1,      # prob of f
                    [1,5,10,50],        # constant of gradient
                    [1e-8, 1e-4, 1e-2, 1e-1, 1])   # Sigma



NonAdap = Parameter.NonAdapParams(true,
                    10000,
                    1e-6,
                    1e-5,
                    5,
                    [0.01, 0.1, 0.5, 1],   # const stepsize
                    [0.6, 0.9],            # decay stepsize
                    0.001,
                    [1e-8, 1e-4, 1e-2, 1e-1, 1])   # Sigma
