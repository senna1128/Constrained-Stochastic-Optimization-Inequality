
AdapNewton = Parameter.AdapNewtonParams(true,
                    10000,    # Max_Iter
                    1e-5,     # EPS
                    5,        # Rep
                    1.5,      # alpha_max
                    1,    # eta
                    1,        # nu
                    1,        # epsilon
                    1,        # delta
                    0.3,      # beta
                    2,      # rho
                    1,        # kap_grad
                    0.04,     # kap_f
                    0.9,      # prob of gradient
                    0.9,      # prob of f
                    2,        # constant of gradient
                    [1e-8, 1e-4, 1e-2, 1e-1, 1])   # Sigma


AdapGD = Parameter.AdapGDParams(true,
                    10000,    # Max_Iter
                    1e-5,     # EPS
                    5,        # Rep
                    1.5,      # alpha_max
                    1,    # eta
                    1,        # nu
                    1,        # epsilon
                    1,        # delta
                    0.3,      # beta
                    2,      # rho
                    1,        # kap_grad
                    0.04,     # kap_f
                    0.9,      # prob of gradient
                    0.9,      # prob of f
                    2,        # constant of gradient
                    [1e-8, 1e-4, 1e-2, 1e-1, 1])   # Sigma



NonAdap = Parameter.NonAdapParams(true,
                    10000,
                    1e-5,
                    5,
                    [0.01, 0.1, 0.5, 1],   # const stepsize
                    [0.6, 0.9],            # decay stepsize
                    0.001,
                    [1e-8, 1e-4, 1e-2, 1e-1, 1])   # Sigma
