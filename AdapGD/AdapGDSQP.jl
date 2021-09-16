include("EstGandH.jl")
include("ComputeAugL.jl")
include("EstAugL.jl")

## Implement adaptive GD SQP
# Input
### nlp: problem
### sigma: variance of noise
### Max_Iter: max number of iteration
### EPS: minimum of difference
### alpha_max, eta, nu, epsilon, delta, beta, rho
### kap_grad,kap_f,p_grad,p_f,C_grad
# Output
### X: iteration sequence
### Mu: Mu iteration sequence
### Lam: Lam iteration sequence
### KKT: KKT residual iteration sequence
### Time: consuming time
### IdCon: indicator of whether convergence
### IdSing: indicator of singular


function AdapGDSQP(nlp,sigma,Max_Iter,EPS,alpha_max,eta,nu,epsilon,delta,beta,rho,kap_grad,kap_f,p_grad,p_f,C_grad)
    # Define constraint types
    nx, necon, nucon, nlcon = nlp.meta.nvar, length(nlp.meta.jfix), length(nlp.meta.jupp), length(nlp.meta.jlow)
    IdUpp, IdLow = necon + 1, necon + nucon + 1
    nicon, ncon = nucon + nlcon, necon + nucon + nlcon
    # Define constraint bound vector to convert lower bound constraint
    # we order constraints as equality, upper, lower
    BV = zeros(ncon)
    BV[IdUpp:IdLow-1] = nlp.meta.ucon[nlp.meta.jupp]
    BV[IdLow:end] = nlp.meta.lcon[nlp.meta.jlow]
    # Initialization
    eps, k, X = 1, 0, [nlp.meta.x0]
    MuLam = [[nlp.meta.y0[nlp.meta.jfix]; nlp.meta.y0[nlp.meta.jupp]; nlp.meta.y0[nlp.meta.jlow]]]
    # an array to save Hessian of constraints
    un_nab2c_k = Array{Array}(undef, ncon)
    # unordered dual vector used for computing Hessian only
    un_MuLam = zeros(ncon)
    un_MuLam[nlp.meta.jfix] = MuLam[end][1:necon]
    un_MuLam[nlp.meta.jupp] = MuLam[end][IdUpp:IdLow-1]
    un_MuLam[nlp.meta.jlow] = -MuLam[end][IdLow:end]
    # objective and its gradient and Hessian
    f_k, nabf_k = objgrad(nlp, X[end])
    nab2f_k = hess(nlp, X[end])
    # constraint and its Jacobian (#cons by #variable) and Hessian
    un_c_k, un_G_k = consjac(nlp, X[end])
    for i = 1:ncon
        un_nab2c_k[i] = hess(nlp,X[end],obj_weight=1.0,1:ncon.==i)-nab2f_k
    end
    # ordered constraints: equality, upper, lower
    c_k = [un_c_k[nlp.meta.jfix];un_c_k[nlp.meta.jupp]-BV[IdUpp:IdLow-1];-un_c_k[nlp.meta.jlow]+BV[IdLow:end]]
    G_k = [un_G_k[nlp.meta.jfix,:];un_G_k[nlp.meta.jupp,:];-un_G_k[nlp.meta.jlow,:]]
    nab2c_k = [un_nab2c_k[nlp.meta.jfix];un_nab2c_k[nlp.meta.jupp];-un_nab2c_k[nlp.meta.jlow]]
    # Some quantities
    G_ktMuLam_k = G_k'MuLam[end]
    diagg2lam = ((c_k.^2).*MuLam[end])[IdUpp:end]
    DKKT_k = [c_k[1:necon]; max.(c_k[IdUpp:end], -MuLam[end][IdUpp:end])]
    Q_23 = 2*(G_k.*c_k.*MuLam[end])'[:,IdUpp:end]
    M_k = G_k*G_k' + Diagonal([zeros(necon); c_k[IdUpp:end].^2])
    G_ktl = G_k'*[zeros(necon); max.(c_k[IdUpp:end],zeros(nicon)).^2]
    # Lagrangian gradient and its Hessian, and KKT
    nab_xL_k = nabf_k + G_ktMuLam_k
    nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, un_MuLam)
    KKT = [norm([nab_xL_k; DKKT_k])]
    # Covariance matrix
    CovM = sigma*(Diagonal(ones(nx)) + ones(nx, nx))
    # Some other parameters that have to be defined in advance
    IdSing, Xi, alpha_k, CountSam = 0, 1, alpha_max, 0
    # Quantities in While loop used outside need to be defined first
    bnab_augL_k, bnab_augL_k1, J_2c = zeros(nx+ncon), zeros(nx+ncon), zeros(nicon)
    omega_epsnu_xlam, act_set, Equ_Act = zeros(nicon), [], []
    SQPDir = zeros(nx+ncon)

#    NewDir =  zeros(nx+ncon)
#    bnabL_aug_k, Quant  = zeros(nx+ncon), 0

    # start the iteration
    Time = time()
    while min(eps, KKT[end]) > EPS && k < Max_Iter
        ## Obtain the estimate for bnabf_k, bnab2f_k
        bnabf_k, bnab2f_k, Xi = EstGandH(nx,Xi,nabf_k,nab2f_k,G_ktMuLam_k,DKKT_k,sigma,CovM,alpha_k,kap_grad,p_grad,C_grad)
        CountSam += Xi
        ## Compute nu
        T_nu = sum( max.(c_k[IdUpp:end],zeros(nicon)).^3 )
        if T_nu > nu/2
            nu = max(2*T_nu+1, rho*nu)
        end
        ## Compute epsilon_k
        # estimated Lagrangian gradient, Hessian, and intermediate quantities
        bnab_xL_k = bnabf_k + G_ktMuLam_k
        bnab_x2L_k = nab_x2L_k - nab2f_k + bnab2f_k
        a_nux = nu - T_nu
        q_nuxlam = a_nux/(1+norm(MuLam[end][IdUpp:end])^2)
        # compute Q matrix
        # Q_1
        Q_11 = bnab_x2L_k*(G_k'[:,1:necon])
        Q_12 = zeros(nx,necon)
        for i = 1:necon
            Q_12[:,i] = Hermitian(nab2c_k[i],:L)*bnab_xL_k
        end
        Q_1 = Q_11 + Q_12
        # Q_2
        Q_21 = bnab_x2L_k*G_k'[:,IdUpp:end]
        Q_22 = zeros(nx,nicon)
        for i = 1:nicon
            Q_22[:,i] = Hermitian(nab2c_k[i+necon],:L)*bnab_xL_k
        end
        Q_2 = Q_21 + Q_22 + Q_23
        # compute transformed gradient
        J_1 = G_k[1:necon,:]*bnab_xL_k
        J_2 = G_k[IdUpp:end,:]*bnab_xL_k + diagg2lam
        # a while loop to select epsilon
        while epsilon > 1e-7
            bnab_augL_k,bnab_augL_k1,J_2c,omega_epsnu_xlam,act_set = ComputeAugL(necon,IdUpp,epsilon,eta,c_k,MuLam[end],a_nux,q_nuxlam,diagg2lam,G_k,bnab_xL_k,Q_1,J_1,Q_2,G_ktl,M_k)
            if norm([c_k[1:necon];omega_epsnu_xlam]) > norm(bnab_augL_k)
                epsilon /= rho
            else
                # compute SQP GD direction
                try
                    Equ_Act = [1:necon;act_set.+necon]
                    FullH = hcat(vcat(Diagonal(ones(nx)),G_k[Equ_Act,:]), vcat(G_k[Equ_Act,:]',zeros(length(Equ_Act),length(Equ_Act))))
                    FullG = [bnabf_k+G_k[Equ_Act,:]'*MuLam[end][Equ_Act];c_k[Equ_Act]]
                    SQPNewDir = lu(FullH)\-FullG
                    SQPdualDir = lu(M_k)\-([J_1;J_2c] + [Q_1'*(SQPNewDir[1:nx]);Q_2'*(SQPNewDir[1:nx])])
                    SQPDir = [SQPNewDir[1:nx];SQPdualDir]
                catch
                    IdSing = 1
                end
                if IdSing == 1
                    break
                elseif (bnab_augL_k1'SQPDir)[1] > -eta/2*norm([SQPDir[1:nx];J_1;J_2c])^2
                    epsilon /= rho
                else
                    break
                end
            end
        end
        ## Decide the search direction
        if ((bnab_augL_k - bnab_augL_k1)'SQPDir)[1]>eta/4*norm([SQPDir[1:nx];J_1;J_2c])^2 || IdSing == 1
            # Compute Hessian
            BarSQP = -bnab_augL_k
        else
            BarSQP = SQPDir
        end

        # Estimate function values
        Quant1 = (bnab_augL_k'BarSQP)[1]
        Quant2 = min((kap_f*alpha_k^2*Quant1)^2,delta^2,1)
        Xi_f = min(C_grad*log(1/p_f)/Quant2, 100000)
        CountSam += Xi_f
        baugL_k,baugL_sk = EstAugL(nlp,nx,necon,nicon,IdLow,IdUpp,eta,epsilon,nu,sigma,CovM,Xi_f,f_k,c_k,X[end],MuLam[end],nabf_k,G_k,G_ktMuLam_k,q_nuxlam,omega_epsnu_xlam,alpha_k,BarSQP,BV,diagg2lam)

        if baugL_sk <= baugL_k + alpha_k*beta*Quant1
            push!(X, X[end]+alpha_k*BarSQP[1:nx])
            push!(MuLam, MuLam[end]+ alpha_k*BarSQP[nx+1:end])
            eps, k = norm(alpha_k*BarSQP), k+1
            un_MuLam[nlp.meta.jfix] = MuLam[end][1:necon]
            un_MuLam[nlp.meta.jupp] = MuLam[end][IdUpp:IdLow-1]
            un_MuLam[nlp.meta.jlow] = -MuLam[end][IdLow:end]
            # prepare for the next iteration
            f_k, nabf_k = objgrad(nlp, X[end])
            nab2f_k = hess(nlp, X[end])
            un_c_k, un_G_k = consjac(nlp, X[end])
            for i = 1:ncon
                un_nab2c_k[i] = hess(nlp,X[end],obj_weight=1.0,1:ncon.==i)-nab2f_k
            end
            c_k = [un_c_k[nlp.meta.jfix];un_c_k[nlp.meta.jupp]-BV[IdUpp:IdLow-1];-un_c_k[nlp.meta.jlow]+BV[IdLow:end]]
            G_k = [un_G_k[nlp.meta.jfix,:];un_G_k[nlp.meta.jupp,:];-un_G_k[nlp.meta.jlow,:]]
            nab2c_k = [un_nab2c_k[nlp.meta.jfix];un_nab2c_k[nlp.meta.jupp];-un_nab2c_k[nlp.meta.jlow]]
            # Some intermediate quantities
            G_ktMuLam_k = G_k'MuLam[end]
            diagg2lam = ((c_k.^2).*MuLam[end])[IdUpp:end]
            DKKT_k = [c_k[1:necon]; max.(c_k[IdUpp:end], -MuLam[end][IdUpp:end])]
            Q_23 = 2*(G_k.*c_k.*MuLam[end])'[:,IdUpp:end]
            M_k = G_k*G_k' + Diagonal([zeros(necon); c_k[IdUpp:end].^2])
            G_ktl = G_k'*[zeros(necon); max.(c_k[IdUpp:end],zeros(nicon)).^2]
            nab_xL_k = nabf_k + G_ktMuLam_k
            nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, un_MuLam)
            # KKT Vector
            push!(KKT, norm([nab_xL_k; DKKT_k]) )
            if -alpha_k*beta*Quant1 >= delta
                delta *= rho
                alpha_k = min(alpha_max, rho*alpha_k)
            else
                delta /= rho
                alpha_k = min(alpha_max, rho*alpha_k)
            end
        else
            k += 1
            alpha_k /= rho
            delta /= rho
        end
    end
    Time = time() - Time
    if k == Max_Iter
        return [], [], [], 0, Time, 0
    else
        return X, MuLam, KKT, floor(CountSam), Time, 1
    end
end
