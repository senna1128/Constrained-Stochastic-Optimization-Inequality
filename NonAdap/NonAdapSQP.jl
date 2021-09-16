
## Implement non adaptive SQP
# Input
### nlp: problem
### Step: constant or decay
### sigma: variance of noise
### Max_Iter: max number of iteration
### EPS: minimum of difference
# Output
### X: iteration sequence
### Mu: Mu iteration sequence
### Lam: Lam iteration sequence
### KKT: KKT residual iteration sequence
### Time: consuming time
### IdCon: indicator of whether convergence
### IdSing: indicator of singular


function NonAdapSQP(nlp,Step,sigma,Max_Iter,EPS,epsilon,IdConst,rho = 1.5)
    # Define constraint types
    nx, necon, nucon, nlcon = nlp.meta.nvar, length(nlp.meta.jfix), length(nlp.meta.jupp), length(nlp.meta.jlow)
    IdUpp, IdLow = necon + 1, necon + nucon + 1
    nicon, ncon = nucon + nlcon, necon + nucon + nlcon
    SQPDir = zeros(nx+ncon)
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
    DKKT_k = [c_k[1:necon]; max.(c_k[IdUpp:end], -MuLam[end][IdUpp:end])]
    Q_23 = 2*(G_k.*c_k.*MuLam[end])'[:,IdUpp:end]
    M_k = G_k*G_k' + Diagonal([zeros(necon); c_k[IdUpp:end].^2])
    # Lagrangian gradient and its Hessian, and KKT
    nab_xL_k = nabf_k + G_ktMuLam_k
    nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, un_MuLam)
    KKT = [norm([nab_xL_k; DKKT_k])]
    # Covariance matrix
    CovM = sigma*(Diagonal(ones(nx)) + ones(nx, nx))
    # Some other parameters that have to be defined in advance
    IdSing, IdCon = 0, 1
    nu = 2*sum( max.(c_k[IdUpp:end],zeros(nicon)).^3 ) + 1


    # start the iteration
    Time = time()
    while min(eps, KKT[end]) > EPS && k < Max_Iter
        ## Obtain the estimate for bnabf_k, bnab2f_k
        bnabf_k = rand(MvNormal(nabf_k, CovM))
        new_bnabf_k = rand(MvNormal(nabf_k, CovM))
        Delta = rand(Normal(0,sigma^(1/2)), nx, nx)
        bnab2f_k = Hermitian(nab2f_k, :L) + (Delta + Delta')/2
        ## Compute nu
        T_nu = sum( max.(c_k[IdUpp:end],zeros(nicon)).^3 )
        if T_nu > nu/2
            nu = max(2*T_nu+1, rho*nu)
        end
        # estimated Lagrangian gradient, Hessian, and intermediate quantities
        bnab_xL_k = bnabf_k + G_ktMuLam_k
        new_bnab_xL_k = new_bnabf_k + G_ktMuLam_k
        bnab_x2L_k = nab_x2L_k - nab2f_k + bnab2f_k
        # some quantities for computing SQP direction
        a_nux = nu - T_nu
        q_nuxlam = a_nux/(1+norm(MuLam[end][IdUpp:end])^2)
        # active set
        act_set = findall(c_k[IdUpp:end].>=-epsilon*q_nuxlam*MuLam[end][IdUpp:end])
        Equ_Act = [1:necon; act_set.+necon]
        # compute Q matrix
        # Q_1
        Q_11 = bnab_x2L_k*(G_k'[:,1:necon])
        Q_12 = zeros(nx,necon)
        for i = 1:necon
            Q_12[:,i] = Hermitian(nab2c_k[i],:L)*new_bnab_xL_k
        end
        Q_1 = Q_11 + Q_12
        # Q_2
        Q_21 = bnab_x2L_k*(G_k'[:,IdUpp:end])
        Q_22 = zeros(nx,nicon)
        for i = 1:nicon
            Q_22[:,i] = Hermitian(nab2c_k[i+necon],:L)*new_bnab_xL_k
        end
        Q_2 = Q_21 + Q_22 + Q_23
        # compute transformed gradient
        J_1 = G_k[1:necon,:]*bnab_xL_k
        diagg2lam = ((c_k.^2).*MuLam[end])[IdUpp:end]
        diagg2lam[act_set] = zeros(length(act_set))
        J_2c = G_k[IdUpp:end,:]*bnab_xL_k + diagg2lam

        try
            FullH = hcat(vcat(Diagonal(ones(nx)),G_k[Equ_Act,:]), vcat(G_k[Equ_Act,:]',zeros(length(Equ_Act),length(Equ_Act))))
            FullG = [bnabf_k+G_k[Equ_Act,:]'*MuLam[end][Equ_Act];c_k[Equ_Act]]
            SQPDir = lu(FullH)\-FullG
            SQPDir[nx+1:end] = lu(M_k)\-([J_1;J_2c] + [Q_1'*(SQPDir[1:nx]);Q_2'*(SQPDir[1:nx])])
        catch
            IdSing = 1
        end
        if IdSing == 1
            return [], [], [], 0, 0, 1
        else
            Stepsize = IdConst*Step+(1-IdConst)/(k+1)^Step
            # prepare for the next iteration
            push!(X, X[end]+Stepsize*SQPDir[1:nx])
            push!(MuLam, MuLam[end]+Stepsize*SQPDir[nx+1:end])

            eps, k = norm(Stepsize*SQPDir), k+1
            un_MuLam[nlp.meta.jfix] = MuLam[end][1:necon]
            un_MuLam[nlp.meta.jupp] = MuLam[end][IdUpp:IdLow-1]
            un_MuLam[nlp.meta.jlow] = -MuLam[end][IdLow:end]
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
            DKKT_k = [c_k[1:necon]; max.(c_k[IdUpp:end], -MuLam[end][IdUpp:end])]
            Q_23 = 2*(G_k.*c_k.*MuLam[end])'[:,IdUpp:end]
            M_k = G_k*G_k' + Diagonal([zeros(necon); c_k[IdUpp:end].^2])
            nab_xL_k = nabf_k + G_ktMuLam_k
            nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, un_MuLam)
            push!(KKT, norm([nab_xL_k; DKKT_k]) )
        end
    end
    Time = time() - Time
    if k == Max_Iter
        return [], [], [], Time, 0, 0
    else
        return X, MuLam, KKT, Time, 1, 0
    end
end
