include("EstGandH.jl")
include("ComputeAugL.jl")
include("EstAugL.jl")

## Implement adaptive GD SQP
# Input
### nlp: problem
### sigma: variance of noise
### kap: radius
### cerr: coefficient of FEC
### cgrad: coefficient of sample size
### Max_Iter: max number of iteration
### EPS_Step, EPS_Res: minimum of difference
### epsilon,delta,eta,beta,alpha_max,rho,kap_grad,kap_f,chi_grad,chi_f,p_grad,p_f
# Output
### KKT: KKT residual iteration sequence
### CountF, CountG, CountH,Alpha,Time: samples, stepsize, time
### FETri,FEOLDTri: feasibility error
### NSRatio,FSRatio: nonstable, failure steps
### IdCon: indicator of whether convergence

function AdapGDSQP(nlp,sigma,kap,cerr,cgrad,Max_Iter,EPS_Step,EPS_Res,epsilon,delta,eta,beta,alpha_max,rho,kap_grad,kap_f,chi_grad,chi_f,p_grad,p_f,Id_Mul=0)
    ## Define constraint types
    nx,necon,nucon,nlcon = nlp.meta.nvar,length(nlp.meta.jfix),length(nlp.meta.jupp),length(nlp.meta.jlow)
    IdUpp,IdLow = necon+1,necon+nucon+1
    nicon,ncon = nucon+nlcon, necon+nucon+nlcon
    ## Define constraint bound vector, Order constraints as equality, upper, lower
    BV = zeros(ncon)
    BV[IdUpp:IdLow-1] = nlp.meta.ucon[nlp.meta.jupp]
    BV[IdLow:end] = nlp.meta.lcon[nlp.meta.jlow]
    ## Initialization
    EPS, Iter, X = 1, 0, nlp.meta.x0
    MuLam = [nlp.meta.y0[nlp.meta.jfix];nlp.meta.y0[nlp.meta.jupp];nlp.meta.y0[nlp.meta.jlow]]
    # an array to save Hessian of constraints, and unordered dual vector for computing Hessian
    un_nab2c_k, un_MuLam = Array{Array}(undef,ncon), zeros(ncon)
    un_MuLam[nlp.meta.jfix] = MuLam[1:necon]
    un_MuLam[nlp.meta.jupp] = MuLam[IdUpp:IdLow-1]
    un_MuLam[nlp.meta.jlow] = -MuLam[IdLow:end]
    # objective and its gradient and Hessian
    f_k, nabf_k = objgrad(nlp,X)
    nab2f_k = hess(nlp,X)
    # constraint and its Jacobian (#cons by #variable) and Hessian
    un_c_k, un_G_k = consjac(nlp,X)
    for i = 1:ncon
        un_nab2c_k[i] = hess(nlp,X,obj_weight=1.0,1:ncon.==i)-nab2f_k
    end
    # ordered constraints: equality, upper, lower
    c_k = [un_c_k[nlp.meta.jfix];un_c_k[nlp.meta.jupp]-BV[IdUpp:IdLow-1];-un_c_k[nlp.meta.jlow]+BV[IdLow:end]]
    G_k = [un_G_k[nlp.meta.jfix,:];un_G_k[nlp.meta.jupp,:];-un_G_k[nlp.meta.jlow,:]]
    nab2c_k = [un_nab2c_k[nlp.meta.jfix];un_nab2c_k[nlp.meta.jupp];-un_nab2c_k[nlp.meta.jlow]]
    ## Some intermediate quantities
    G_ktMuLam_k = G_k'MuLam
    diagg2lam = ((c_k.^2).*MuLam)[IdUpp:end]
    DKKT_k = [c_k[1:necon];max.(c_k[IdUpp:end],-MuLam[IdUpp:end])]
    Q_23 = 2*(G_k.*c_k.*MuLam)'[:,IdUpp:end]
    M_k = G_k*G_k'+Diagonal([zeros(necon);c_k[IdUpp:end].^2])
    G_ktl = G_k'*[zeros(necon);max.(c_k[IdUpp:end],zeros(nicon)).^2]
    # Lagrangian gradient and its Hessian, and KKT
    nab_xL_k = nabf_k+G_ktMuLam_k
    nab_x2L_k = hess(nlp,X,obj_weight=1.0,un_MuLam)
    KKT = [norm([nab_xL_k;DKKT_k])]
    # Covariance matrix
    sigma_ = min((1-Id_Mul)*sigma+Id_Mul*(1+norm(X))*sigma,1e2)
    CovM = sigma_*(Diagonal(ones(nx))+ones(nx,nx))
    T_nu = sum(max.(c_k[IdUpp:end],zeros(nicon)).^3)
    nu = kap*T_nu+1
    ## Some other parameters that have to be defined in advance
    IdSing1, IdSing2, Id_Tri1, Id_Tri2, alpha_k = 0, 0, 0, 0, alpha_max
    CountF, CountG, CountH, Alpha = [], [], [], []
    CountFE, CountFEOLD, LastNS, CountFS, PrevSucc  = 0, 0, 0, 0, 0
    ## Quantities in While loop used outside need to be defined first
    bnab_augL_k,bnab_augL_k1,J_2c,barR_k = zeros(nx+ncon),zeros(nx+ncon),zeros(nicon),0
    omega_epsnu_xlam, act_set, Equ_Act = zeros(nicon), [], []
    SQPDir, BarSQP = zeros(nx+ncon), zeros(nx+ncon)

    ## Start The Iteration
    Time = time()
    while EPS>EPS_Step && KKT[end]>EPS_Res && Iter<Max_Iter
        ### Obtain the estimate for bnabf_k, bnab2f_k
        bnab_xL_k,bnab_x2L_k,Xi_grad1,Xi_H,barR_k = EstGandH(nx,nab_xL_k,nab_x2L_k,cgrad,DKKT_k,sigma_,CovM,alpha_k,delta,kap_grad,chi_grad,p_grad,PrevSucc)
        push!(CountG,Xi_grad1)
        push!(CountH,Xi_H)
        ### Compute epsilon_k
        a_nux = nu-T_nu
        q_nuxlam = a_nux/(1+norm(MuLam[IdUpp:end])^2)
        ## compute Q matrix
        # Q_1
        Q_11 = bnab_x2L_k*(G_k'[:,1:necon])
        Q_12 = zeros(nx,necon)
        for i = 1:necon
            Q_12[:,i] = Hermitian(nab2c_k[i],:L)*bnab_xL_k
        end
        Q_1 = Q_11 + Q_12
        # Q_2
        Q_21 = bnab_x2L_k*(G_k'[:,IdUpp:end])
        Q_22 = zeros(nx,nicon)
        for i = 1:nicon
            Q_22[:,i] = Hermitian(nab2c_k[i+necon],:L)*bnab_xL_k
        end
        Q_2 = Q_21 + Q_22 + Q_23
        ## compute transformed gradient
        J_1 = G_k[1:necon,:]*bnab_xL_k
        J_21 = G_k[IdUpp:end,:]*bnab_xL_k
        J_2 = J_21+diagg2lam
        while epsilon > 1e-9
            bnab_augL_k,bnab_augL_k1,J_2c,omega_epsnu_xlam,act_set = ComputeAugL(necon,epsilon,eta,c_k,MuLam,a_nux,q_nuxlam,diagg2lam,J_21,bnab_xL_k,Q_1,J_1,Q_2,G_ktl,G_k,M_k)
            if norm([c_k[1:necon];omega_epsnu_xlam])>cerr*norm(bnab_augL_k) && cerr*norm(bnab_augL_k)<=barR_k
                # feasibility condition
                epsilon /= rho
                Id_Tri1 = 1
            else
                # compute SQP direction
                Equ_Act = [1:necon;act_set.+necon]
                try
                    FullH = hcat(vcat(Diagonal(ones(nx)),G_k[Equ_Act,:]), vcat(G_k[Equ_Act,:]',zeros(length(Equ_Act),length(Equ_Act))))
                    FullG = [bnab_xL_k-G_ktMuLam_k+G_k[Equ_Act,:]'*MuLam[Equ_Act];c_k[Equ_Act]]
                    SQPNewDir = lu(FullH)\-FullG
                    SQPdualDir = lu(M_k)\-([J_1;J_2c]+[Q_1'*(SQPNewDir[1:nx]);Q_2'*(SQPNewDir[1:nx])])
                    SQPDir = [SQPNewDir[1:nx];SQPdualDir]
                catch
                    IdSing1 = 1
                end
                if IdSing1 == 1
                    break
                elseif (bnab_augL_k1'SQPDir)[1]>-eta/2*norm([SQPDir[1:nx];J_1;J_2c])^2
                    epsilon /= rho
                    Id_Tri2 = 1
                else
                    break
                end
            end
        end
        if Id_Tri1 == 1
            LastNS, CountFE = Iter, CountFE + 1
            if PrevSucc == 0
                CountFEOLD += 1
            end
        elseif Id_Tri2 == 1
            LastNS = Iter
        end
        ## Decide the search direction
        if IdSing1==1 || ((bnab_augL_k-bnab_augL_k1)'SQPDir)[1]>eta/4*norm([SQPDir[1:nx];J_1;J_2c])^2
            if IdSing1 == 0
                CountFS += 1
            end
            # Compute GD
            BarSQP = -bnab_augL_k
        else
            BarSQP = SQPDir
        end
        ### Perform the next step
        XX = X+alpha_k*BarSQP[1:nx]
        uun_c_k= cons(nlp,XX)
        cc_k = [uun_c_k[nlp.meta.jfix];uun_c_k[nlp.meta.jupp]-BV[IdUpp:IdLow-1];-uun_c_k[nlp.meta.jlow]+BV[IdLow:end]]
        TT_nu = sum(max.(cc_k[IdUpp:end],zeros(nicon)).^3)
        if TT_nu > nu/kap
            nu = rho^(ceil(log(kap*TT_nu/nu)/log(rho)))*nu
            if nu > 1e10
                return [],[],[],[],[],[],[],[],[],[],0.1
            end
            Iter += 1
            IdSing1,IdSing2,Id_Tri1,Id_Tri2,PrevSucc = 0,0,0,0,0
        else
            # Estimate function values
            Quant1 = (bnab_augL_k'BarSQP)[1]
            Quant2 = min((kap_f*alpha_k^2*Quant1)^2,chi_f*delta^2,1)
            Xi_f = max(min(cgrad*log(nx/p_f)/Quant2, 1e8*sigma_),1)
            if isnan(Xi_f)
                return [],[],[],[],[],[],[],[],[],[],0.2
            end
            Xi_grad2 = min(max(barR_k^2*Xi_f,sqrt(log(nx/p_f)*Xi_f)),Xi_f)
            push!(CountF,2*Xi_f)
            CountG[end] += 2*Xi_grad2
            baugL_k,baugL_sk = EstAugL(nlp,nx,necon,nicon,IdLow,IdUpp,cgrad,eta,epsilon,nu,sigma_,CovM,Xi_f,Xi_grad2,f_k,nab_xL_k,G_k,c_k,X,MuLam,q_nuxlam,omega_epsnu_xlam,alpha_k,BarSQP,BV,diagg2lam)
            if baugL_sk <= baugL_k + alpha_k*beta*Quant1
                X = X+alpha_k*BarSQP[1:nx]
                MuLam = MuLam+alpha_k*BarSQP[nx+1:end]
                push!(Alpha, alpha_k)
                EPS, Iter = norm(alpha_k*BarSQP), Iter+1
                un_MuLam[nlp.meta.jfix] = MuLam[1:necon]
                un_MuLam[nlp.meta.jupp] = MuLam[IdUpp:IdLow-1]
                un_MuLam[nlp.meta.jlow] = -MuLam[IdLow:end]
                # prepare for the next iteration
                f_k, nabf_k = objgrad(nlp,X)
                nab2f_k = hess(nlp,X)
                un_c_k,un_G_k = consjac(nlp,X)
                for i = 1:ncon
                    un_nab2c_k[i] = hess(nlp,X,obj_weight=1.0,1:ncon.==i)-nab2f_k
                end
                c_k = [un_c_k[nlp.meta.jfix];un_c_k[nlp.meta.jupp]-BV[IdUpp:IdLow-1];-un_c_k[nlp.meta.jlow]+BV[IdLow:end]]
                G_k = [un_G_k[nlp.meta.jfix,:];un_G_k[nlp.meta.jupp,:];-un_G_k[nlp.meta.jlow,:]]
                nab2c_k = [un_nab2c_k[nlp.meta.jfix];un_nab2c_k[nlp.meta.jupp];-un_nab2c_k[nlp.meta.jlow]]
                T_nu = sum(max.(c_k[IdUpp:end],zeros(nicon)).^3)
                # Covariance matrix
                sigma_ = min((1-Id_Mul)*sigma+Id_Mul*(1+norm(X))*sigma,1e2)
                CovM = sigma_*(Diagonal(ones(nx))+ones(nx,nx))
                # Some intermediate quantities
                G_ktMuLam_k = G_k'MuLam
                diagg2lam = ((c_k.^2).*MuLam[end])[IdUpp:end]
                DKKT_k = [c_k[1:necon];max.(c_k[IdUpp:end],-MuLam[IdUpp:end])]
                Q_23 = 2*(G_k.*c_k.*MuLam)'[:,IdUpp:end]
                M_k = G_k*G_k'+Diagonal([zeros(necon);c_k[IdUpp:end].^2])
                G_ktl = G_k'*[zeros(necon); max.(c_k[IdUpp:end],zeros(nicon)).^2]
                nab_xL_k = nabf_k+G_ktMuLam_k
                nab_x2L_k = hess(nlp,X,obj_weight=1.0,un_MuLam)
                # KKT Vector
                push!(KKT,norm([nab_xL_k;DKKT_k]))
                if -alpha_k*beta*Quant1 >= delta
                    delta *= rho
                    alpha_k = min(alpha_max, rho*alpha_k)
                else
                    delta /= rho
                    alpha_k = min(alpha_max, rho*alpha_k)
                end
                IdSing1,IdSing2,Id_Tri1,Id_Tri2,PrevSucc = 0,0,0,0,1
            else
                Iter += 1
                alpha_k /= rho
                delta /= rho
                IdSing1,IdSing2,Id_Tri1,Id_Tri2,PrevSucc = 0,0,0,0,0
            end
        end
    end
    Time = time() - Time
    if Iter == Max_Iter
        return [],[],[],[],[],[],[],[],[],[],0.3
    else
        FETri,FEOLDTri,NSRatio,FSRatio = CountFE/Iter,CountFEOLD/Iter,LastNS/Iter,CountFS/Iter
        return KKT,CountF,CountG,CountH,Alpha,Time,FETri,FEOLDTri,NSRatio,FSRatio,1
    end

end
