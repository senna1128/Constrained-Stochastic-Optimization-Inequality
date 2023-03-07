include("EstGandH.jl")
include("ComputeAugL.jl")
include("EstAugL.jl")

## Implement adaptive Newton SQP

function AdapNewtonSQP(Id_Design,Id_Sig,Gau_Sigma,Exp_Sigma,C_matrix,q_vector,kap,cerr,cgrad,Max_Iter,EPS_Step,EPS_Res,epsilon,delta,eta,beta,alpha_max,rho,kap_grad,kap_f,chi_grad,chi_f,p_grad,p_f)
    ## Define constraint types
    ncon, nx = size(C_matrix)
    ## Initialization
    EPS, Iter, X, MuLam = 1, 0, ones(nx), ones(ncon)
    ## Constraint
    c_k, G_kk, G_tkk = C_matrix*X+q_vector, C_matrix*C_matrix', C_matrix'*C_matrix
    ## Some intermediate quantities
    G_ktMuLam_k = C_matrix'MuLam
    diagg2lam = (c_k.^2).*MuLam
    DKKT_k = max.(c_k,-MuLam)
    Q_23 = 2*(C_matrix.*c_k.*MuLam)'
    M_k = G_kk+Diagonal(c_k.^2)
    G_ktl = C_matrix'*max.(c_k,zeros(ncon)).^2
    T_nu = sum(max.(c_k,zeros(ncon)).^3)
    nu = kap*T_nu+1
    ## Some other parameters that have to be defined in advance
    IdSing1, IdSing2, PrevSucc, alpha_k, KKT = 0, 0, 0, alpha_max, []
    ## Quantities in While loop used outside need to be defined first
    bnab_augL_k,bnab_augL_k1,J_2c,barR_k = zeros(nx+ncon),zeros(nx+ncon),zeros(ncon),1
    omega_epsnu_xlam, act_set = zeros(ncon), []
    SQPDir, BarSQP = zeros(nx+ncon), zeros(nx+ncon)

    ## Start The Iteration
    while EPS>EPS_Step && barR_k>EPS_Res && Iter<Max_Iter
#        println(Iter,"-KKT-", barR_k)
        ### Obtain the estimate for bnabf_k, bnab2f_k
        bnab_xL_k,bnab_x2L_k,Xi_grad1,Xi_H,barR_k = EstGandH(nx,X,Id_Design,Id_Sig,Gau_Sigma,Exp_Sigma,G_ktMuLam_k,cgrad,DKKT_k,alpha_k,delta,kap_grad,chi_grad,p_grad,PrevSucc)
        push!(KKT,barR_k)
        ### Compute epsilon_k
        a_nux = nu-T_nu
        q_nuxlam = a_nux/(1+norm(MuLam)^2)
        ## compute Q matrix
        # Q_2
        Q_2 = bnab_x2L_k*C_matrix' + Q_23
        J_21 = C_matrix*bnab_xL_k
        J_2 = J_21+diagg2lam
        while epsilon > 1e-9
            bnab_augL_k,bnab_augL_k1,J_2c,omega_epsnu_xlam,act_set = ComputeAugL(epsilon,eta,c_k,MuLam,a_nux,q_nuxlam,diagg2lam,J_21,bnab_xL_k,Q_2,G_ktl,C_matrix,M_k)
            if norm(omega_epsnu_xlam)>cerr*norm(bnab_augL_k) && cerr*norm(bnab_augL_k)<=barR_k
                epsilon /= rho
            else
                # compute SQP direction
                try
                    FullH = hcat(vcat(Diagonal(ones(nx)),C_matrix[act_set,:]), vcat(C_matrix[act_set,:]',zeros(length(act_set),length(act_set))))
                    FullG = [bnab_xL_k-G_ktMuLam_k+C_matrix[act_set,:]'*MuLam[act_set];c_k[act_set]]
                    SQPNewDir = lu(FullH)\-FullG
                    SQPdualDir = lu(M_k)\-(J_2c+Q_2'*(SQPNewDir[1:nx]))
                    SQPDir = [SQPNewDir[1:nx];SQPdualDir]
                catch
                    IdSing1 = 1
                end
                if IdSing1 == 1
                    break
                elseif (bnab_augL_k1'SQPDir)[1]>-eta/2*norm([SQPDir[1:nx];J_2c])^2
                    epsilon /= rho
                else
                    break
                end
            end
        end
        ## Decide the search direction
        if IdSing1==1 || ((bnab_augL_k-bnab_augL_k1)'SQPDir)[1]>eta/4*norm([SQPDir[1:nx];J_2c])^2
            # Compute Hessian
            Hxx = Diagonal(ones(nx))+eta*G_tkk+1/(epsilon*q_nuxlam)*C_matrix[act_set,:]'C_matrix[act_set,:]
            diagTerm = c_k.^2
            diagTerm[act_set] = zeros(length(act_set))
            Term1 = Diagonal(diagTerm)
            Term2 = zeros(ncon,nx)
            Term2[act_set,:] = C_matrix[act_set,:]
            Term3 = G_kk+Term1
            Hmlx = Term2 + eta*Term3*C_matrix
            Term4 = -epsilon*q_nuxlam*ones(ncon)
            Term4[act_set] = zeros(length(act_set))
            Hmlml = Diagonal(Term4) + eta*Term3^2
            Hhat = hcat(vcat(Hxx,Hmlx),vcat(Hmlx', Hmlml))
            try
                eigm = eigmin(Matrix(Hhat))
                if eigm < 0
                    Hhat = Hhat+(0.1-eigm)*Diagonal(ones(nx+ncon))
                end
                BarSQP = lu(Hhat)\-bnab_augL_k
            catch
                IdSing2 = 1
            end
            if IdSing2 == 1
                BarSQP = -bnab_augL_k
            end
        else
            BarSQP = SQPDir
        end
        ### Perform the next step
        XX = X+alpha_k*BarSQP[1:nx]
        cc_k = C_matrix*XX+q_vector
        TT_nu = sum(max.(cc_k,zeros(ncon)).^3)
        if TT_nu > nu/kap
            nu = rho^(ceil(log(kap*TT_nu/nu)/log(rho)))*nu
            if nu > 1e10
                return [],[],[],[],0.1
            end
            Iter += 1
            IdSing1,IdSing2,PrevSucc = 0,0,0
        else
            # Estimate function values
            Quant1 = (bnab_augL_k'BarSQP)[1]
            Quant2 = min((kap_f*alpha_k^2*Quant1)^2,chi_f*delta^2,1)
            Xi_f = max(min(cgrad*log(nx/p_f)/Quant2, 1e4),1)
            if isnan(Xi_f)
                return [],[],[],[],0.2
            end
            Xi_grad2 = min(max(barR_k^2*Xi_f,sqrt(log(nx/p_f)*Xi_f)),Xi_f)
            baugL_k,baugL_sk = EstAugL(nx,ncon,X,MuLam,Id_Design,Id_Sig,Gau_Sigma,Exp_Sigma,eta,epsilon,nu,Xi_f,Xi_grad2,C_matrix,c_k,q_nuxlam,omega_epsnu_xlam,alpha_k,BarSQP,diagg2lam,G_ktMuLam_k,q_vector)
            if baugL_sk <= baugL_k + alpha_k*beta*Quant1
                X = X+alpha_k*BarSQP[1:nx]
                MuLam = MuLam+alpha_k*BarSQP[nx+1:end]
                EPS, Iter = norm(alpha_k*BarSQP), Iter+1
                c_k = C_matrix*X+q_vector
                T_nu = sum(max.(c_k,zeros(ncon)).^3)
                # Some intermediate quantities
                G_ktMuLam_k = C_matrix'MuLam
                diagg2lam = (c_k.^2).*MuLam
                DKKT_k = max.(c_k,-MuLam)
                Q_23 = 2*(C_matrix.*c_k.*MuLam)'
                M_k = G_kk+Diagonal(c_k.^2)
                G_ktl = C_matrix'*max.(c_k,zeros(ncon)).^2
                if -alpha_k*beta*Quant1 >= delta
                    delta *= rho
                    alpha_k = min(alpha_max, rho*alpha_k)
                else
                    delta /= rho
                    alpha_k = min(alpha_max, rho*alpha_k)
                end
                IdSing1,IdSing2,PrevSucc = 0,0,1
            else
                Iter += 1
                alpha_k /= rho
                delta /= rho
                IdSing1,IdSing2,PrevSucc = 0,0,0
            end
        end
    end
    return KKT,1

end
