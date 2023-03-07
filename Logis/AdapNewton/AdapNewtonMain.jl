include("AdapNewtonSQP.jl")
struct AdapNewtonResult
    KKTStep::Array
end

## Implement Adaptive Newton for whole problem set
# AdapNewton: parameters of Nonadaptive algorithm
function AdapNewtonMain(AdapNewton)
    ## Obtain parameters
    # fixed
    Max_Iter,EPS_Step,EPS_Res = AdapNewton.MaxIter,AdapNewton.EPS_Step,AdapNewton.EPS_Res
    epsilon,delta,eta = AdapNewton.epsilon,AdapNewton.delta,AdapNewton.eta
    Rep,beta,alpha_max = AdapNewton.Rep,AdapNewton.beta,AdapNewton.alpha_max
    rho,kap_grad,kap_f = AdapNewton.rho,AdapNewton.kap_grad,AdapNewton.kap_f
    chi_grad,chi_f = AdapNewton.chi_grad,AdapNewton.chi_f
    p_grad,p_f = AdapNewton.p_grad,AdapNewton.p_f
    kap,cerr,cgrad = AdapNewton.Kappa,AdapNewton.chi_err,AdapNewton.C_grad
    # Data generation
    x_d, c_r = AdapNewton.x_d, AdapNewton.c_r
    Gau_Sigma,Exp_Sigma = AdapNewton.Gau_Sigma,AdapNewton.Exp_Sigma
    LenDesign = [length(Gau_Sigma),length(Exp_Sigma)]
    ## Define results
    AdapN = Array{AdapNewtonResult}(undef,length(LenDesign))
    ## Generate constraints
    C_matrix, q_vector = rand(Normal(0,1),c_r,x_d), rand(Normal(0,1),c_r)
    ### Go over all designs
    for Id_Design = 1:length(LenDesign)
        ## Construct Result Matrix
        KKTStep = [[] for i = 1:LenDesign[Id_Design]]
        for Id_Sig = 1:LenDesign[Id_Design]
            Id_Rep = 1
            while Id_Rep <= Rep
                println("AdapNewton-Des-",Id_Design,"-Sig-",Id_Sig,"-Rep-",Id_Rep)
                KKT,Id_Con = AdapNewtonSQP(Id_Design,Id_Sig,Gau_Sigma,Exp_Sigma,C_matrix,q_vector,kap,cerr,cgrad,Max_Iter,EPS_Step,EPS_Res,epsilon,delta,eta,beta,alpha_max,rho,kap_grad,kap_f,chi_grad,chi_f,p_grad,p_f)
                if Id_Con != 1
                    Id_Rep += 1
                else
                    push!(KKTStep[Id_Sig], KKT)
                    Id_Rep += 1
                end
            end
        end
        ## Save result
        AdapN[Id_Design] = AdapNewtonResult(KKTStep)
    end

    return AdapN
end
