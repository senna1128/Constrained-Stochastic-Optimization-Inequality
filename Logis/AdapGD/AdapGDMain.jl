include("AdapGDSQP.jl")
struct AdapGDResult
    KKTStep::Array
end

## Implement Adaptive GD for whole problem set
# AdapGD: parameters of Nonadaptive algorithm
function AdapGDMain(AdapGD)
    ## Obtain parameters
    # fixed
    Max_Iter,EPS_Step,EPS_Res = AdapGD.MaxIter,AdapGD.EPS_Step,AdapGD.EPS_Res
    epsilon,delta,eta = AdapGD.epsilon,AdapGD.delta,AdapGD.eta
    Rep,beta,alpha_max = AdapGD.Rep,AdapGD.beta,AdapGD.alpha_max
    rho,kap_grad,kap_f = AdapGD.rho,AdapGD.kap_grad,AdapGD.kap_f
    chi_grad,chi_f = AdapGD.chi_grad,AdapGD.chi_f
    p_grad,p_f = AdapGD.p_grad,AdapGD.p_f
    kap,cerr,cgrad = AdapGD.Kappa,AdapGD.chi_err,AdapGD.C_grad
    # Data generation
    x_d, c_r = AdapGD.x_d, AdapGD.c_r
    Gau_Sigma,Exp_Sigma = AdapGD.Gau_Sigma,AdapGD.Exp_Sigma
    LenDesign = [length(Gau_Sigma),length(Exp_Sigma)]
    ## Define results
    AdapG = Array{AdapGDResult}(undef,length(LenDesign))
    ## Generate constraints
    C_matrix, q_vector = rand(Normal(0,1),c_r,x_d), rand(Normal(0,1),c_r)
    ### Go over all designs
    for Id_Design = 1:length(LenDesign)
        ## Construct Result Matrix
        KKTStep = [[] for i = 1:LenDesign[Id_Design]]
        for Id_Sig = 1:LenDesign[Id_Design]
            Id_Rep = 1
            while Id_Rep <= Rep
                println("AdapGD-Des-",Id_Design,"-Sig-",Id_Sig,"-Rep-",Id_Rep)
                KKT,Id_Con = AdapGDSQP(Id_Design,Id_Sig,Gau_Sigma,Exp_Sigma,C_matrix,q_vector,kap,cerr,cgrad,Max_Iter,EPS_Step,EPS_Res,epsilon,delta,eta,beta,alpha_max,rho,kap_grad,kap_f,chi_grad,chi_f,p_grad,p_f)
                if Id_Con != 1
                    Id_Rep += 1
                else
                    push!(KKTStep[Id_Sig], KKT)
                    Id_Rep += 1
                end
            end
        end
        ## Save result
        AdapG[Id_Design] = AdapGDResult(KKTStep)
    end


    return AdapG
end
