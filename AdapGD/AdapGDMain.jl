include("AdapGDSQP.jl")
struct AdapGDResult
    XStep::Array
    MuLamStep::Array
    KKTStep::Array
    Count_G_Step::Array
    Count_F_Step::Array
    alpha_Step::Array
    TimeStep::Array
end

## Implement Adaptive GD for whole problem set
# AdapGD: parameters of Nonadaptive algorithm
# Prob: problem name set

function AdapGDMain(AdapGD, Prob)
    # Obtain parameters
    Max_Iter = AdapGD.MaxIter
    EPS_Step = AdapGD.EPS_Step
    EPS_Res = AdapGD.EPS_Res
    epsilon = AdapGD.epsilon
    eta = AdapGD.eta
    delta = AdapGD.delta
    TotalRep = AdapGD.Rep
    beta = AdapGD.beta
    alpha_max = AdapGD.alpha_max
    rho = AdapGD.rho
    kap_grad = AdapGD.kap_grad
    kap_f = AdapGD.kap_f
    p_grad = AdapGD.p_grad
    p_f = AdapGD.p_f
    CC_grad = AdapGD.C_grad
    Sigma = AdapGD.Sigma
    LenC_grad = length(CC_grad)
    LenSigma = length(Sigma)

    # Define results
    AdapG = Array{AdapGDResult}(undef,length(Prob))

    # Go oveer all Problems
    for Idprob = 1:length(Prob)
        # load problems
        nlp = CUTEstModel(Prob[Idprob])
        # define result for adaptive GD #sigma x #replication
        XStep = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        MuLamStep = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        KKTStep = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        Count_G_Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        Count_F_Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        alpha_Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        TimeStep = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))

        # go over Sigma level and replicate
        i = 1
        while i <= LenC_grad
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("AdapGD","-",Idprob,"-",i,"-",j,"-",rep)

                    X,MuLam,KKT,Count_G,Count_F,Alpha,Time,IdCon = AdapGDSQP(nlp,Sigma[j],Max_Iter,EPS_Step,EPS_Res,alpha_max,eta,epsilon,delta,beta,rho,kap_grad,kap_f,p_grad,p_f,CC_grad[i])

                    if IdCon == 0
                        rep += 1
                    else
                        push!(XStep[i,j], X)
                        push!(MuLamStep[i,j], MuLam)
                        push!(KKTStep[i,j], KKT)
                        push!(Count_G_Step[i,j], Count_G)
                        push!(Count_F_Step[i,j], Count_F)
                        push!(alpha_Step[i,j], Alpha)
                        push!(TimeStep[i,j], Time)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end
        AdapG[Idprob] = AdapGDResult(XStep,MuLamStep,KKTStep,Count_G_Step,Count_F_Step,alpha_Step,TimeStep)
        finalize(nlp)
    end

    return AdapG
end
