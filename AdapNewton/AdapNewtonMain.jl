include("AdapNewtonSQP.jl")
struct AdapNewtonResult
    XStep::Array
    MuLamStep::Array
    KKTStep::Array
    Count_G_Step::Array
    Count_F_Step::Array
    alpha_Step::Array
    TimeStep::Array
end

## Implement Adaptive Newton for whole problem set
# AdapNewton: parameters of Nonadaptive algorithm
# Prob: problem name set

function AdapNewtonMain(AdapNewton, Prob)
    # Obtain parameters
    Max_Iter = AdapNewton.MaxIter
    EPS_Step = AdapNewton.EPS_Step
    EPS_Res = AdapNewton.EPS_Res
    epsilon = AdapNewton.epsilon
    eta = AdapNewton.eta
    delta = AdapNewton.delta
    TotalRep = AdapNewton.Rep
    beta = AdapNewton.beta
    alpha_max = AdapNewton.alpha_max
    rho = AdapNewton.rho
    kap_grad = AdapNewton.kap_grad
    kap_f = AdapNewton.kap_f
    p_grad = AdapNewton.p_grad
    p_f = AdapNewton.p_f
    CC_grad = AdapNewton.C_grad
    Sigma = AdapNewton.Sigma
    LenC_grad = length(CC_grad)
    LenSigma = length(Sigma)

    # Define results
    AdapN = Array{AdapNewtonResult}(undef,length(Prob))

    # Go oveer all Problems
    for Idprob = 1:length(Prob)
        # load problems
        nlp = CUTEstModel(Prob[Idprob])

        # define result for adaptive Newton #sigma x #replication
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
                    println("AdapNewton","-",Idprob,"-",i,"-",j,"-",rep)

                    X,MuLam,KKT,Count_G,Count_F,Alpha,Time,IdCon = AdapNewtonSQP(nlp,Sigma[j],Max_Iter,EPS_Step,EPS_Res,alpha_max,eta,epsilon,delta,beta,rho,kap_grad,kap_f,p_grad,p_f,CC_grad[i])

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
        AdapN[Idprob] = AdapNewtonResult(XStep,MuLamStep,KKTStep,Count_G_Step,Count_F_Step,alpha_Step,TimeStep)
        finalize(nlp)
    end

    return AdapN
end
