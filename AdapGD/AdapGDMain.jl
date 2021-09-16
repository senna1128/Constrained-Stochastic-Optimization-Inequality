include("AdapGDSQP.jl")
struct AdapGDResult
    XStep::Array
    MuLamStep::Array
    KKTStep::Array
    CountStep::Array
    TimeStep::Array
end

## Implement Adaptive GD for whole problem set
# AdapGD: parameters of Nonadaptive algorithm
# Prob: problem name set

function AdapGDMain(AdapGD, Prob)
    # Obtain parameters
    Max_Iter = AdapGD.MaxIter
    EPS = AdapGD.EPS
    TotalRep = AdapGD.Rep
    alpha_max = AdapGD.alpha_max
    eta = AdapGD.eta
    nu = AdapGD.nu
    epsilon = AdapGD.epsilon
    delta = AdapGD.delta
    beta = AdapGD.beta
    rho = AdapGD.rho
    kap_grad = AdapGD.kap_grad
    kap_f = AdapGD.kap_f
    p_grad = AdapGD.p_grad
    p_f = AdapGD.p_f
    C_grad = AdapGD.C_grad
    Sigma = AdapGD.Sigma
    LenSigma = length(Sigma)

    # Define results
    AdapG = Array{AdapGDResult}(undef,length(Prob))

    # Go oveer all Problems
    for Idprob = 1:length(Prob)
        # load problems
        nlp = CUTEstModel(Prob[Idprob])
        # define result for adaptive GD #sigma x #replication
        XStep = [[] for i=1:LenSigma]
        MuLamStep = [[] for i=1:LenSigma]
        KKTStep = [[] for i=1:LenSigma]
        CountStep = [[] for i=1:LenSigma]
        TimeStep = [[] for i=1:LenSigma]
        # go over Sigma level and replicate
        i = 1
        while i <= LenSigma
            rep = 1
            while rep <= TotalRep
                println("AdapGD", Idprob, i, rep)
                X, MuLam, KKT, Count, Time, IdCon = AdapGDSQP(nlp,Sigma[i],Max_Iter,EPS,alpha_max,eta,nu,epsilon,delta,beta,rho,kap_grad,kap_f,p_grad,p_f,C_grad)
                if IdCon == 0
                    rep += 1
                else
                    push!(XStep[i], X)
                    push!(MuLamStep[i], MuLam)
                    push!(KKTStep[i], KKT)
                    push!(CountStep[i], Count)
                    push!(TimeStep[i], Time)
                    rep += 1
                end
            end
            i += 1
        end
        AdapG[Idprob] = AdapGDResult(XStep,MuLamStep,KKTStep,CountStep,TimeStep)
        finalize(nlp)
    end
    return AdapG
end
