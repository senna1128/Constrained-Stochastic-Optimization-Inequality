include("AdapNewtonSQP.jl")
struct AdapNewtonResult
    XStep::Array
    MuLamStep::Array
    KKTStep::Array
    CountStep::Array
    TimeStep::Array
end

## Implement Adaptive Newton for whole problem set
# AdapNewton: parameters of Nonadaptive algorithm
# Prob: problem name set

function AdapNewtonMain(AdapNewton, Prob)
    # Obtain parameters
    Max_Iter = AdapNewton.MaxIter
    EPS = AdapNewton.EPS
    TotalRep = AdapNewton.Rep
    alpha_max = AdapNewton.alpha_max
    eta = AdapNewton.eta
    nu = AdapNewton.nu
    epsilon = AdapNewton.epsilon
    delta = AdapNewton.delta
    beta = AdapNewton.beta
    rho = AdapNewton.rho
    kap_grad = AdapNewton.kap_grad
    kap_f = AdapNewton.kap_f
    p_grad = AdapNewton.p_grad
    p_f = AdapNewton.p_f
    C_grad = AdapNewton.C_grad
    Sigma = AdapNewton.Sigma
    LenSigma = length(Sigma)

    # Define results
    AdapN = Array{AdapNewtonResult}(undef,length(Prob))

    # Go oveer all Problems
    for Idprob = 1:length(Prob)
        # load problems
        nlp = CUTEstModel(Prob[Idprob])
        # define result for adaptive Newton #sigma x #replication
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
                println("AdapNewton", Idprob, i, rep)
                X, MuLam, KKT, Count, Time, IdCon = AdapNewtonSQP(nlp,Sigma[i],Max_Iter,EPS,alpha_max,eta,nu,epsilon,delta,beta,rho,kap_grad,kap_f,p_grad,p_f,C_grad)
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
        AdapN[Idprob] = AdapNewtonResult(XStep,MuLamStep,KKTStep,CountStep,TimeStep)
        finalize(nlp)
    end
    return AdapN
end
