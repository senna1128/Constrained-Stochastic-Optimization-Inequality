include("NonAdapSQP.jl")
struct NonAdapResult
    XCStep::Array
    MuLamCStep::Array
    KKTCStep::Array
    TimeCStep::Array
    XDStep::Array
    MuLamDStep::Array
    KKTDStep::Array
    TimeDStep::Array
end

## Implement NonAdaptive SQP for whole problem set
# NonAdap: parameters of Nonadaptive algorithm
# Prob: problem name set

function NonAdapMain(NonAdap, Prob)
    # Obtain parameters
    Max_Iter = NonAdap.MaxIter
    EPS = NonAdap.EPS
    TotalRep = NonAdap.Rep
    epsilon = NonAdap.epsilon
    StepCSet = NonAdap.NoAdapCAlpha
    StepDSet = NonAdap.NoAdapDAlpha
    Sigma = NonAdap.Sigma
    LenCStep = length(StepCSet)
    LenDStep = length(StepDSet)
    LenSigma = length(Sigma)

    # Define results
    NonAdapR = Array{NonAdapResult}(undef,length(Prob))

    # Go over all Problems
    for Idprob = 1:length(Prob)
        # load problems
        nlp = CUTEstModel(Prob[Idprob])

        # define results vector for constant stepsize
        XCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)
        MuLamCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)
        KKTCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)
        TimeCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)

        # go over constant stepsize
        i = 1
        while i <= LenCStep
            j = 1 
            while j <= LenSigma
                rep = 1 
                while rep <= TotalRep
                    println("NonAdapSQP ConstStep", Idprob, i, j, rep)
                    X, MuLam, KKT, Time, IdCon, IdSing = NonAdapSQP(nlp,StepCSet[i],Sigma[j],Max_Iter,EPS,epsilon,1)                    
                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(XCStep[i, j], X)
                        push!(MuLamCStep[i, j], MuLam)
                        push!(KKTCStep[i, j], KKT)
                        push!(TimeCStep[i, j], Time)
                        rep += 1
                    end
                end 
                j += 1 
            end 
            i += 1
        end

        # define results vector for constant stepsize
        XDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)
        MuLamDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)
        KKTDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)
        TimeDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)

        # go over decay stepsize
        i = 1
        while i <= LenDStep
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("NonAdapSQP DecayStep", Idprob, i, j, rep)
                    X, MuLam, KKT, Time, IdCon, IdSing = NonAdapSQP(nlp,StepDSet[i],Sigma[j],Max_Iter,EPS,epsilon,0)                    
                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(XDStep[i, j], X)
                        push!(MuLamDStep[i, j], MuLam)
                        push!(KKTDStep[i, j], KKT)
                        push!(TimeDStep[i, j], Time)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end
        NonAdapR[Idprob] = NonAdapResult(XCStep, MuLamCStep, KKTCStep, TimeCStep, XDStep, MuLamDStep, KKTDStep, TimeDStep)
        finalize(nlp)
    end
    return NonAdapR
end
