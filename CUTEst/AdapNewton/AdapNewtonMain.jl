include("AdapNewtonSQP.jl")
struct AdapNewtonResult
    KKTStep::Array
    CountFStep::Array
    CountGStep::Array
    CountHStep::Array
    alphaStep::Array
    TimeStep::Array
    FETriStep::Array
    FEOLDTriStep::Array
    NSRatioStep::Array
    FSRatioStep::Array
end

## Implement Adaptive Newton for whole problem set
# AdapNewton: parameters of Nonadaptive algorithm
# Prob: problem name set
function AdapNewtonMain(AdapNewton, Prob)
    ## Obtain parameters
    # fixed
    Max_Iter,EPS_Step,EPS_Res = AdapNewton.MaxIter,AdapNewton.EPS_Step,AdapNewton.EPS_Res
    epsilon,delta,eta = AdapNewton.epsilon,AdapNewton.delta,AdapNewton.eta
    Rep,beta,alpha_max = AdapNewton.Rep,AdapNewton.beta,AdapNewton.alpha_max
    rho,kap_grad,kap_f = AdapNewton.rho,AdapNewton.kap_grad,AdapNewton.kap_f
    chi_grad,chi_f = AdapNewton.chi_grad,AdapNewton.chi_f
    p_grad,p_f = AdapNewton.p_grad,AdapNewton.p_f
    # tuning
    Kappa,chi_err,C_grad = AdapNewton.Kappa,AdapNewton.chi_err,AdapNewton.C_grad
    Sigma = AdapNewton.Sigma
    LenTest = [length(C_grad),length(Kappa),length(chi_err),1]
    LenSigma = length(Sigma)
    # default tuning
    kap, cerr, cgrad = 2, 1, 2
    ## Define results
    AdapN = Array{AdapNewtonResult}(undef,length(Prob))

    ### Go over all Problems
    for Id_Prob = 1:length(Prob)
        ## Construct Result Matrix
        KKTStep,CountFStep = Array{Array}(undef,length(LenTest)),Array{Array}(undef,length(LenTest))
        CountGStep,CountHStep = Array{Array}(undef,length(LenTest)), Array{Array}(undef,length(LenTest))
        alphaStep,TimeStep = Array{Array}(undef,length(LenTest)), Array{Array}(undef,length(LenTest))
        FETriStep,FEOLDTriStep = Array{Array}(undef,length(LenTest)),Array{Array}(undef,length(LenTest))
        NSRatioStep, FSRatioStep = Array{Array}(undef,length(LenTest)), Array{Array}(undef,length(LenTest))
        for IdRes = 1:length(LenTest)
            KKTStep[IdRes] = reshape([[] for i = 1:LenTest[IdRes]*LenSigma],(LenTest[IdRes],LenSigma))
            CountFStep[IdRes] = reshape([[] for i = 1:LenTest[IdRes]*LenSigma],(LenTest[IdRes],LenSigma))
            CountGStep[IdRes] = reshape([[] for i = 1:LenTest[IdRes]*LenSigma],(LenTest[IdRes],LenSigma))
            CountHStep[IdRes] = reshape([[] for i = 1:LenTest[IdRes]*LenSigma],(LenTest[IdRes],LenSigma))
            alphaStep[IdRes] = reshape([[] for i = 1:LenTest[IdRes]*LenSigma],(LenTest[IdRes],LenSigma))
            TimeStep[IdRes] = reshape([[] for i = 1:LenTest[IdRes]*LenSigma],(LenTest[IdRes],LenSigma))
            FETriStep[IdRes] = reshape([[] for i = 1:LenTest[IdRes]*LenSigma],(LenTest[IdRes],LenSigma))
            FEOLDTriStep[IdRes] = reshape([[] for i = 1:LenTest[IdRes]*LenSigma],(LenTest[IdRes],LenSigma))
            NSRatioStep[IdRes] = reshape([[] for i = 1:LenTest[IdRes]*LenSigma],(LenTest[IdRes],LenSigma))
            FSRatioStep[IdRes] = reshape([[] for i = 1:LenTest[IdRes]*LenSigma],(LenTest[IdRes],LenSigma))
        end
        ## Load problems
        nlp = CUTEstModel(Prob[Id_Prob])
        ## Default setup
        for Id_Sig = 1:LenSigma
            Id_Rep = 1
            while Id_Rep <= Rep
                println("AdapNewton-",Id_Prob,"-Default-Sig-",Id_Sig,"-Rep-",Id_Rep)
                KKT,CountF,CountG,CountH,Alpha,Time,FETri,FEOLDTri,NSRatio,FSRatio,Id_Con = AdapNewtonSQP(nlp,Sigma[Id_Sig],kap,cerr,cgrad,Max_Iter,EPS_Step,EPS_Res,epsilon,delta,eta,beta,alpha_max,rho,kap_grad,kap_f,chi_grad,chi_f,p_grad,p_f,0)
                if Id_Con != 1
                    Id_Rep += 1
                else
                    for IdRes = 1:length(LenTest)-1
                        push!(KKTStep[IdRes][1,Id_Sig], KKT)
                        push!(CountFStep[IdRes][1,Id_Sig], CountF)
                        push!(CountGStep[IdRes][1,Id_Sig], CountG)
                        push!(CountHStep[IdRes][1,Id_Sig], CountH)
                        push!(alphaStep[IdRes][1,Id_Sig], Alpha)
                        push!(TimeStep[IdRes][1,Id_Sig], Time)
                        push!(FETriStep[IdRes][1,Id_Sig], FETri)
                        push!(FEOLDTriStep[IdRes][1,Id_Sig], FEOLDTri)
                        push!(NSRatioStep[IdRes][1,Id_Sig], NSRatio)
                        push!(FSRatioStep[IdRes][1,Id_Sig], FSRatio)
                    end
                    Id_Rep += 1
                end
            end
        end
        ## Test C_grad
        for Id_Cgrad = 2:LenTest[1]
            for Id_Sig = 1:LenSigma
                Id_Rep = 1
                while Id_Rep <= Rep
                    println("AdapNewton-",Id_Prob,"-Cgrad-",Id_Cgrad,"-Sig-",Id_Sig,"-Rep-",Id_Rep)
                    KKT,CountF,CountG,CountH,Alpha,Time,FETri,FEOLDTri,NSRatio,FSRatio,Id_Con = AdapNewtonSQP(nlp,Sigma[Id_Sig],kap,cerr,C_grad[Id_Cgrad],Max_Iter,EPS_Step,EPS_Res,epsilon,delta,eta,beta,alpha_max,rho,kap_grad,kap_f,chi_grad,chi_f,p_grad,p_f,0)
                    if Id_Con != 1
                        Id_Rep += 1
                    else
                        push!(KKTStep[1][Id_Cgrad,Id_Sig], KKT)
                        push!(CountFStep[1][Id_Cgrad,Id_Sig], CountF)
                        push!(CountGStep[1][Id_Cgrad,Id_Sig], CountG)
                        push!(CountHStep[1][Id_Cgrad,Id_Sig], CountH)
                        push!(alphaStep[1][Id_Cgrad,Id_Sig], Alpha)
                        push!(TimeStep[1][Id_Cgrad,Id_Sig], Time)
                        push!(FETriStep[1][Id_Cgrad,Id_Sig], FETri)
                        push!(FEOLDTriStep[1][Id_Cgrad,Id_Sig], FEOLDTri)
                        push!(NSRatioStep[1][Id_Cgrad,Id_Sig], NSRatio)
                        push!(FSRatioStep[1][Id_Cgrad,Id_Sig], FSRatio)
                        Id_Rep += 1
                    end
                end
            end
        end
        ## Test Kappa
        for Id_Kap = 2:LenTest[2]
            for Id_Sig = 1:LenSigma
                Id_Rep = 1
                while Id_Rep <= Rep
                    println("AdapNewton-",Id_Prob,"-Kap-",Id_Kap,"-Sig-",Id_Sig,"-Rep-",Id_Rep)
                    KKT,CountF,CountG,CountH,Alpha,Time,FETri,FEOLDTri,NSRatio,FSRatio,Id_Con = AdapNewtonSQP(nlp,Sigma[Id_Sig],Kappa[Id_Kap],cerr,cgrad,Max_Iter,EPS_Step,EPS_Res,epsilon,delta,eta,beta,alpha_max,rho,kap_grad,kap_f,chi_grad,chi_f,p_grad,p_f,0)
                    if Id_Con != 1
                        Id_Rep += 1
                    else
                        push!(KKTStep[2][Id_Kap,Id_Sig], KKT)
                        push!(CountFStep[2][Id_Kap,Id_Sig], CountF)
                        push!(CountGStep[2][Id_Kap,Id_Sig], CountG)
                        push!(CountHStep[2][Id_Kap,Id_Sig], CountH)
                        push!(alphaStep[2][Id_Kap,Id_Sig], Alpha)
                        push!(TimeStep[2][Id_Kap,Id_Sig], Time)
                        push!(FETriStep[2][Id_Kap,Id_Sig], FETri)
                        push!(FEOLDTriStep[2][Id_Kap,Id_Sig], FEOLDTri)
                        push!(NSRatioStep[2][Id_Kap,Id_Sig], NSRatio)
                        push!(FSRatioStep[2][Id_Kap,Id_Sig], FSRatio)
                        Id_Rep += 1
                    end
                end
            end
        end
        ## Test CE
        for Id_CE = 2:LenTest[3]
            for Id_Sig = 1:LenSigma
                Id_Rep = 1
                while Id_Rep <= Rep
                    println("AdapNewton-",Id_Prob,"-CE-",Id_CE,"-Sig-",Id_Sig,"-Rep-",Id_Rep)
                    KKT,CountF,CountG,CountH,Alpha,Time,FETri,FEOLDTri,NSRatio,FSRatio,Id_Con = AdapNewtonSQP(nlp,Sigma[Id_Sig],kap,chi_err[Id_CE],cgrad,Max_Iter,EPS_Step,EPS_Res,epsilon,delta,eta,beta,alpha_max,rho,kap_grad,kap_f,chi_grad,chi_f,p_grad,p_f,0)
                    if Id_Con != 1
                        Id_Rep += 1
                    else
                        push!(KKTStep[3][Id_CE,Id_Sig], KKT)
                        push!(CountFStep[3][Id_CE,Id_Sig], CountF)
                        push!(CountGStep[3][Id_CE,Id_Sig], CountG)
                        push!(CountHStep[3][Id_CE,Id_Sig], CountH)
                        push!(alphaStep[3][Id_CE,Id_Sig], Alpha)
                        push!(TimeStep[3][Id_CE,Id_Sig], Time)
                        push!(FETriStep[3][Id_CE,Id_Sig], FETri)
                        push!(FEOLDTriStep[3][Id_CE,Id_Sig], FEOLDTri)
                        push!(NSRatioStep[3][Id_CE,Id_Sig], NSRatio)
                        push!(FSRatioStep[3][Id_CE,Id_Sig], FSRatio)
                        Id_Rep += 1
                    end
                end
            end
        end
        ## Test MN
        for Id_Sig = 1:LenSigma
            Id_Rep = 1
            while Id_Rep <= Rep
                println("AdapNewton-",Id_Prob,"-MN-Sig-",Id_Sig,"-Rep-",Id_Rep)
                KKT,CountF,CountG,CountH,Alpha,Time,FETri,FEOLDTri,NSRatio,FSRatio,Id_Con = AdapNewtonSQP(nlp,Sigma[Id_Sig],kap,cerr,cgrad,Max_Iter,EPS_Step,EPS_Res,epsilon,delta,eta,beta,alpha_max,rho,kap_grad,kap_f,chi_grad,chi_f,p_grad,p_f,1)
                if Id_Con != 1
                    Id_Rep += 1
                else
                    push!(KKTStep[end][1,Id_Sig], KKT)
                    push!(CountFStep[end][1,Id_Sig], CountF)
                    push!(CountGStep[end][1,Id_Sig], CountG)
                    push!(CountHStep[end][1,Id_Sig], CountH)
                    push!(alphaStep[end][1,Id_Sig], Alpha)
                    push!(TimeStep[end][1,Id_Sig], Time)
                    push!(FETriStep[end][1,Id_Sig], FETri)
                    push!(FEOLDTriStep[end][1,Id_Sig], FEOLDTri)
                    push!(NSRatioStep[end][1,Id_Sig], NSRatio)
                    push!(FSRatioStep[end][1,Id_Sig], FSRatio)
                    Id_Rep += 1
                end
            end
        end
        ## Save result
        AdapN[Id_Prob] = AdapNewtonResult(KKTStep,CountFStep,CountGStep,CountHStep,alphaStep,TimeStep,FETriStep,FEOLDTriStep,NSRatioStep,FSRatioStep)
        finalize(nlp)
    end

    return AdapN
end
