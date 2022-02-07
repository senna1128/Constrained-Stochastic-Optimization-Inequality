
## Load packages
using NLPModels
using JuMP
using LinearOperators
using OptimizationProblems
using MathProgBase
using ForwardDiff
using CUTEst
using NLPModelsJuMP
using LinearAlgebra
using Distributed
using Ipopt
using DataFrames
using PyPlot
using MATLAB
using ADNLPModels
using Glob
using DelimitedFiles
using Random
using Distributions

cd("/.../NonAdap")
######################################
######  Load problems    #############
######################################
Prob = readdlm(string(pwd(),"/../Parameter/problems.txt"))

# define parameter module
module Parameter
    struct AdapNewtonParams
        verbose                            # Do we create dump dir?
        # stopping parameters
        MaxIter::Int                       # Maximum Iteration
        EPS_Step::Float64                  # minimum of difference
        EPS_Res::Float64                   # minimum of difference
        # adaptive parameters
        epsilon::Float64                   # epsilon
        eta::Float64                       # penalty parameter
        delta::Float64                     # delta
        # fixed parameters
        Rep::Int                           # Number of Independent runs
        beta::Float64                      # beta
        alpha_max::Float64                 # maximum of stepsize
        rho::Float64                       # rho
        kap_grad::Float64                  # kappa of gradient
        kap_f::Float64                     # kappa of f
        p_grad::Float64                    # prob of gradient
        p_f::Float64                       # prob of f
        # test parameters
        C_grad::Array{Float64}             # constant of gradient
        Sigma::Array{Float64}              # variance of gradient
    end

    struct AdapGDParams
        verbose                            # Do we create dump dir?
        # stopping parameters
        MaxIter::Int                       # Maximum Iteration
        EPS_Step::Float64                  # minimum of difference
        EPS_Res::Float64                   # minimum of difference
        # adaptive parameters
        epsilon::Float64                   # epsilon
        eta::Float64                       # penalty parameter
        delta::Float64                     # delta
        # fixed parameters
        Rep::Int                           # Number of Independent runs
        beta::Float64                      # beta
        alpha_max::Float64                 # maximum of stepsize
        rho::Float64                       # rho
        kap_grad::Float64                  # kappa of gradient
        kap_f::Float64                     # kappa of f
        p_grad::Float64                    # prob of gradient
        p_f::Float64                       # prob of f
        # test parameters
        C_grad::Array{Float64}             # constant of gradient
        Sigma::Array{Float64}              # variance of gradient
    end

    struct NonAdapParams
        verbose                            # Do we create dump dir?
        MaxIter::Int                       # Maximum Iteration
        EPS_Step::Float64                  # minimum of difference
        EPS_Res::Float64                   # minimum of difference
        Rep::Int                           # Number of Independent runs
        NoAdapCAlpha::Array{Float64}       # Nonadaptive constant stepsize
        NoAdapDAlpha::Array{Float64}       # Nonadaptive decay stepsize 1/(K^p) with 0.5<p<1
        epsilon::Float64                   # epsilon
        Sigma::Array{Float64}              # variance of gradient
    end

end


using Main.Parameter
include("NonAdapMain.jl")


#######################################
#########  run main file    ###########
#######################################
function main()
    Random.seed!(2021)
    ## include parameter
    include("../Parameter/Param.jl")
    ## run nonadaptive SQP
    NonAdapR = NonAdapMain(NonAdap, Prob)
    ## save result
    if NonAdap.verbose
        NumProb = 10
        decom = convert(Int64, floor(length(NonAdapR)/NumProb))
        for i = 1:decom
            path = string("../Solution/NonAdap", i, ".mat")
            Result = NonAdapR[(i-1)*NumProb+1:i*NumProb]
            write_matfile(path; Result)
        end
        path = string("../Solution/NonAdap", decom+1, ".mat")
        Result = NonAdapR[decom*NumProb+1:end]
        write_matfile(path; Result)
    end
end

main()
