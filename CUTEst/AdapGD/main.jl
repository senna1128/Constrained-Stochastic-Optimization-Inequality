
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
using NLPModelsIpopt

cd("/.../AdapGD")

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
        EPS_Res::Float64                   # residual threshold
        # adaptive parameters
        epsilon::Float64                   # epsilon
        delta::Float64                     # delta
        # fixed parameters
        eta::Float64                       # penalty parameter
        Rep::Int                           # Number of Independent runs
        beta::Float64                      # beta
        alpha_max::Float64                 # maximum of stepsize
        rho::Float64                       # rho
        kap_grad::Float64                  # kappa of gradient
        kap_f::Float64                     # kappa of f
        chi_grad::Float64                  # coefficient grad
        chi_f::Float64                     # coefficient f
        p_grad::Float64                    # prob of gradient
        p_f::Float64                       # prob of f
        # test parameters
        Kappa::Array{Float64}              # radius
        chi_err::Array{Float64}            # coefficient error cond
        C_grad::Array{Float64}             # constant of gradient
        Sigma::Array{Float64}              # variance of gradient
    end

    struct AdapGDParams
        verbose                            # Do we create dump dir?
        # stopping parameters
        MaxIter::Int                       # Maximum Iteration
        EPS_Step::Float64                  # minimum of difference
        EPS_Res::Float64                   # residual threshold
        # adaptive parameters
        epsilon::Float64                   # epsilon
        delta::Float64                     # delta
        # fixed parameters
        eta::Float64                       # penalty parameter
        Rep::Int                           # Number of Independent runs
        beta::Float64                      # beta
        alpha_max::Float64                 # maximum of stepsize
        rho::Float64                       # rho
        kap_grad::Float64                  # kappa of gradient
        kap_f::Float64                     # kappa of f
        chi_grad::Float64                  # coefficient grad
        chi_f::Float64                     # coefficient f
        p_grad::Float64                    # prob of gradient
        p_f::Float64                       # prob of f
        # test parameters
        Kappa::Array{Float64}              # radius
        chi_err::Array{Float64}            # coefficient error cond
        C_grad::Array{Float64}             # constant of gradient
        Sigma::Array{Float64}              # variance of gradient
    end
end
using Main.Parameter

include("AdapGDMain.jl")

#######################################
#########  run main file    ###########
#######################################
function main()
    ## include parameter
    include("../Parameter/Param.jl")
    ## run adaptive SQPGD
    AdapG = AdapGDMain(AdapGD, Prob)
    ## save result
    if AdapGD.verbose
        NumProb = 10
        decom = convert(Int64,floor(length(AdapG)/NumProb))
        for i = 1:decom
            path = string("../Solution/AdapGD",i,".mat")
            Result = AdapG[(i-1)*NumProb+1:i*NumProb]
            write_matfile(path; Result)
        end
        path = string("../Solution/AdapGD", decom+1, ".mat")
        Result = AdapG[decom*NumProb+1:end]
        write_matfile(path; Result)
    end
end

main()

