
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
        Kappa::Float64                     # radius
        chi_err::Float64                   # coefficient error cond
        C_grad::Float64                    # constant of gradient
        ## Data generating process
        x_d::Int                           # dimension of x
        c_r::Int                           # dimension of constraint
        Gau_Sigma::Array{Float64}          # Gaussian variance
        Exp_Sigma::Array{Float64}          # Exponential distribution parameter
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
        Kappa::Float64                     # radius
        chi_err::Float64                   # coefficient error cond
        C_grad::Float64                    # constant of gradient
        ## Data generating process
        x_d::Int                           # dimension of x
        c_r::Int                           # dimension of constraint
        Gau_Sigma::Array{Float64}          # Gaussian variance
        Exp_Sigma::Array{Float64}          # Exponential distribution parameter
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
    AdapG = AdapGDMain(AdapGD)
    ## save result
    path = string("../Solution/AdapGD.mat")
    Result = AdapG
    write_matfile(path; Result)
end

main()
