#=
Pkg.add("NLPModels")
Pkg.add("JuMP")
Pkg.add("LinearOperators")
Pkg.add("OptimizationProblems")
Pkg.add("MathProgBase")
Pkg.add("ForwardDiff")
Pkg.add("CUTEst")
Pkg.add("NLPModelsJuMP")
Pkg.add("LinearAlgebra")
Pkg.add("Distributed")
Pkg.add("Ipopt")
Pkg.add("DataFrames")
Pkg.add("PyPlot")
Pkg.add("MATLAB")
Pkg.add("ADNLPModels")
Pkg.add("Glob")
Pkg.add("DelimitedFiles")
Pkg.add("Random")
Pkg.add("Distributions")
=#


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
        MaxIter::Int                       # Maximum Iteration
        EPS::Float64                       # minimum of difference
        Rep::Int                           # Number of Independent runs
        alpha_max::Float64                 # maximum of stepsize
        eta::Float64                       # penalty parameter
        nu::Float64                        # nu
        epsilon::Float64                   # epsilon
        delta::Float64                     # delta
        beta::Float64                      # beta
        rho::Float64                       # rho
        kap_grad::Float64                  # kappa of gradient
        kap_f::Float64                     # kappa of f
        p_grad::Float64                    # prob of gradient
        p_f::Float64                       # prob of f
        C_grad::Float64                    # constant of gradient
        Sigma::Array{Float64}              # variance of gradient
    end

    struct AdapGDParams
        verbose                            # Do we create dump dir?
        MaxIter::Int                       # Maximum Iteration
        EPS::Float64                       # minimum of difference
        Rep::Int                           # Number of Independent runs
        alpha_max::Float64                 # maximum of stepsize
        eta::Float64                       # penalty parameter
        nu::Float64                        # nu
        epsilon::Float64                   # epsilon
        delta::Float64                     # delta
        beta::Float64                      # beta
        rho::Float64                       # rho
        kap_grad::Float64                  # kappa of gradient
        kap_f::Float64                     # kappa of f
        p_grad::Float64                    # prob of gradient
        p_f::Float64                       # prob of f
        C_grad::Float64                    # constant of gradient
        Sigma::Array{Float64}              # variance of gradient
    end

    struct NonAdapParams
        verbose                            # Do we create dump dir?
        MaxIter::Int                       # Maximum Iteration
        EPS::Float64                       # minimum of difference
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
        write_matfile("../Solution/NonAdap.mat"; NonAdapR)
    end
end

main()
