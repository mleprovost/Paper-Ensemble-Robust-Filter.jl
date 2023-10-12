module RobustFilter

using Distributions
using DocStringExtensions
using JLD
using LinearAlgebra
using PDMats
using ProgressMeter
using OrdinaryDiffEq
using Roots
using StochasticDiffEq
using SpecialFunctions
using StatsBase
using Statistics
using TransportBasedInference
using SparseArrays
using Graphs
using Lasso
using Random

include("rossler/rossler.jl")
include("loss/loss.jl")

include("tools/benchmark_setup.jl")
include("tools/tools.jl")
include("tools/EM.jl")
include("tools/EMq.jl")
include("tools/EMq_dogru.jl")
include("tools/output_metrics.jl")
include("tools/root_solver.jl")
include("tools/glasso.jl")
include("tools/EM_tlasso.jl")


include("filter/tdist_inflation.jl")
include("filter/lik_enrf.jl")

include("filter/lik_tlasso_enrf.jl")
include("filter/lik_tlasso_enkf.jl")

include("filter/seqassim_lagrangian_lik_enrf.jl")
include("filter/seqassim_lik_enrf.jl")
include("filter/seqassim_lik_tlasso_enrf.jl")
include("filter/seqassim_loclik_enrf.jl")


include("filter/seqlik_enrf.jl")
include("filter/lik_senkf.jl")
include("filter/seqlik_senkf.jl")


include("advection/advection.jl")

# Lorenz 63
include("lorenz63/lorenz63.jl")

# Rossler system



end # module
