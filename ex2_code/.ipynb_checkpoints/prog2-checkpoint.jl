@everywhere include("code/acoustic_solver_parallel.jl");
@everywhere include("code/acoustic_solver.jl");
@everywhere include("code/adjoint_method_ot.jl")
@everywhere include("code/optimization.jl")
@everywhere include("code/Mixed_Wasserstein.jl")

using JLD2, FileIO