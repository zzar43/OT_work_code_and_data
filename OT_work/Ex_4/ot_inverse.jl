using Distributed, SharedArrays, JLD2

@everywhere include("code/acoustic_solver_parallel.jl");
@everywhere include("code/acoustic_solver.jl");
@everywhere include("code/adjoint_method_ot.jl")
@everywhere include("code/optimization.jl")

@load "ex4_data.jld2"
println("Data loaded.")

k1 = 0.5
reg = 1e-4
reg_m = 1e2
OTiterMax = 100;

eval_fn_ot(x) = obj_func_ot(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position, reg, reg_m, OTiterMax, k1; pml_len=pml_len, pml_coef=pml_coef)
eval_grad_ot(x) = grad_ot(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position, reg, reg_m, OTiterMax, k1; pml_len=pml_len, pml_coef=pml_coef)


# min_value = minimum(c_true)
# max_value = maximum(c_true)
min_value = 0
max_value = 10
alpha = 2e-5
iterNum = 50
rrho = 0.5
cc = 1e-10
maxSearchTime = 3
# @load "temp_data/ot_new_21-40/data_iter_20.jld2"
# x0 = copy(xk)
x0 = reshape(c, Nx*Ny, 1);

println("Start nonlinear CG.")
xk, fn = nonlinear_cg(eval_fn_ot, eval_grad_ot, x0, alpha, iterNum, min_value, max_value; rho=rrho, c=cc, maxSearchTime=maxSearchTime, threshold=1e-10);

@save "ex4_ot_result.jld2" xk