using Distributed, SharedArrays, JLD2

@everywhere include("code/acoustic_solver_parallel.jl");
@everywhere include("code/acoustic_solver.jl");
@everywhere include("code/adjoint_method_ot.jl")
@everywhere include("code/optimization.jl")

@load "ex4_data.jld2"
println("Data loaded.")

eval_fn(x) = obj_func_l2(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
eval_grad(x) = grad_l2(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);


# min_value = minimum(c_true)
# max_value = maximum(c_true)
min_value = 0
max_value = 10
alpha = 1e-8
iterNum = 40
rrho = 0.2
cc = 1e-10
maxSearchTime = 5
# x0 = reshape(c, Nx*Ny, 1);
@load "temp_data/ot_new_41-80/data_iter_40.jld2"
x0 = copy(xk)

println("Start nonlinear CG.")
xk, fn = nonlinear_cg(eval_fn, eval_grad, x0, alpha, iterNum, min_value, max_value; rho=rrho, c=cc, maxSearchTime=maxSearchTime, threshold=1e-10);

@save "ex4_l2_result.jld2" xk