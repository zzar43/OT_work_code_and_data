using Distributed, SharedArrays

@everywhere include("code/acoustic_solver_parallel.jl");
@everywhere include("code/acoustic_solver.jl");
@everywhere include("code/adjoint_method_ot.jl")
@everywhere include("code/optimization.jl")

using JLD2
@everywhere using MAT, ImageFiltering

@everywhere begin
    vars = matread("marmousi_model/vp_5.mat")
    c_true = get(vars,"vp_5",3)
    c_true = c_true[1:6:end, 1:6:end]
    c_true = c_true[:,150:450];
#     c_true = vars["vp_5"];
#     c_true = c_true[1:4:500,1000:4:1000+1600];

    c = imfilter(c_true, Kernel.gaussian(15));
    c0 = (c .- minimum(c)) ./ (maximum(c)-minimum(c)) * (maximum(c_true)-minimum(c_true)) .+ minimum(c_true);
    c0[1:16,:] .= 1.5
    c = copy(c0)
    
    c_true = c_true[1:84, :]
    c = c[1:84, :]
    
    Nx, Ny = size(c_true)
    println("Nx: ", Nx, ". Ny: ", Ny, ".")
    rho = ones(Nx,Ny)
    h = 30*1e-3

    # time
    Fs = 400;
    dt = 1/Fs;
    Nt = 2000;
    t = range(0,length=Nt,step=dt);
    println("CFL: ", maximum(c_true) * dt / h);

    # source
    source = source_ricker(5,0.2,t)
    source_num = 11
    source_position = zeros(Int,source_num,2)
    for i = 1:source_num
            source_position[i,:] = [1 1+30(i-1)]
    end
    source = repeat(source, 1, 1);

    # receiver
    receiver_num = 101
    receiver_position = zeros(Int,receiver_num,2)
    for i = 1:receiver_num
        receiver_position[i,:] = [1, (i-1)*3+1]
    end

    # PML
    pml_len = 50
    pml_coef = 50;
    
end

@time uu, received_data = multi_solver(c_true, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);

@save "ex4_data.jld2" received_data c c_true rho Nx Ny h Fs Nt dt t source source_position receiver_position pml_len pml_coef
println("data saved")

# compute adjoint source
@time data = multi_solver_no_wavefield(c, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
M = cost_matrix_1d(t,t)
k1 = 0.5
k2 = 1
k3 = 2
reg = 1e-4
reg_m = 1e2
OTiterMax = 100;
# source ind
ind = 6

adj_l2 = data[:,:,ind] - received_data[:,:,ind];
dd, adj_ot = adj_source_ot_exp(data[:,:,ind], received_data[:,:,ind], M; reg=reg, reg_m=reg_m, iterMax=OTiterMax, k=k1);

@save "ex4_adj.jld2" adj_l2 adj_ot
println("adjoint sources saved")