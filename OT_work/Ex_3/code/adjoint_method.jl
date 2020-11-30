using Distributed, LinearAlgebra
include("acoustic_solver_parallel.jl")

function obj_func_l2(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=20, pml_coef=200)
    c = reshape(c, Nx, Ny)
    
    data = multi_solver_no_wavefield(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
    
    return 0.5*norm(received_data-data,2)^2
end

function grad_l2(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=20, pml_coef=200)
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    grad = SharedArray{Float64}(Nx, Ny, source_num)
    obj_value = SharedArray{Float64}(source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        u, data_forward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        @views adj_source = data_forward - received_data[:,:,ind]
        @views adj_source = adj_source[end:-1:1,:];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        
        utt = similar(u)
        @views @. utt[:,:,2:end-1] = (u[:,:,3:end]-2u[:,:,2:end-1]+u[:,:,1:end-2]) ./ dt^2;
        @views @. utt = 1 ./ c^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(utt,dims=3)
        grad[:,:,ind] = grad0
        
        # evaluate the objctive function
        obj_value[ind] = 0.5 * norm(data_forward-received_data[:,:,ind],2).^2
    end
    grad = Array(grad)
    grad = sum(grad, dims=3)
    grad = grad[:,:,1]

    obj_value = sum(obj_value)
    grad = reshape(grad, Nx*Ny,1)
    return obj_value, grad
end