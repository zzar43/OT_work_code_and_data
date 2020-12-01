# This is the algorithm of L1/Wasserstein mixed distance
# created by: 2020/11/30

using LinearAlgebra
include("Sinkhorn.jl")

# compute the gradient and value of the mixed Wasserstein distance in 1d case
# J(a,b) = W_2^2(\hat a, \hat b) + \lambda \|\sum a_i - \sum b_i\|^2

function Mixed_Wasserstein(a, b, eps, lambda_mix, M; threshold=1e-5, iter_num=iteration_number, verbose=false);
    a = a[:]
    b = b[:]

    sum_a = sum(a)
    Na = length(a)
    a_hat = a ./ (sum(a))
    b_hat = b ./ (sum(b))
    
    T1, grad1, dist1 = sinkhorn_1d_TP(a_hat, b_hat, eps, M; threshold=threshold, iter_num=iter_num, verbose=verbose);
    
    grad = zeros(Na)
    temp = zeros(Na)
    # compute the Jacobian
    for j = 1:Na
        temp = - a .* grad1 ./ (sum(a)^2)
        temp[j] += grad1[j] / (sum(a))
        grad[j] = sum(temp)
    end
    
    grad += 2 * lambda_mix * a;
    dist = dist1 + lambda_mix * (sum_a - sum(b))^2
    return grad, dist
end

function Mixed_Wasserstein_exp(a, b, eps, lambda_mix, M, k_normalize; threshold=1e-5, iter_num=iteration_number, verbose=false);
    # normalization
    a_exp = exp.(k_normalize.*a)
    b_exp = exp.(k_normalize.*b)
    
    grad, dist = Mixed_Wasserstein(a_exp, b_exp, eps, lambda_mix, M; iter_num=iter_num);
    return grad, dist
end