# This file has functions for setting GP hyperparameter priors and fitting a
# GLOM model

const d_err_msg = "d needs to be 0 <= d <= 2"
using Optim
import GPLinearODEMaker; GLOM = GPLinearODEMaker

# hyperparameter priors for kernels with one lenghtscale i.e. pp, se, m52
function kernel_hyper_priors_1λ(hps::Vector{<:Real}, d::Integer, μ::Real, σ::Real)
    @assert 1 == length(hps)
    priors = [log_gamma(hps[1], gamma_mode_std_2_alpha_theta(μ, σ); d=d)]
    if d==0; return sum(priors) end
    if d==1; return priors end
    if d==2; return Diagonal(priors) end
    @error d_err_msg
end
function add_kick_1λ!(hps::Vector{<:Real})
    @assert length(hps) == 1
    hps .*= centered_rand(rng, length(hps); center=1, scale=0.5)
    return hps
end

# hyperparameter priors for kernels with two lengthscales i.e. se_se, m52_m52
function kernel_hyper_priors_2λ(hps::Vector{<:Real}, d::Integer, μs::Vector{T}, σs::Vector{T}) where T<:Real
    @assert length(μs) == length(σs) == length(hps)
    priors = [log_gamma(hps[1], gamma_mode_std_2_alpha_theta(μs[1], σs[1]); d=d), log_gamma(hps[2], gamma_mode_std_2_alpha_theta(μs[2], σs[2]); d=d), log_gaussian(hps[3], [μs[3], σs[3]]; d=d)]
    if d==0; return sum(priors) end
    if d==1; return priors end
    if d==2; return Diagonal(priors) end
    @error d_err_msg
end
function add_kick_2λ!(hps::Vector{<:Real})
    @assert length(hps) == 3
    if hps[1] > hps[2]
        hps[3] = 1 / hps[3]
        hp[1], hp[2] = hp[2], hp[1]
    end
    hps[1] *= centered_rand(rng; center=0.8, scale=0.4)
    hps[2] *= centered_rand(rng; center=1.2, scale=0.4)
    hps[3] *= centered_rand(rng; center=1.0, scale=0.4)
    return hps
end

function kernel_hyper_priors_qp(hps::Vector{<:Real}, d::Integer, μs::Vector{T}, σs::Vector{T}; ρ::Real=0.9) where {T<:Real}
    @assert length(μs) == length(σs) == length(hps)
    paramsλp = gamma_mode_std_2_alpha_theta(μs[3], σs[3])
    Σ_qp_prior = bvnormal_covariance(σs[1], σs[2], ρ)  # σP, σse, ρ
    μ_qp_prior = [μs[1], μs[2]]
    if d == 0
        return log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.]) + log_gamma(hps[3], paramsλp)
    elseif d == 1
        return [log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[1,0]), log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[0,1]), log_gamma(hps[3], paramsλp; d=d)]
    elseif d == 2
        H = zeros(3, 3)
        H[1,1] = log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[2,0])
        H[2,2] = log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[0,2])
        H[3,3] = log_gamma(hps[3], paramsλp; d=d)
        H[1,2] = log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[1,1])
        return Symmetric(H)
    end
    @error d_err_msg
end
function add_kick_qp!(hps::Vector{<:Real})
    @assert length(hps) == 3
    hps[1] *= centered_rand(rng; center=1.0, scale=0.4)
    hps[2] *= centered_rand(rng; center=1.2, scale=0.4)
    hps[3] *= centered_rand(rng; center=0.8, scale=0.4)
    return hps
end

function nlogprior_hyperparameters(kernel_hyper_priors::Function, n_kern_hyper::Integer, total_hyperparameters::Vector{<:Real}, d::Integer)
    hps = total_hyperparameters[(end - (n_kern_hyper - 1)):end]
    if d == 0
        return -kernel_hyper_priors(hps, d)
    elseif d == 1
        return append!(zeros(length(total_hyperparameters) - n_kern_hyper), -kernel_hyper_priors(hps, d))
    elseif d == 2
        H = zeros(length(total_hyperparameters), length(total_hyperparameters))
        H[(end - (n_kern_hyper - 1)):end, (end - (n_kern_hyper - 1)):end] -= kernel_hyper_priors(hps, d)
        return H
    end
    @error d_err_msg
end


# ends optimization if true
function optim_cb(x::OptimizationState; print_stuff::Bool=true)
    if print_stuff
        println()
        if x.iteration > 0
            println("Iteration:              ", x.iteration)
            println("Time so far:            ", x.metadata["time"], " s")
            println("Unnormalized posterior: ", x.value)
            println("Gradient norm:          ", x.g_norm)
            println()
        end
    end
    return false
end


function do_gp_fit_gridsearch!(f::Function, non_zero_hyper::Vector{<:Real}, ind::Integer)
    println("gridsearching over hp$ind")
    current_hp = non_zero_hyper[ind]
    spread = max(0.5, abs(current_hp) / 4)
    possible_hp = range(current_hp - spread, current_hp + spread, length=11)
    new_hp = current_hp
    new_f = f(non_zero_hyper)
    println("starting at hp = ", new_hp, " -> ", new_f)
    println("searching from ", current_hp - spread, " to ", current_hp + spread)
    for hp in possible_hp
        hold = copy(non_zero_hyper)
        hold[ind] = hp
        possible_f = f(hold)
        if possible_f < new_f
            new_hp = hp
            new_f = possible_f
        end
    end
    non_zero_hyper[ind] = new_hp
    println("ending at hp   = ", new_hp, " -> ", new_f)
    return non_zero_hyper
end


function fit_GLOM!(workspace::GLOM.nlogL_matrix_workspace,
    problem_definition::GLOM.GLO,
    initial_total_hyperparameters::Vector{<:Real},
    kernel_hyper_priors::Function,
    add_kick!::Function;
    g_tol=1e-6,
    iterations=200,
    print_stuff::Bool=true,
    y_obs::Vector{<:Real}=problem_definition.y_obs)

    optim_cb_local(x::OptimizationState) = optim_cb(x; print_stuff=print_stuff)

    # storing initial hyperparameters
    initial_x = GLOM.remove_zeros(initial_total_hyperparameters)

    f_no_print_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.nlogL_GLOM!(workspace, problem_definition, non_zero_hyper; y_obs=y_obs)
    g!_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.∇nlogL_GLOM!(workspace, problem_definition, non_zero_hyper; y_obs=y_obs)
    h!_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.∇∇nlogL_GLOM!(workspace, problem_definition, non_zero_hyper; y_obs=y_obs)

    function f_no_print(non_zero_hyper::Vector{T}) where {T<:Real}
        nprior = nlogprior_hyperparameters(kernel_hyper_priors, problem_definition.n_kern_hyper, non_zero_hyper, 0)
        if nprior == Inf
            return nprior
        else
            return f_no_print_helper(non_zero_hyper) + nprior
        end
    end

    function f(non_zero_hyper::Vector{T}) where {T<:Real}
        if print_stuff; println(non_zero_hyper) end
        global current_hyper[:] = non_zero_hyper
        return f_no_print(non_zero_hyper)
    end

    function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
        if nlogprior_hyperparameters(kernel_hyper_priors, problem_definition.n_kern_hyper, non_zero_hyper, 0) == Inf
            G[:] .= 0
        else
            global current_hyper[:] = non_zero_hyper
            G[:] = g!_helper(non_zero_hyper) + nlogprior_hyperparameters(kernel_hyper_priors, problem_definition.n_kern_hyper, non_zero_hyper, 1)
        end
    end

    function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
        H[:, :] = h!_helper(non_zero_hyper) + nlogprior_hyperparameters(kernel_hyper_priors, problem_definition.n_kern_hyper, non_zero_hyper, 2)
    end

    # time0 = Libc.time()
    attempts = 0
    in_saddle = true
    kernel_name = string(problem_definition.kernel)
    global current_hyper = copy(initial_x)
    while attempts < 10 && in_saddle
        attempts += 1
        if attempts > 1;
            println("found saddle point. starting attempt $attempts with a perturbation")
            global current_hyper[1:end-problem_definition.n_kern_hyper] += centered_rand(rng, length(current_hyper) - problem_definition.n_kern_hyper)
            global current_hyper[end-problem_definition.n_kern_hyper+1:end] = add_kick!(current_hyper[end-problem_definition.n_kern_hyper+1:end])
        end
        if kernel_name == "qp_kernel"
            gridsearch_every = 100
            iterations = Int(ceil(iterations * 1.5))
            converged = false
            i = 0
            before_grid = zeros(length(current_hyper))
            global current_hyper = do_gp_fit_gridsearch!(f_no_print, current_hyper, length(current_hyper) - 2)
            try
                while i < Int(ceil(iterations / gridsearch_every)) && !converged
                    global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb_local, g_tol=g_tol, iterations=gridsearch_every)) # 27s
                    before_grid[:] = current_hyper
                    global current_hyper = do_gp_fit_gridsearch!(f_no_print, current_hyper, length(current_hyper) - 2)
                    converged = result.g_converged && isapprox(before_grid, current_hyper)
                    i += 1
                end
            catch
                println("retrying fit")
                i = 0
                while i < Int(ceil(iterations / gridsearch_every)) && !converged
                    global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb_local, g_tol=g_tol, iterations=gridsearch_every)) # 27s
                    before_grid[:] = current_hyper
                    global current_hyper = do_gp_fit_gridsearch!(f_no_print, current_hyper, length(current_hyper) - 2)
                    converged = result.g_converged && isapprox(before_grid, current_hyper)
                    i += 1
                end
            end
        else
            try
                global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb_local, g_tol=g_tol, iterations=iterations)) # 27s
            catch
                println("retrying fit")
                global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb_local, g_tol=g_tol, iterations=iterations))
            end
        end
        current_det = det(h!(zeros(length(initial_x), length(initial_x)), current_hyper))
        if print_stuff; println(current_det) end
        in_saddle = current_det <= 0
    end
    # return Libc.time() - time0  # returns time used

    # vector of num_kernel_hyperparameters gp hyperparameters followed by the
    # GLOM coefficients and Optim result
    return GLOM.reconstruct_total_hyperparameters(problem_definition, result.minimizer), result
end
fit_GLOM(problem_definition::GLOM.GLO,
    initial_total_hyperparameters::Vector{<:Real},
    kernel_hyper_priors::Function,
    add_kick!::Function;
    g_tol=1e-6,
    iterations=200,
    print_stuff::Bool=true,
    y_obs::Vector{<:Real}=problem_definition.y_obs) = fit_GLOM!(
        GLOM.nlogL_matrix_workspace(problem_definition, initial_total_hyperparameters),
        problem_definition,
        initial_total_hyperparameters,
        kernel_hyper_priors,
        add_kick!;
        g_tol=g_tol,
        iterations=iterations,
        print_stuff=print_stuff,
        y_obs=y_obs)


function GLOM_posteriors(
    problem_definition::GLOM.GLO,
    xs_eval::Vector{T1},
    fit_total_hyperparameters::Vector{T2}
    ) where {T1<:Real, T2<:Real}

    unriffle_posts(post) = [post[i:problem_definition.n_out:end] .* problem_definition.normals[i] for i in 1:problem_definition.n_out]

    results = GLOM.GP_posteriors(problem_definition, xs_eval, fit_total_hyperparameters; return_Σ=true, return_mean_obs=true)
    posts = [unriffle_posts(result) for result in results]
    return posts
end


function valid_a0s!(possible_a0s::Vector, a0::Matrix; i::Int=1)
    n_out = size(a0, 1)
    if i == 10 &&
        # any rows are all 0s
        !any([all(a0[j,:] .== 0) for j in 1:n_out]) &&
        # all rows are the same
        !all([a0[j,:] == a0[j+1,:] for j in 1:n_out-1])
        append!(possible_a0s, [copy(a0)])
    end
    if i < 10
        a0[i] = 0
        valid_a0s!(possible_a0s, a0; i=i+1)
        a0[i] = 1
        valid_a0s!(possible_a0s, a0; i=i+1)
    end
end


function fit_GLOM_and_kep!(
    workspace::GLOM.nlogL_matrix_workspace,
    prob_def_rv::GLO_RV,
    init_total_hyper::Vector{<:Real},
    kernel_hyper_priors::Function,
    add_kick!::Function,
    current_ks::Union{kep_signal, kep_signal_wright};
    avoid_saddle::Bool=true,
    print_stuff::Bool=true)

    current_hyper = GLOM.remove_zeros(init_total_hyper)
    current_y = copy(prob_def_rv.GLO.y_obs)
    results = [Inf, Inf]
    result_change = Inf
    num_iter = 0
    while result_change > 1e-4 && num_iter < 30
        current_ks = fit_kepler(prob_def_rv, workspace.Σ_obs, current_ks; print_stuff=false, avoid_saddle=avoid_saddle)
        current_y[:] = remove_kepler(prob_def_rv, current_ks)
        fit_total_hyperparameters, result = fit_GLOM!(
            workspace,
            prob_def_rv.GLO,
            current_hyper,
            kernel_hyper_priors,
            add_kick!;
            print_stuff=false,
            y_obs=current_y)
        current_hyper[:] = GLOM.remove_zeros(fit_total_hyperparameters)
        results[:] = [results[2], copy(result.minimum)]
        result_dif = results[1] - results[2]
        if result_dif < 0; @warn "result increased occured on iteration $num_iter" end
        result_change = abs(result_dif)
        num_iter += 1
        if print_stuff
            println("iteration:     ", num_iter)
            println("current kep:   ", kep_parms_str(current_ks))
            println("current hyper: ", current_hyper)
            println("result change: ", result_change)
        end
    end
    fit_total_hyperparameters = GLOM.reconstruct_total_hyperparameters(prob_def_rv.GLO, current_hyper)
    return fit_total_hyperparameters, current_ks

end
