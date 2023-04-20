# This file has functions for setting GP hyperparameter priors and fitting a
# GLOM model

const d_err_msg = "d needs to be 0 <= d <= 2"
using Optim
import GPLinearODEMaker; GLOM = GPLinearODEMaker

# hyperparameter priors for kernels with one lenghtscale i.e. pp, se, m52
function kernel_hyper_priors_1λ(hps::AbstractVector{<:Real}, d::Integer, μ::AbstractVector{T}, σ::AbstractVector{T}) where T<:Real
    @assert length(hps) == 1 "There should be 1 hyperparameter and 1 prior distribution μ and σ in calls to kernel_hyper_priors_1λ(). In code: @assert length(hps) == 1"
    priors = [GLOM.log_gamma(hps[1], GLOM.gamma_mode_std_to_α_θ(μ[1], σ[1]); d=d)]
    if d==0; return sum(priors) end
    if d==1; return priors end
    if d==2; return Diagonal(priors) end
    @error d_err_msg
end
function add_kick_1λ!(hps::Vector{<:Real})
    @assert length(hps) == 1
    hps .*= centered_rand(length(hps); center=1, scale=0.5)
    return hps
end

# hyperparameter priors for kernels with two lengthscales i.e. se_se, m52_m52
function kernel_hyper_priors_2λ(hps::Vector{<:Real}, d::Integer, μs::Vector{T}, σs::Vector{T}) where T<:Real
    @assert length(μs) == length(σs) == length(hps) == 3 "There should be 3 hyperparameters and 3 prior distribution μs and σs in calls to kernel_hyper_priors_2λ(). In code: @assert length(μs) == length(σs) == length(hps) == 3"
    priors = [GLOM.log_gamma(hps[1], GLOM.gamma_mode_std_to_α_θ(μs[1], σs[1]); d=d), GLOM.log_gamma(hps[2], GLOM.gamma_mode_std_to_α_θ(μs[2], σs[2]); d=d), GLOM.log_gaussian(hps[3], [μs[3], σs[3]]; d=d)]
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
    hps[1] *= centered_rand(; center=0.8, scale=0.4)
    hps[2] *= centered_rand(; center=1.2, scale=0.4)
    hps[3] *= centered_rand(; center=1.0, scale=0.4)
    return hps
end

function kernel_hyper_priors_qp(hps::Vector{<:Real}, d::Integer, μs::Vector{T}, σs::Vector{T}; ρ::Real=0.9) where {T<:Real}
    @assert length(μs) == length(σs) == length(hps) == 3 "There should be 3 hyperparameters and 3 prior distribution μs and σs in calls to kernel_hyper_priors_qp(). In code: @assert length(μs) == length(σs) == length(hps) == 3"
    paramsλp = GLOM.gamma_mode_std_to_α_θ(μs[3], σs[3])
    Σ_qp_prior = GLOM.bvnormal_covariance(σs[1], σs[2], ρ)  # σP, σse, ρ
    μ_qp_prior = [μs[1], μs[2]]
    if d == 0
        return GLOM.log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.]) + GLOM.log_gamma(hps[3], paramsλp)
    elseif d == 1
        return [GLOM.log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[1,0]), GLOM.log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[0,1]),  GLOM.log_gamma(hps[3], paramsλp; d=d)]
    elseif d == 2
        H = zeros(3, 3)
        H[1,1] = GLOM.log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[2,0])
        H[2,2] = GLOM.log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[0,2])
        H[3,3] = GLOM.log_gamma(hps[3], paramsλp; d=d)
        H[1,2] = GLOM.log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[1,1])
        return Symmetric(H)
    end
    @error d_err_msg
end
function add_kick_qp!(hps::Vector{<:Real})
    @assert length(hps) == 3
    hps[1] *= centered_rand(; center=1.0, scale=0.4)
    hps[2] *= centered_rand(; center=1.2, scale=0.4)
    hps[3] *= centered_rand(; center=0.8, scale=0.4)
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
    glo::GLOM.GLO,
    initial_total_hyperparameters::Vector{<:Real},
    kernel_hyper_priors::Function,
    add_kick!::Function;
    g_tol=1e-6,
    f_tol=1e-10,
    iterations=150,
    print_stuff::Bool=true,
    y_obs::Vector{<:Real}=glo.y_obs)

    optim_cb_local(x::OptimizationState) = optim_cb(x; print_stuff=print_stuff)

    # storing initial hyperparameters
    initial_x = GLOM.remove_zeros(initial_total_hyperparameters)

    f_no_print_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.nlogL_GLOM!(workspace, glo, non_zero_hyper; y_obs=y_obs)
    g!_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.∇nlogL_GLOM!(workspace, glo, non_zero_hyper; y_obs=y_obs)
    h!_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.∇∇nlogL_GLOM!(workspace, glo, non_zero_hyper; y_obs=y_obs)

    function f_no_print(non_zero_hyper::Vector{T}) where {T<:Real}
        nprior = nlogprior_hyperparameters(kernel_hyper_priors, glo.n_kern_hyper, non_zero_hyper, 0)
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
        if nlogprior_hyperparameters(kernel_hyper_priors, glo.n_kern_hyper, non_zero_hyper, 0) == Inf
            G[:] .= 0
        else
            global current_hyper[:] = non_zero_hyper
            G[:] = g!_helper(non_zero_hyper) + nlogprior_hyperparameters(kernel_hyper_priors, glo.n_kern_hyper, non_zero_hyper, 1)
        end
    end

    function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
        H[:, :] = h!_helper(non_zero_hyper) + nlogprior_hyperparameters(kernel_hyper_priors, glo.n_kern_hyper, non_zero_hyper, 2)
    end

    # time0 = Libc.time()
    attempts = 0
    in_saddle = true
    kernel_name = string(glo.kernel)
    global current_hyper = copy(initial_x)
    while attempts < 10 && in_saddle
        attempts += 1
        if attempts > 1;
            println("found saddle point. starting attempt $attempts with a perturbation")
            global current_hyper[1:end-glo.n_kern_hyper] += centered_rand(length(current_hyper) - glo.n_kern_hyper)
            global current_hyper[end-glo.n_kern_hyper+1:end] = add_kick!(current_hyper[end-glo.n_kern_hyper+1:end])
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
                    global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(;callback=optim_cb_local, g_tol=g_tol, f_tol=f_tol, iterations=gridsearch_every)) # 27s
                    before_grid[:] = current_hyper
                    global current_hyper = do_gp_fit_gridsearch!(f_no_print, current_hyper, length(current_hyper) - 2)
                    converged = result.g_converged && isapprox(before_grid, current_hyper)
                    i += 1
                end
            catch
                println("retrying fit")
                i = 0
                while i < Int(ceil(iterations / gridsearch_every)) && !converged
                    global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(;callback=optim_cb_local, g_tol=g_tol, f_tol=f_tol, iterations=gridsearch_every)) # 27s
                    before_grid[:] = current_hyper
                    global current_hyper = do_gp_fit_gridsearch!(f_no_print, current_hyper, length(current_hyper) - 2)
                    converged = result.g_converged && isapprox(before_grid, current_hyper)
                    i += 1
                end
            end
        else
            try
                global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(;callback=optim_cb_local, g_tol=g_tol, f_tol=f_tol, iterations=iterations)) # 27s
            catch
                println("retrying fit")
                global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(;callback=optim_cb_local, g_tol=g_tol, f_tol=f_tol, iterations=iterations))
            end
        end
        current_det = det(h!(zeros(length(initial_x), length(initial_x)), current_hyper))
        if print_stuff; println("Hessian determinant: ", current_det) end
        in_saddle = current_det <= 0
    end
    # return Libc.time() - time0  # returns time used

    # vector of num_kernel_hyperparameters gp hyperparameters followed by the
    # GLOM coefficients and Optim result
    return GLOM.reconstruct_total_hyperparameters(glo, result.minimizer), result
end
fit_GLOM(glo::GLOM.GLO,
    initial_total_hyperparameters::Vector{<:Real},
    kernel_hyper_priors::Function,
    add_kick!::Function;
    kwargs...) = fit_GLOM!(
        GLOM.nlogL_matrix_workspace(glo, initial_total_hyperparameters),
        glo,
        initial_total_hyperparameters,
        kernel_hyper_priors,
        add_kick!;
        kwargs...)


_unriffle_posts(post::AbstractVector, glo::GLOM.GLO) = [post[i:glo.n_out:end] .* glo.normals[i] for i in 1:glo.n_out]
function GLOM_posteriors(
    glo::GLOM.GLO,
    xs_eval::Vector{T1},
    fit_total_hyperparameters::Vector{T2};
    inflate_errors::Int=0,
    inflation_factor::Real=100,
    kwargs...
    ) where {T1<:Real, T2<:Real}

    @assert 0 <= inflate_errors <= glo.n_out
    if inflate_errors!=0
        glo.noise[inflate_errors:glo.n_out:end] .*= inflation_factor
        results = GLOM.GP_posteriors(glo, xs_eval, fit_total_hyperparameters; return_mean_obs=true, return_Σ=false, kwargs...)
        glo.noise[inflate_errors:glo.n_out:end] ./= inflation_factor
    else
        results = GLOM.GP_posteriors(glo, xs_eval, fit_total_hyperparameters; return_mean_obs=true, return_Σ=false, kwargs...)
    end
    return [_unriffle_posts(result, glo) for result in results]
end
function GLOM_posteriors(
    glo::GLOM.GLO,
    fit_total_hyperparameters::Vector{T2};
    inflate_errors::Int=0,
    inflation_factor::Real=100,
    kwargs...
    ) where {T1<:Real, T2<:Real}

    @assert 0 <= inflate_errors <= glo.n_out
    if inflate_errors!=0
        glo.noise[inflate_errors:glo.n_out:end] .*= inflation_factor
        GLOM_at_obs_xs = GLOM.GP_posteriors(glo, fit_total_hyperparameters; kwargs...)
        glo.noise[inflate_errors:glo.n_out:end] ./= inflation_factor
    else
        GLOM_at_obs_xs = GLOM.GP_posteriors(glo, fit_total_hyperparameters; kwargs...)
    end
    return _unriffle_posts(GLOM_at_obs_xs, glo)
end


function valid_as!(possible_as::Vector, a::Matrix; i::Int=1)
    n_out = size(a, 1)
    if i == 10 &&
        # any rows are all 0s
        !any([all(a[j,:] .== 0) for j in 1:n_out]) &&
        # all rows are the same
        !all([a[j,:] == a[j+1,:] for j in 1:n_out-1])
        append!(possible_as, [copy(a)])
    end
    if i < 10
        a[i] = 0
        valid_as!(possible_as, a; i=i+1)
        a[i] = 1
        valid_as!(possible_as, a; i=i+1)
    end
end


function fit_GLOM_and_kep!(
    workspace::GLOM.nlogL_matrix_workspace,
    glo_rv::GLO_RV,
    init_total_hyper::Vector{<:Real},
    kernel_hyper_priors::Function,
    add_kick!::Function,
    current_ks::KeplerSignal,
    offset::Offset;
    print_stuff::Bool=true,
    ignore_increases::Bool=true,
    kwargs...)

    current_hyper = GLOM.remove_zeros(init_total_hyper)
    current_y = copy(glo_rv.GLO.y_obs)
    results = [Inf, Inf]
    result_change = Inf
    num_iter = 0
    while result_change > 1e-4 && num_iter < 30
        current_y[:] = remove_kepler(glo_rv, current_ks, offset)
        fit_total_hyperparameters, result = fit_GLOM!(
            workspace,
            glo_rv.GLO,
            current_hyper,
            kernel_hyper_priors,
            add_kick!;
            print_stuff=false,
            y_obs=current_y)
        current_hyper[:] = GLOM.remove_zeros(fit_total_hyperparameters)
        current_ks = fit_kepler(glo_rv, workspace.Σ_obs, current_ks, offset; print_stuff=false, kwargs...)
        results[:] = [results[2], copy(result.minimum)]
        result_change = results[1] - results[2]
        if ignore_increases && result_change < 0
            @warn "result increased occured on iteration $num_iter"
            result_change = abs(result_change)
        end
        num_iter += 1
        if print_stuff
            println("iteration:     ", num_iter)
            println("current kep:   ", kep_parms_str(current_ks))
            println("current hyper: ", current_hyper)
            println("result change: ", result_change)
        end
    end
    fit_total_hyperparameters = GLOM.reconstruct_total_hyperparameters(glo_rv.GLO, current_hyper)
    return fit_total_hyperparameters, current_ks

end
fit_GLOM_and_kep(glo_rv::GLO_RV,
    init_total_hyper::Vector{<:Real},
    kernel_hyper_priors::Function,
    add_kick!::Function,
    current_ks::KeplerSignal,
    offset::Offset;
    kwargs...) = fit_GLOM_and_kep!(
        GLOM.nlogL_matrix_workspace(glo_rv.GLO, init_total_hyper), glo_rv,
        init_total_hyper, kernel_hyper_priors, add_kick!, current_ks, offset; kwargs...)
