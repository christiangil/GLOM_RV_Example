# these are all custom diagnostic functions. May help with debugging
using Statistics
using LinearAlgebra

"""
	est_d(f, input; dif=1e-7)


Estimate the derivative/gradient of `f` at `inputs` with forward differences

# Keyword Arguments
- `dif::Real=1e-7`: The forward differencing step size
"""
est_d(f::Function, input; dif::Real=1e-7) = (f(input + dif) - f(input)) ./ dif

"""
	est_∇nlogL_kep

Estimate the gradient of `nlogL_kep(data, times, covariance, ks)` with forward
differences

# Keyword Arguments
- `data_unit::Unitful.Velocity=1u"m/s"`: The units of the input data vector
- `include_priors::Bool=true`: Whether to add a prior evaluation to the output
- `use_hk::Bool=true`: Whether to use h and k or e and ω parameterization
- `kwargs...`: keyword arguments that are passed to GLOM.est_∇()
"""
function est_∇nlogL_kep(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    ks::kep_signal,
    offset::Offset;
    data_unit::Unitful.Velocity=1u"m/s",
    include_priors::Bool=true,
    use_hk::Bool=true,
    kwargs...
    ) where T<:Real

    K_u = unit(ks.K)
    P_u = unit(ks.P)
    og = ustrip.([ks.K, ks.P, ks.M0, ks.h, ks.k])
    function f2(vector)
        ks_internal = ks_from_vec(vector, K_u, P_u; use_hk=true)
        return GLOM.nlogL(covariance, remove_kepler(data, times, ks_internal, offset; data_unit=data_unit)) - (include_priors * logprior_kepler(ks_internal; use_hk=use_hk))
    end
    return GLOM.est_∇(f2, og; kwargs...)
end
function est_∇nlogL_kep(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    ks::kep_signal_wright,
    offset::Offset;
    data_unit::Unitful.Velocity=1u"m/s",
    include_priors::Bool=true,
    use_hk::Bool=false,
    kwargs...
    ) where T<:Real

    P_u = unit(ks.P)
    og = ustrip.([ks.P, ks.M0, ks.e])
    function f2(vector)
        ks_internal = fit_kepler_wright_linear_step(data, times, covariance, vector[1] * P_u, vector[2], vector[3], offset; data_unit=data_unit)
        return GLOM.nlogL(covariance, remove_kepler(data, times, ks_internal, offset; data_unit=data_unit)) - (include_priors * logprior_kepler(ks_internal; use_hk=use_hk))
    end
    return GLOM.est_∇(f2, og; kwargs...)
end
est_∇nlogL_kep(
    glo_rv::GLO_RV,
    covariance::Union{Cholesky,Diagonal},
    ks::Union{kep_signal, kep_signal_wright},
    offset::Offset;
    include_priors::Bool=true,
    use_hk::Bool=false,
    kwargs...
    ) where T<:Real = est_∇nlogL_kep(glo_rv.GLO.y_obs, glo_rv.time, covariance, ks, offset; data_unit=glo_rv.rv_factor, include_priors=include_priors, use_hk=use_hk, kwargs...)

function est_∇nlogL_offset(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    ks::Union{kep_signal, kep_signal_wright},
    offset::Offset;
    data_unit::Unitful.Velocity=1u"m/s",
    kwargs...
    ) where T<:Real

    o_u = unit(offset.offsets[1])
    og = ustrip.(offset.offsets)
    offset_copy = copy(offset)
    function f2(vector)
        offset_copy.offsets[:] = vector .* o_u
        return GLOM.nlogL(covariance, remove_kepler(data, times, ks, offset_copy; data_unit=data_unit))
    end
    return GLOM.est_∇(f2, og; kwargs...)
end


function test_∇nlogL_kep(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    ks::Union{kep_signal, kep_signal_wright},
    offset::Offset;
    data_unit::Unitful.Velocity=1u"m/s",
    include_priors::Bool=true
    ) where T<:Real

    grad_est = est_∇nlogL_kep(data, times, covariance, ks, offset; data_unit=data_unit, include_priors=include_priors)
    grad = ∇nlogL_kep(data, times, covariance, ks, offset; data_unit=data_unit, include_priors=include_priors)
    return GLOM.test_∇(grad_est, grad; function_name="∇nlogL_kep")
end
test_∇nlogL_kep(
    glo_rv::GLO_RV,
    covariance::Union{Cholesky,Diagonal},
    ks::Union{kep_signal, kep_signal_wright},
    offset::Offset;
    include_priors::Bool=true
    ) where T<:Real = test_∇nlogL_kep(glo_rv.GLO.y_obs, glo_rv.time, covariance, ks, offset; data_unit=glo_rv.rv_factor, include_priors=include_priors)



function est_∇∇nlogL_kep(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    ks::Union{kep_signal, kep_signal_wright};
    kwargs...
    ) where T<:Real

    K_u = unit(ks.K)
    P_u = unit(ks.P)
    og = ustrip.([ks.K, ks.P, ks.M0, ks.h, ks.k])
    g2(vector) = ∇nlogL_kep(data, times, covariance, ks_from_vec(vector, K_u, P_u; use_hk=true); kwargs...)
    return GLOM.est_∇∇(g2, og)
end
est_∇∇nlogL_kep(
    glo_rv::GLO_RV,
    covariance::Union{Cholesky,Diagonal},
    ks::Union{kep_signal, kep_signal_wright};
    kwargs...
    ) where T<:Real = est_∇∇nlogL_kep(glo_rv.GLO.y_obs, glo_rv.time, covariance, ks; data_unit=glo_rv.rv_factor, kwargs...)


function test_∇∇nlogL_kep(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    ks::Union{kep_signal, kep_signal_wright};
    kwargs...
    ) where T<:Real

    hess_est = est_∇∇nlogL_kep(data, times, covariance, ks; kwargs...)
    hess = ∇∇nlogL_kep(data, times, covariance, ks; kwargs...)
    return GLOM.test_∇∇(hess_est, hess; function_name="∇∇nlogL_kep")
end
test_∇∇nlogL_kep(
    glo_rv::GLO_RV,
    covariance::Union{Cholesky,Diagonal},
    ks::Union{kep_signal, kep_signal_wright};
    kwargs...
    ) where T<:Real = test_∇∇nlogL_kep(glo_rv.GLO.y_obs, glo_rv.time, covariance, ks; data_unit=glo_rv.rv_factor, kwargs...)
