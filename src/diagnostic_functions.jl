# these are all custom diagnostic functions. May help with debugging
using Statistics
using LinearAlgebra

function est_∇nlogL_kep(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ};
    data_unit::Unitful.Velocity=1u"m/s"
    ) where T<:Real

    K_u = unit(ks.K)
    P_u = unit(ks.P)
    γ_u = unit(ks.γ)
    og = ustrip.([ks.K, ks.P, ks.M0, ks.h, ks.k, ks.γ])
    f2(vector) = GLOM.nlogL(covariance, remove_kepler(data, times, ks_from_vec(vector, K_u, P_u, γ_u; use_hk=true); data_unit=data_unit))
    return GLOM.est_∇(f2, og)
end
est_∇nlogL_kep(
    prob_def::GLO_RV,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ}
    ) where T<:Real = est_∇nlogL_kep(prob_def.GLO.y_obs, prob_def.time, covariance, ks; data_unit=prob_def.rv_unit*prob_def.GLO.normals[1])


function test_∇nlogL_kep(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ};
    data_unit::Unitful.Velocity=1u"m/s"
    ) where T<:Real

    grad_est = est_∇nlogL_kep(data, times, covariance, ks; data_unit=data_unit)
    grad = ∇nlogL_kep(data, times, covariance, ks; data_unit=data_unit)
    return GLOM.test_∇(grad_est, grad; function_name="∇nlogL_kep")
end
test_∇nlogL_kep(
    prob_def::GLO_RV,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ}
    ) where T<:Real = test_∇nlogL_kep(prob_def.GLO.y_obs, prob_def.time, covariance, ks; data_unit=prob_def.rv_unit*prob_def.GLO.normals[1])



function est_∇∇nlogL_kep(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ};
    data_unit::Unitful.Velocity=1u"m/s"
    ) where T<:Real

    K_u = unit(ks.K)
    P_u = unit(ks.P)
    γ_u = unit(ks.γ)
    og = ustrip.([ks.K, ks.P, ks.M0, ks.h, ks.k, ks.γ])
    g2(vector) = ∇nlogL_kep(data, times, covariance, ks_from_vec(vector, K_u, P_u, γ_u; use_hk=true); data_unit=data_unit)
    return GLOM.est_∇∇(g2, og)
end
est_∇∇nlogL_kep(
    prob_def::GLO_RV,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ}
    ) where T<:Real = est_∇∇nlogL_kep(prob_def.GLO.y_obs, prob_def.time, covariance, ks; data_unit=prob_def.rv_unit*prob_def.GLO.normals[1])


function test_∇∇nlogL_kep(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ};
    data_unit::Unitful.Velocity=1u"m/s"
    ) where T<:Real

    hess_est = est_∇∇nlogL_kep(data, times, covariance, ks; data_unit=data_unit)
    hess = ∇∇nlogL_kep(data, times, covariance, ks; data_unit=data_unit)
    return GLOM.test_∇∇(hess_est, hess; function_name="∇∇nlogL_kep")
end
test_∇∇nlogL_kep(
    prob_def::GLO_RV,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ}
    ) where T<:Real = test_∇∇nlogL_kep(prob_def.GLO.y_obs, prob_def.time, covariance, ks; data_unit=prob_def.rv_unit*prob_def.GLO.normals[1])
