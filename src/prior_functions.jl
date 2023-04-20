# these functions are related to calculating priors on the model parameters
using SpecialFunctions
using LinearAlgebra
using Unitful
using UnitfulAstro
import GPLinearODEMaker; GLOM = GPLinearODEMaker

# Keplerian priors references
# https://arxiv.org/abs/astro-ph/0608328
# table 1

const prior_K_min = 1e-4#u"m/s"  # m/s
const prior_K_max = 2129#u"m/s"  # m/s
const prior_γ_min = -prior_K_max  # m/s
const prior_γ_max = prior_K_max  # m/s
const prior_P_min = 1 * convert_and_strip_units(u"d", 1u"minute")#u"d"  # days
const prior_P_max = 1e3 * convert_and_strip_units(u"d", 1u"yr")#u"d"
const prior_K0 = 0.3#u"m/s"  # * sqrt(50 / n_meas)  # m/s
const prior_e_min = 0
const prior_e_max = 1
const prior_ω_min = 0  # radians
const prior_ω_max = 2 * π  # radians
const prior_M0_min = 0  # radians
const prior_M0_max = 2 * π  # radians
const characteristic_P = 10  # days


# function logprior_K(K::Unitful.Velocity; d::Integer=0, P::Unitful.Time=characteristic_P * u"d")
#     return GLOM.log_loguniform(convert_and_strip_units(u"m/s", K), [prior_K_min, cbrt(prior_P_min / convert_and_strip_units(u"d", P)) * prior_K_max]; d=d, shift=prior_K0)
# end
function logprior_K(K::Unitful.Velocity; d::Integer=0)
    return GLOM.log_loguniform(convert_and_strip_units(u"m/s", K), [prior_K_min, prior_K_max]; d=d, shift=prior_K0)
end

function logprior_P(P::Unitful.Time; d::Integer=0)
    # return GLOM.log_loguniform(convert_and_strip_units(u"d", P), [prior_P_min, prior_P_max]; d=d)
    return GLOM.log_uniform(convert_and_strip_units(u"d", P); min_max=[prior_P_min, prior_P_max], d=d)
end

function logprior_M0(M0::Real; d::Integer=0)
    return GLOM.log_uniform(M0; min_max=[prior_M0_min, prior_M0_max], d=d)
end

function logprior_e(e::Real; d::Integer=0)
    # return log_uniform(e; min_max=[prior_e_min, prior_e_max], d=d)
    return GLOM.log_Rayleigh(e, 1/5; d=d, cutoff=1)
end

function logprior_ω(ω::Real; d::Integer=0)
    return GLOM.log_uniform(ω; min_max=[prior_ω_min, prior_ω_max], d=d)
end

function logprior_hk(h::Real, k::Real; d::Vector{<:Integer}=[0,0])
    # return log_quad_cone([h, k]; d=d)
    return GLOM.log_rot_Rayleigh([h, k], 1/5; d=d, cutoff=1)
end

function logprior_γ(γ::Unitful.Velocity; d::Integer=0)
    return GLOM.log_uniform(convert_and_strip_units(u"m/s", γ); min_max=[prior_γ_min, prior_γ_max], d=d)
end

function logprior_kepler(
    K::Unitful.Velocity,
    P::Unitful.Time,
    M0::Real,
    e_or_h::Real,
    ω_or_k::Real;
    d::Vector{<:Integer}=[0,0,0,0,0],
    use_hk::Bool=false)

    @assert all(0 .<= d .<= 2)
    @assert sum(d) <= 2

    if length(d) > 6 && any(view(d, 6:length(d)) .> 0); return 0. end

    if use_hk
        if any(d[4:5] .!= 0) && all(d[[1,2,3]] .== 0); return logprior_hk(e_or_h, ω_or_k; d=d[4:5]) end
        if sum(d .!= 0) > 1; return 0 end
    else
        if sum(d .!= 0) > 1; return 0 end
        if d[4] != 0; return logprior_e(e_or_h; d=d[4]) end
        if d[5] != 0; return logprior_ω(ω_or_k; d=d[5]) end
    end
    if d[1] != 0; return logprior_K(K; d=d[1]) end
    if d[2] != 0; return logprior_P(P; d=d[2]) end
    if d[3] != 0; return logprior_M0(M0; d=d[3]) end

    logP = logprior_K(K)
    # logP += logprior_K(K; P=P)
    logP += logprior_P(P)
    logP += logprior_M0(M0)
    use_hk ? logP += logprior_hk(e_or_h, ω_or_k) : logP += logprior_e(e_or_h; d=d[4]) + logprior_ω(ω_or_k; d=d[5])

    return logP

end
logprior_kepler(ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright}; d::Vector{<:Integer}=zeros(Int, n_kep_parms), use_hk::Bool=false) =
    use_hk ? logprior_kepler(ks.K, ks.P, ks.M0, ks.h, ks.k; d=d, use_hk=use_hk) : logprior_kepler(ks.K, ks.P, ks.M0, ks.e, ks.ω; d=d, use_hk=use_hk)
function logprior_kepler_tot(ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright}; d_tot::Integer=0, use_hk::Bool=false)
    @assert 0 <= d_tot <= 2
    if d_tot == 0
        return logprior_kepler(ks; d=zeros(Int, n_kep_parms), use_hk=use_hk)
    elseif d_tot == 1
        G = zeros(n_kep_parms)
        d = zeros(Int, n_kep_parms)
        for i in 1:n_kep_parms
            d[i] = 1
            G[i] = logprior_kepler(ks; use_hk=use_hk, d=d)
            d[i] = 0
        end
        return G
    else
        H = zeros(n_kep_parms, n_kep_parms)
        d = zeros(Int, n_kep_parms)
        for i in 1:n_kep_parms
            for j in 1:n_kep_parms
                if i <= j
                    d[i] += 1
                    d[j] += 1
                    # println(typeof(d))
                    H[i, j] = logprior_kepler(ks; d=d, use_hk=use_hk)
                    d[i] = 0
                    d[j] = 0
                end
            end
        end
        return Symmetric(H)
    end
end
