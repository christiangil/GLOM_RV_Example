# these functions are related to calculating RV quantities
using UnitfulAstro, Unitful
using LinearAlgebra
using Optim, LineSearches
import GPLinearODEMaker; GLOM = GPLinearODEMaker

const n_kep_parms = 6

abstract type KeplerSignal end

"Convert Unitful units from one to another and strip the final units"
convert_and_strip_units(new_unit::Unitful.FreeUnits, quant::Quantity) = ustrip(uconvert(new_unit, quant))
convert_and_strip_units(new_unit::Unitful.FreeUnits, quant) = quant


"""
Calculate velocity semi-amplitude for RV signal of planets in m/s
adapted from eq. 12 of (http://exoplanets.astro.yale.edu/workshop/EPRV/Bibliography_files/Radial_Velocity.pdf)
"""
function velocity_semi_amplitude(
    m_star::Unitful.Mass,
    m_planet::Unitful.Mass,
    P::Unitful.Time;
    e::Real=0.,
    i::Real=π/2)

    # m_star = convert_and_strip_units(u"Msun", m_star)
    # m_planet = convert_and_strip_units(u"Msun", m_planet)
    # P = convert_and_strip_units(u"yr", P)
    assert_positive(P, m_star, m_planet)
    comb_mass = m_star + m_planet
    K_au_yr = (2 * π * sin(i) * m_planet / (sqrt((1-e)*(1+e)) * cbrt(comb_mass * comb_mass * P)))u"AU/yr"
    return uconvert(u"m/s", K_au_yr)
end


# unit_phase(t::Unitful.Time, P::Unitful.Time) = mod(t / P * 2 * π, 2 * π)
unit_phase(t::Unitful.Time, P::Unitful.Time) = t / P * 2 * π


"calculate mean anomaly"
mean_anomaly(t::Unitful.Time, P::Unitful.Time, M0::Real) = unit_phase(t, P) - M0
# mean_anomaly(t::Unitful.Time, P::Unitful.Time, M0::Real) = mod(unit_phase(t, P) - M0, 2 * π)
# mean_anomaly(t::Unitful.Time, P::Unitful.Time, M0::Real) = mod(t / P * 2 * π - M0, 2 * π)


"""   ecc_anom_init_guess_danby(M, ecc)
Initial guess for eccentric anomaly given mean anomaly (M) and eccentricity (ecc)
    Based on "The Solution of Kepler's Equations - Part Three"
    Danby, J. M. A. (1987) Journal: Celestial Mechanics, Volume 40, Issue 3-4, pp. 303-312  1987CeMec..40..303D
Julia function originally written by Eric Ford
"""
function ecc_anom_init_guess_danby(M::Real, e::Real)
    k = convert(typeof(e), 0.85)
    if M < zero(M); M += 2 * π end
    (M < π) ? M + k * e : M - k * e;
end


"""   update_ecc_anom_laguerre(E, M, ecc)
Update the current guess (E) for the solution to Kepler's equation given mean anomaly (M) and eccentricity (ecc)
    Based on "An Improved Algorithm due to Laguerre for the Solution of Kepler's Equation"
    Conway, B. A.  (1986) Celestial Mechanics, Volume 39, Issue 2, pp.199-211  1986CeMec..39..199C
Julia function originally written by Eric Ford
Basically looks for root of an nth order polynomial assuming that 1 root is
the one you are looking for (E) and the rest are evenly spaced
"""
function update_ecc_anom_laguerre(E::Real, M::Real, e::Real)
    # es = e * sin(E)
    # ec = e * cos(E)
    F = (E - e * sin(E)) - M
    Fp = one(E) - e * cos(E)
    # Fpp = es
    n = 5
    root = sqrt(abs((n - 1) * ((n - 1) * Fp * Fp - n * F * e * sin(E))))
    denom = Fp > zero(E) ? Fp + root : Fp - root
    return E - n * F / denom
end


"""
Loop to update the current estimate of the solution to Kepler's equation
Julia function originally written by Eric Ford
"""
function ecc_anomaly(
    t::Unitful.Time,
    P::Unitful.Time,
    M0::Number,
    e::Number;
    tol::Number=1e-8,
    max_its::Integer=200)

    # P = convert_and_strip_units(u"yr", P)
    # @assert (0 <= e <= 1) "eccentricity has to be between 0 and 1"
    M = mean_anomaly(t, P, M0)
    E = ecc_anom_init_guess_danby(M, e)
    for i in 1:max_its
       E_old = E
       E = update_ecc_anom_laguerre(E_old, M, e)
       if abs(E - E_old) < tol; break end
    end
    return E
end


# """
# Iterative solution for eccentric anomaly from mean anomaly
# from (http://www.csun.edu/~hcmth017/master/node16.html)
# could also implement third order method from
# (http://alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf)
# """
# function ecc_anomaly(t::Unitful.Time, P::Unitful.Time, e::Real)
#     @assert (0 <= e <= 1) "eccentricity has to be between 0 and 1"
#     M = mean_anomaly(t, P)
#     dif_thres = 1e-8
#     dif = 1
#     ecc_anom = copy(M)
#     while dif > dif_thres
#         ecc_anom_old = copy(ecc_anom)
#         ecc_anom -= (ecc_anom - e * sin(ecc_anom) - M) / (1 - e * cos(ecc_anom))  # first order method
#         dif = abs((ecc_anom - ecc_anom_old) / ecc_anom)
#     end
#     return ecc_anom
# end


"""
Solution for true anomaly from eccentric anomaly
from (https://en.wikipedia.org/wiki/True_anomaly#From_the_eccentric_anomaly)
"""
ϕ(
    E::Real,
    e::Real
    ) = 2 * atan( sqrt(1 + e)*sin(0.5*E), sqrt(1 - e) * cos(0.5*E) )

ϕ(
    t::Unitful.Time,
    P::Unitful.Time,
    M0::Real,
    e::Real
    ) = ϕ( ecc_anomaly(t, P, M0, e), e)

# """
# Calculate true anomaly for small e using equation of center approximating true
# anomaly from (https://en.wikipedia.org/wiki/Equation_of_the_center) O(e^8)
# """
# function ϕ_approx(t::Unitful.Time, P::Unitful.Time, e::Real)
#     @assert (0 <= e <= 1) "eccentricity has to be between 0 and 1"
#     M = mean_anomaly(t, P)
#
#     term_list = [eval_polynomial(e, [0, 2, 0, - 1/4, 0, 5/96, 0, 107/4608]),
#     eval_polynomial(e, [0, 0, 5/4, 0, -11/24, 0, 17/192, 0]),
#     eval_polynomial(e, [0, 0, 0, 13/12, 0, -43/64, 0, 95/512]),
#     eval_polynomial(e, [0, 0, 0, 0, 103/96, 0, -451/480, 0]),
#     eval_polynomial(e, [0, 0, 0, 0, 0, 1097/960, 0, -5957/4608]),
#     eval_polynomial(e, [0, 0, 0, 0, 0, 0, 1223/960, 0]),
#     eval_polynomial(e, [0, 0, 0, 0, 0, 0, 0, 47273/32256])]
#
#     sin_list = [sin(i * M) for i in 1:7]
#
#     return M + dot(term_list, sin_list)
#
# end
#
# ϕ_approx(t::Unitful.Time, P::Unitful.Time, e::Real) = ϕ_approx(t, convert_and_strip_units(u"yr", P), e)


"""
kepler_rv(t, K, P, M0, e, ω; γ)
Calculate radial velocity of star due to planet following Keplerian orbit at specified time.
Inputs:
- time
- Period
- eccentricity
- mean anomaly at time=0
- RV amplitude
- argument of periapsis
Optionally:
- velocity offset
"""
function kepler_rv(
    t::Unitful.Time,
    K::Unitful.Velocity,
    P::Unitful.Time,
    M0::Real,
    e_or_h::Real,
    ω_or_k::Real;
    kwargs...)

    return kepler_rv_ecc_anom(t, K, P, M0, e_or_h, ω_or_k; kwargs...)
    # return kepler_rv_true_anom(t, P, M0, e, K, ω; kwargs...)
end


function kepler_rv(
    t::Unitful.Time,
    m_star::Unitful.Mass,
    m_planet::Unitful.Mass,
    P::Unitful.Time,
    M0::Real,
    e::Real,
    ω::Real;
    i::Real=π/2,
    kwargs...)

    K = velocity_semi_amplitude(P, m_star, m_planet, e=e, i=i)
    return kepler_rv(t, K, P, M0, e, ω; kwargs...)
end


struct kep_signal <: KeplerSignal
    K::Unitful.Velocity
    P::Unitful.Time
    M0::Real  # ::Unitful.NoDims
    e::Real
    ω::Real  # ::Unitful.NoDims
    γ::Unitful.Velocity
    h::Real  # ::Unitful.NoDims
    k::Real  # ::Unitful.NoDims

    kep_signal(; K=0u"m/s", P=1u"d", M0=0, e_or_h=0, ω_or_k=0, γ=0u"m/s", use_hk::Bool=false) = kep_signal(K, P, M0, e_or_h, ω_or_k, γ; use_hk=use_hk)
    function kep_signal(K, P, M0, e_or_h, ω_or_k, γ; use_hk::Bool=false)
        if use_hk
            h = e_or_h
            k = ω_or_k
            e, ω = hk_2_eω(h, k)
        else
            e = e_or_h
            ω = ω_or_k
            h, k = eω_2_hk(e, ω)
        end
        return kep_signal(K, P, M0, e, ω, γ, h, k)
    end
    function kep_signal(
        K::Unitful.Velocity,
        P::Unitful.Time,
        M0::Real,
        e::Real,
        ω::Real,
        γ::Unitful.Velocity,
        h::Real,
        k::Real)

        # @assert 0 <= e < 1 "orbit needs to be bound"
        M0 = mod2pi(M0)
        ω = mod2pi(ω)
        # assert_positive(P)
        return new(K, P, M0, e, ω, γ, h, k)
    end
end
kepler_rv(t::Unitful.Time, ks::kep_signal) = kepler_rv(t, ks.K, ks.P, ks.M0, ks.e, ks.ω; γ=ks.γ)
(ks::KeplerSignal)(t::Unitful.Time) = kepler_rv(t, ks)

mutable struct kep_buffer{T1<:Real}
    ks::kep_signal
    rm_kep::Vector{T1}
    nprior::T1
end

function ∇nlogL_kep!(
    G::Vector{T},
    d::Vector{<:Integer},
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    ks::kep_signal,
    y::Vector{T};
    data_unit::Unitful.Velocity=1u"m/s",
    include_priors::Bool=true) where T<:Real

    α = covariance \ y
    for i in 1:length(G)
        d[i] = 1
        G[i] = GLOM.dnlogLdθ(remove_kepler(data, times, ks; data_unit=data_unit, d=d), α) - (include_priors * logprior_kepler(ks; d=d, use_hk=true))
        d[i] = 0
    end
end
∇nlogL_kep(data, times, covariance, ks; data_unit=data_unit, kwargs...) =
    ∇nlogL_kep(data, times, covariance, ks, remove_kepler(data, times, ks; data_unit=data_unit); data_unit=data_unit, kwargs...)
function ∇nlogL_kep(data, times, covariance, ks, y; G=zeros(typeof(data[1]), n_kep_parms), d=zeros(Int, n_kep_parms), kwargs...)
    ∇nlogL_kep!(G, d, data, times, covariance, ks, y; kwargs...)
    return G
end
∇nlogL_kep(glo_rv::GLO_RV, covariance, ks; kwargs...) =
    ∇nlogL_kep(glo_rv.GLO.y_obs, glo_rv.time, covariance, ks; data_unit=glo_rv.rv_factor, kwargs...)

function fit_kepler(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    init_ks::kep_signal;
    data_unit::Unitful.Velocity=1u"m/s",
    print_stuff::Bool=true,
    include_priors::Bool=true,
    avoid_saddle::Bool=true
    ) where T<:Real

    # if P==0; return kep_signal_epicyclic(p, P, M0, e, ω, γ, coefficients) end
    assert_positive(init_ks.P)
    for i in 1:ndims(covariance)
        @assert size(covariance,i)==length(data) "covariance incompatible with data"
    end
    n_total_samp_points = length(data)
    n_samp_points = length(times)
    normalization = logdet(covariance) + n_total_samp_points * log(2 * π)
    K_u = unit(init_ks.K)
    P_u = unit(init_ks.P)
    γ_u = unit(init_ks.γ)

    current_x = ustrip.([init_ks.K, init_ks.P, init_ks.M0, init_ks.h, init_ks.k, init_ks.γ])
    buffer = kep_buffer(init_ks, remove_kepler(data, times, init_ks; data_unit=data_unit), -logprior_kepler(init_ks; use_hk=true))
    last_x = similar(current_x)

    function calculate_common!(x::Vector{T}, last_x::Vector{T}, buffer::kep_buffer) where {T<:Real}
        if x != last_x
            # copy!(last_x, x)
            last_x[:] = x
            buffer.ks = ks_from_vec(x, K_u, P_u, γ_u; use_hk=true)
            buffer.nprior = -logprior_kepler(buffer.ks; use_hk=true)
            buffer.nprior == Inf ? buffer.rm_kep[:] .= 0 : buffer.rm_kep = remove_kepler(data, times, buffer.ks; data_unit=data_unit)
        end
        # println("buffer: ", kep_parms_str(buffer.ks))
    end

    function f(x::Vector{T}, buffer::kep_buffer, last_x::Vector{T}) where {T<:Real}
        calculate_common!(x, last_x, buffer)
        # println(kep_parms_str(buffer.ks))
        # println(buffer.nprior)
        if buffer.nprior == Inf
            return buffer.nprior
        else
            return GLOM.nlogL(covariance, buffer.rm_kep, nlogL_normalization=normalization) + (include_priors * buffer.nprior)
        end
    end
    f(x) = f(x, buffer, last_x)

    d = zeros(Int, n_kep_parms)
    function g!(G::Vector{T}, x::Vector{T}, buffer::kep_buffer, last_x::Vector{T}) where {T<:Real}
        calculate_common!(x, last_x, buffer)
        if buffer.nprior == Inf
            G[:] .= 0
        else
            ∇nlogL_kep!(G, d, data, times, covariance, buffer.ks, buffer.rm_kep; data_unit=data_unit, include_priors=include_priors)
        end
    end
    g!(G, x) = g!(G, x, buffer, last_x)
    attempts = 0
    in_saddle = true
    while attempts < 120 && in_saddle
        attempts += 1
        if attempts > 1;
            if print_stuff; println("found saddle point. starting attempt $attempts with a perturbation") end
            current_x[1] = maximum([current_x[1] + centered_rand(; scale=0.3), 0.2])
            current_x[2] *= centered_rand(; scale=2e-2, center=1)
            current_x[3] = mod2pi(current_x[3] + centered_rand(; scale=4e-1))
            # e, ω = hk_2_eω(current_x[4], current_x[5])
            # e = maximum([minimum([e, 0.1]), 0])
            # ω = mod2pi(ω)
            current_x[4:5] .= eω_2_hk(0.05 * rand(), mod2pi(attempts * π / 3))
            current_x[6] += centered_rand(; scale=1e-1)
        end
        # println(current_x)
        result = optimize(f, g!, current_x, LBFGS(alphaguess=LineSearches.InitialStatic(alpha=3e-3))) # 27s
        current_x = copy(result.minimizer)
        ks = ks_from_vec(current_x, K_u, P_u, γ_u; use_hk=true)
        if print_stuff; println("fit attempt $attempts: "kep_parms_str(ks)) end
        # println(∇∇nlogL_kep(data, times, covariance, ks; data_unit=data_unit))
        new_det = det(∇∇nlogL_kep(data, times, covariance, ks; data_unit=data_unit, include_priors=include_priors))
        if print_stuff; println("determinant: ", new_det) end
        in_saddle = (new_det <= 0) && avoid_saddle
        if !in_saddle; return ks end
    end
    @error "no non-saddle point soln found"
end


"""
Simple radial velocity formula using true anomaly
adapted from eq. 11 of (http://exoplanets.astro.yale.edu/workshop/EPRV/Bibliography_files/Radial_Velocity.pdf)
"""
kepler_rv_true_anom(
    t::Unitful.Time,
    K::Unitful.Velocity,
    P::Unitful.Time,
    M0::Real,
    e::Real,
    ω::Real;
    γ::Unitful.Velocity=0u"m/s"
    ) = K * (e * cos(ω) + cos(ω + ϕ(t, P, M0, e))) + γ


function eω_2_hk(e, ω)
    return e * sin(ω), e * cos(ω)
end
function hk_2_eω(h, k)
    return sqrt(h * h + k * k), atan(h, k)
end


"""
Radial velocity formula directly using eccentric anomaly
Based on simplification of "An analytical solution for Kepler's problem"
Pál, András, Monthly Notices of the Royal Astronomical Society, 396, 3, 1737-1742.  2009MNRAS.396.1737P
see ηdot part of eq. 19
"""
function kepler_rv_ecc_anom(
    t::Unitful.Time,
    K::Unitful.Velocity,
    P::Unitful.Time,
    M0::Real,
    e_or_h::Real,
    ω_or_k::Real;
    γ::Unitful.Velocity=0u"m/s",
    use_hk::Bool=false)

    if use_hk
        h = e_or_h
        k = ω_or_k
        e, ω = hk_2_eω(h, k)
    else
        e = e_or_h
        ω = ω_or_k
        h, k = eω_2_hk(e, ω)
    end

    # k = e * cos(ω)
    E = ecc_anomaly(t, P, M0, e)
    j = sqrt((1-e)*(1+e))
    q = e * cos(E)
    # return K * j / (1 - q) * (cos(ω + E) - (1 - j) * cos(E) * cos(ω)) + γ  # equivalent
    return K * j / (1 - q) * (cos(ω + E) - k * q / (1 + j)) + γ
end


struct kep_signal_circ <: KeplerSignal
    K::Unitful.Velocity
    P::Unitful.Time
    M0minusω::Real  # ::Unitful.NoDims
    γ::Unitful.Velocity
    coefficients::Vector{T} where T<:Unitful.Velocity

    kep_signal_circ(; K=0u"m/s", P=0u"d", M0minusω=0, γ=0u"m/s") = kep_signal_circ(K, P, M0minusω, γ)
    kep_signal_circ(K, P, M0minusω, γ) = kep_signal_circ(K, P, M0minusω, γ, [K * cos(M0minusω), K * sin(M0minusω), γ])
    function kep_signal_circ(
        P::Unitful.Time,
        coefficients::Vector{T} where T<:Unitful.Velocity)

        return kep_signal_circ(
            sqrt(coefficients[1]^2 + coefficients[2]^2),
            P,
            mod2pi(atan(coefficients[2], coefficients[1])),
            coefficients[3],
            coefficients)
    end
    function kep_signal_circ(
        K::Unitful.Velocity,
        P::Unitful.Time,
        M0minusω::Real,  # ::Unitful.NoDims
        γ::Unitful.Velocity,
        coefficients::Vector{T} where T<:Unitful.Velocity)

        M0minusω = mod2pi(M0minusω)
        assert_positive(P)
        @assert length(coefficients) == 3 "wrong number of coefficients"
        return new(K, P, M0minusω, γ, coefficients)
    end
end
function kepler_rv_circ(t::Unitful.Time, ks::kep_signal_circ)
    phase = unit_phase.(t, ks.P)
    return (ks.coefficients[1] .* cos.(phase)) + (ks.coefficients[2] .* sin.(phase)) .+ ks.coefficients[3]
end
kepler_rv(t::Unitful.Time, ks::kep_signal_circ) = kepler_rv_circ(t, ks)


"""
A linear formulation of a Keplerian RV signal for low eccentricity by expanding about e=0
average difference is < 1% for e < 0.1 and < 10% for e < 0.35
for initial full orbit fits
coefficients[1] = K * cos(M0 - ω)
coefficients[2] = K * sin(M0 - ω)
coefficients[3] = e * K * cos(2 * M0 - ω)
coefficients[4] = e * K * sin(2 * M0 - ω)
coefficients[5] = γ
so
K = sqrt(coefficients[1]^2 + coefficients[2]^2)
e = sqrt(coefficients[3]^2 + coefficients[4]^2) / K
M0 = atan(coefficients[4], coefficients[3]) - atan(coefficients[2], coefficients[1])
ω = atan(coefficients[4], coefficients[3]) - 2 * atan(coefficients[2], coefficients[1])
"""
struct kep_signal_epicyclic <: KeplerSignal
    K::Unitful.Velocity
    P::Unitful.Time
    M0::Real
    e::Real
    ω::Real  # ::Unitful.NoDims
    γ::Unitful.Velocity
    h::Real
    k::Real  # ::Unitful.NoDims
    coefficients::Vector{T} where T<:Unitful.Velocity

    kep_signal_epicyclic(; K=0u"m/s", P=1u"d", M0=0, e_or_h=0, ω_or_k=0, γ=0u"m/s", use_hk::Bool=false) = kep_signal_epicyclic(K, P, M0, e_or_h, ω_or_k, γ; use_hk=use_hk)
    function kep_signal_epicyclic(K, P, M0, e_or_h, ω_or_k, γ; use_hk::Bool=false)
        if use_hk
            h = e_or_h
            k = ω_or_k
            e, ω = hk_2_eω(h, k)
        else
            e = e_or_h
            ω = ω_or_k
            h, k = eω_2_hk(e, ω)
        end
        return kep_signal_epicyclic(K, P, M0, e, ω, γ, h, k)
    end
    kep_signal_epicyclic(K, P, M0, e, ω, γ, h, k) = kep_signal_epicyclic(K, P, M0, e, ω, γ, h, k,
        [K * cos(M0 - ω), K * sin(M0 - ω), e * K * cos(2 * M0 - ω), e * K * sin(2 * M0 - ω), γ])
    function kep_signal_epicyclic(P::Unitful.Time, coefficients)
        K = sqrt(coefficients[1]^2 + coefficients[2]^2)
        e = sqrt(coefficients[3]^2 + coefficients[4]^2) / K
        # if e > 0.35; @warn "an orbit with this eccentricity would be very different from the output of kepler_rv_epicyclic" end
        M0 = mod2pi(atan(coefficients[4], coefficients[3]) - atan(coefficients[2], coefficients[1]))
        ω = mod2pi(atan(coefficients[4], coefficients[3]) - 2 * atan(coefficients[2], coefficients[1]))
        γ = coefficients[5]
        h, k = eω_2_hk(e, ω)
        return kep_signal_epicyclic(K, P, M0, e, ω, γ, h, k, coefficients)
    end
    function kep_signal_epicyclic(
        K::Unitful.Velocity,
        P::Unitful.Time,
        M0::Real,
        e::Real,
        ω::Real,  # ::Unitful.NoDims
        γ::Unitful.Velocity,
        h::Real,
        k::Real,
        coefficients::Vector{T} where T<:Unitful.Velocity)

        M0 = mod2pi(M0)
        ω = mod2pi(ω)
        assert_positive(P)
        @assert length(coefficients) == 5 "wrong number of coefficients"
        return new(K, P, M0, e, ω, γ, h, k, coefficients)
    end
end
function kepler_rv_epicyclic(t::Unitful.Time, ks::kep_signal_epicyclic)
    phase = unit_phase.(t, ks.P)
    return return (ks.coefficients[1] .* cos.(phase)) + (ks.coefficients[2] .* sin.(phase)) + (ks.coefficients[3] .* cos.(2 .* phase)) + (ks.coefficients[4] .* sin.(2 .* phase)) .+ ks.coefficients[5]
end
kepler_rv(t::Unitful.Time, ks::kep_signal_epicyclic) = kepler_rv_epicyclic(t, ks)
function fit_kepler_epicyclic(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    P::Unitful.Time;
    data_unit::Unitful.Velocity=1u"m/s",
    print_stuff::Bool=true,
    avoid_saddle::Bool=true
    ) where T<:Real


    # if P==0; return kep_signal_epicyclic(p, P, M0, e, ω, γ, coefficients) end
    assert_positive(P)
    for i in 1:ndims(covariance)
        @assert size(covariance,i)==length(data) "covariance incompatible with data"
    end
    n_total_samp_points = length(data)
    n_samp_points = length(times)
    n_out = Int(n_total_samp_points / n_samp_points)
    phases = unit_phase.(times, P)
    design_matrix = zeros(n_total_samp_points, 5)
    design_matrix[1:n_out:end, :] = hcat(cos.(phases), sin.(phases), cos.(2 .* phases), sin.(2 .* phases), ones(length(times)))
    # n_total_samp_points > n_samp_points ? design_matrix = vcat(design_matrix, zeros(n_total_samp_points - n_samp_points, size(design_matrix, 2))) : design_matrix = design_matrix
    x = general_lst_sq(design_matrix, data; Σ=covariance)
    return kep_signal_epicyclic(P, x .* data_unit)
end
fit_kepler(data, times, covariance, ks::kep_signal_epicyclic; kwargs...) = fit_kepler_epicyclic(data, times, covariance, ks.P; kwargs...)

"""
A linear formulation of a Keplerian RV signal for a given eccentricity, period and initial mean anomaly
for fitting non-(e, P, M0) components
see Wright and Howard 2009 "EFFICIENT FITTING OF MULTI-PLANET KEPLERIAN MODELS TO RADIAL VELOCITY AND ASTROMETRY DATA" (https://arxiv.org/abs/0904.3725)
coefficients[1] = K * cos(ω)
coefficients[2] = -K * sin(ω)
coefficients[3] = γ + K * e * cos(ω)
so
K = sqrt(coefficients[1]^2 + coefficients[2]^2)
ω = atan(-coefficients[2], coefficients[1])
γ = coefficients[3] - K * e * cos(ω) = coefficients[3] - e * coefficients[1]
"""
struct kep_signal_wright <: KeplerSignal
    K::Unitful.Velocity
    P::Unitful.Time
    M0::Real
    e::Real
    ω::Real  # ::Unitful.NoDims
    γ::Unitful.Velocity
    h::Real
    k::Real
    coefficients::Vector{T} where T<:Unitful.Velocity
    kep_signal_wright(; K=0u"m/s", P=1u"d", M0=0, e_or_h=0, ω_or_k=0, γ=0u"m/s", use_hk::Bool=false) = kep_signal_wright(K, P, M0, e_or_h, ω_or_k, γ; use_hk=use_hk)
    function kep_signal_wright(K, P, M0, e_or_h, ω_or_k, γ; use_hk::Bool=false)
        if use_hk
            h = e_or_h
            k = ω_or_k
            e, ω = hk_2_eω(h, k)
        else
            e = e_or_h
            ω = ω_or_k
            h, k = eω_2_hk(e, ω)
        end
        return kep_signal_wright(K, P, M0, e, ω, γ, h, k)
    end
    kep_signal_wright(K, P, M0, e, ω, γ, h, k) = kep_signal_wright(K, P, M0, e, ω, γ, h, k,
        [K * cos(ω), -K * sin(ω), γ + K * e * cos(ω)])
    function kep_signal_wright(P::Unitful.Time, M0::Real, e::Real, coefficients::Vector{T} where T<:Unitful.Velocity)
        K = sqrt(coefficients[1]^2 + coefficients[2]^2)
        ω = mod2pi(atan(-coefficients[2], coefficients[1]))
        γ = coefficients[3] - e * coefficients[1]
        h, k = eω_2_hk(e, ω)
        return kep_signal_wright(K, P, M0, e, ω, γ, h, k, coefficients)
    end
    function kep_signal_wright(
        K::Unitful.Velocity,
        P::Unitful.Time,
        M0::Real,
        e::Real,
        ω::Real,  # ::Unitful.NoDims
        γ::Unitful.Velocity,
        h::Real,
        k::Real,
        coefficients::Vector{T} where T<:Unitful.Velocity)

        M0 = mod2pi(M0)
        ω = mod2pi(ω)
        assert_positive(P)
        @assert length(coefficients) == 3 "wrong number of coefficients"
        return new(K, P, M0, e, ω, γ, h, k, coefficients)
    end
end
function kepler_rv_wright(t::Unitful.Time, ks::kep_signal_wright)
    ϕ_t = ϕ(t, ks.P, ks.M0, ks.e)
    return (ks.coefficients[1] * cos.(ϕ_t)) + (ks.coefficients[2] * sin.(ϕ_t)) + ks.coefficients[3]
end
kepler_rv(t::Unitful.Time, ks::kep_signal_wright) = kepler_rv_wright(t, ks)
# kepler_rv(t::Unitful.Time, ks::kep_signal_wright) = kepler_rv(t, ks.K, ks.P, ks.M0, ks.e, ks.ω; γ=ks.γ)
function fit_kepler_wright_linear_step(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    P::Unitful.Time,
    M0::Real,
    e::Real;
    data_unit::Unitful.Velocity=1u"m/s",
    return_extra::Bool=false
    ) where T<:Real


    # if P==0; return kep_signal_epicyclic(p, P, M0, e, ω, γ, coefficients) end
    # println(P, M0, e)
    assert_positive(P, e)
    for i in 1:ndims(covariance)
        @assert size(covariance,i)==length(data) "covariance incompatible with data"
    end
    n_total_samp_points = length(data)
    n_samp_points = length(times)
    n_out = Int(n_total_samp_points / n_samp_points)
    # println(e)
    if return_extra
        E_t = ecc_anomaly.(times, P, M0, e)
        ϕ_t = ϕ.(E_t, e)
    else
        ϕ_t = ϕ.(times, P, M0, e)
    end
    design_matrix = zeros(n_total_samp_points, 3)
    design_matrix[1:n_out:end, :] = hcat(cos.(ϕ_t), sin.(ϕ_t), ones(n_samp_points))
    # n_total_samp_points > n_samp_points ? design_matrix = vcat(design_matrix, zeros(n_total_samp_points - n_samp_points, size(design_matrix, 2))) : design_matrix = design_matrix
    if return_extra
        β, ϵ_int, ϵ_inv = general_lst_sq(design_matrix, data; Σ=covariance, return_ϵ_inv=true)
        return kep_signal_wright(P, M0, e, β .* data_unit), ϵ_int, ϵ_inv, design_matrix, E_t
    else
        return kep_signal_wright(P, M0, e, general_lst_sq(design_matrix, data; Σ=covariance) .* data_unit)
    end
end
fit_kepler_wright_linear_step(
    glo_rv::GLO_RV,
    covariance::Union{Cholesky,Diagonal} where T<:Real,
    P::Unitful.Time,
    M0::Real,
    e::Real;
    kwargs...) =
    fit_kepler_wright_linear_step(glo_rv.GLO.y_obs, glo_rv.time, covariance, P, M0, e; data_unit=glo_rv.rv_factor, kwargs...)


mutable struct kep_buffer_wright{T1<:Real}
    ks::kep_signal_wright
    rm_kep::Vector{T1}
    nprior::T1
    ϵ_int::Matrix{T1}
    ϵ_inv::Matrix{T1}
    design_matrix::Matrix{T1}
    E_t::Vector{T1}
end

function ∇nlogL_kep!(
    G::Vector{T},
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    buffer::kep_buffer_wright;
    data_unit::Unitful.Velocity=1u"m/s",
    hold_P::Bool=false,
    include_priors::Bool=true) where T<:Real

    @assert length(G) == (3 - hold_P)

    α_data = covariance \ data
    α_rm_kep = covariance \ buffer.rm_kep
    n_total_samp_points = length(data)
    n_samp_points = length(times)
    n_out = Int(n_total_samp_points / n_samp_points)
    e = buffer.ks.e
    mod_e2 = (1 + e) * (1 - e)
    sqrt_mod_e2 = sqrt(mod_e2)

    n_out = Int(length(data) / length(times))
    F = buffer.design_matrix
	sin_ϕ = F[1:n_out:end, 2]
	cos_ϕ = F[1:n_out:end, 1]
    β = buffer.ks.coefficients ./ data_unit

    dFdx_holder = zeros(n_total_samp_points, 3)
    function dudx!(dFdx::Matrix{T}, dϕdx::Vector{T}) where T<:Real
        dFdx[1:n_out:end, 1] = -sin_ϕ .* dϕdx
        dFdx[1:n_out:end, 2] = cos_ϕ .* dϕdx
        dβdx = (-buffer.ϵ_inv \ (((dFdx' * buffer.ϵ_int) + (F' * (covariance \ dFdx))) * (buffer.ϵ_inv \ (F' * α_data)))) + (buffer.ϵ_inv \ (dFdx' * α_data))
        return (dFdx * β) + (F * dβdx), dβdx
    end

    factor = 1 .- (e .* cos.(buffer.E_t))
    factor2 = factor .* factor
    dϕdM0 = -sqrt_mod_e2 ./ factor2
    dϕde = (mod_e2 .+ factor) .* sin.(buffer.E_t) ./ (sqrt_mod_e2 .* factor2)

    dudM0, dβdM0 = dudx!(dFdx_holder, dϕdM0)
    dude, dβde = dudx!(dFdx_holder, dϕde)

    G[end-1] = GLOM.dnlogLdθ(-dudM0, α_rm_kep)
    G[end] = GLOM.dnlogLdθ(-dude, α_rm_kep)

    if !hold_P
        t = convert_and_strip_units.(unit(buffer.ks.P), times)
        P = ustrip(buffer.ks.P)
        dϕdP = (-2 * π * sqrt_mod_e2 / P / P) .* t ./ factor2
        dudP, dβdP = dudx!(dFdx_holder, dϕdP)
        G[1] = GLOM.dnlogLdθ(-dudP, α_rm_kep)
        # println("dβdP:  ", dβdP)
        # println("dβdM0: ", dβdM0)
        # println("dβde:  ", dβde)
    end

    if include_priors
        kep_prior_G = logprior_kepler_tot(buffer.ks; d_tot=1, use_hk=false)

        K = buffer.ks.K / data_unit
        # K = sqrt(β[1]^2 + β[2]^2)
        function dPriordx(dβdx::Vector{T}, δPriorδx::T, dedx::Integer) where T<:Real
            dKdx = (β[1] * dβdx[1] + β[2] * dβdx[2]) / K * ustrip(data_unit)
            dωdx = (β[2] * dβdx[1] - β[1] * dβdx[2]) / (K * K)
            dγdx = (dβdx[3] - e * dβdx[1] - dedx * β[1]) * ustrip(data_unit)
            return kep_prior_G[1] * dKdx + kep_prior_G[5] * dωdx + kep_prior_G[6] * dγdx + δPriorδx
        end

        if !hold_P; G[1] -= dPriordx(dβdP, kep_prior_G[2], 0) end
        G[end-1] -= dPriordx(dβdM0, kep_prior_G[3], 0)
        G[end] -= dPriordx(dβde, kep_prior_G[4], 1)
    end
end
function ∇nlogL_kep(data, times, covariance, init_ks::kep_signal_wright; kwargs...)
    ks, ϵ_int, ϵ_inv, design_matrix, E_t = fit_kepler_wright_linear_step(data, times, covariance, init_ks.P, init_ks.M0, init_ks.e; data_unit=data_unit, return_extra=true)
    buffer = kep_buffer_wright(ks, remove_kepler(data, times, ks; data_unit=data_unit), -logprior_kepler(ks; use_hk=false), ϵ_int, ϵ_inv, design_matrix, E_t)
    G = zeros(3 - hold_P)
    ∇nlogL_kep!(G, data, times, covariance, buffer; kwargs...)
    return G
end


"You may have to mess wit the fit alpha"
function fit_kepler_wright(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    init_ks::kep_signal_wright;
    data_unit::Unitful.Velocity=1u"m/s",
    print_stuff::Bool=true,
    hold_P::Bool=false,
    avoid_saddle::Bool=true,
    fit_alpha::Real=3e-3,
    include_priors::Bool=true
    ) where T<:Real

    optim_cb_local(x::OptimizationState) = optim_cb(x; print_stuff=print_stuff)

    # print("its this function")
    assert_positive(init_ks.P, init_ks.e)
    for i in 1:ndims(covariance)
        @assert size(covariance,i)==length(data) "covariance incompatible with data"
    end

    n_total_samp_points = length(data)
    n_samp_points = length(times)
    normalization = logdet(covariance) + n_total_samp_points * log(2 * π)
    P_u = unit(init_ks.P)

    ks, ϵ_int, ϵ_inv, design_matrix, E_t = fit_kepler_wright_linear_step(data, times, covariance, init_ks.P, init_ks.M0, init_ks.e; data_unit=data_unit, return_extra=true)
    if hold_P
        current_x = ustrip.([ks.M0, ks.e])
    else
        current_x = ustrip.([ks.P, ks.M0, ks.e])
    end
    buffer = kep_buffer_wright(ks, remove_kepler(data, times, ks; data_unit=data_unit), -logprior_kepler(ks; use_hk=false), ϵ_int, ϵ_inv, design_matrix, E_t)
    last_x = similar(current_x)

    function calculate_common!(x::Vector{T}, last_x::Vector{T}, buffer::kep_buffer_wright) where {T<:Real}
        if x != last_x
            # copy!(last_x, x)
            last_x[:] = x
            if (0 > x[end] || x[end] > 1)  # is 0 <= e <= 1 ?
                buffer.nprior = Inf
            else
                if hold_P
                    buffer.ks, buffer.ϵ_int, buffer.ϵ_inv, buffer.design_matrix, buffer.E_t = fit_kepler_wright_linear_step(data, times, covariance, init_ks.P, x[1], x[2]; data_unit=data_unit, return_extra=true)
                else
                    buffer.ks, buffer.ϵ_int, buffer.ϵ_inv, buffer.design_matrix, buffer.E_t = fit_kepler_wright_linear_step(data, times, covariance, x[1] * P_u, x[2], x[3]; data_unit=data_unit, return_extra=true)
                end
                buffer.nprior = -logprior_kepler(buffer.ks; use_hk=false)
            end
            buffer.nprior == Inf ? buffer.rm_kep = zeros(length(buffer.rm_kep)) : buffer.rm_kep = remove_kepler(data, times, buffer.ks; data_unit=data_unit)
        end
        # println("buffer: ", kep_parms_str(buffer.ks))
        # println(buffer.rm_kep[1:40:end])
    end

    function f(x::Vector{T}, buffer::kep_buffer_wright, last_x::Vector{T}) where {T<:Real}
        calculate_common!(x, last_x, buffer)
        if buffer.nprior == Inf
            return buffer.nprior
        else
            return GLOM.nlogL(covariance, buffer.rm_kep, nlogL_normalization=normalization) + (include_priors * buffer.nprior)
        end
    end
    f(x) = f(x, buffer, last_x)

    function g!(G::Vector{T}, x::Vector{T}, buffer::kep_buffer_wright, last_x::Vector{T}) where {T<:Real}
        calculate_common!(x, last_x, buffer)
        if buffer.nprior == Inf
            G[:] .= 0
        else
            ∇nlogL_kep!(G, data, times, covariance, buffer; data_unit=data_unit, hold_P=hold_P, include_priors=include_priors)
        end
    end
    g!(G, x) = g!(G, x, buffer, last_x)

    if avoid_saddle
        attempts = 0
        in_saddle = true
        while attempts < 10 && in_saddle
            attempts += 1
            if attempts > 1;
                if print_stuff; println("found saddle point. starting attempt $attempts with a perturbation") end
                if !hold_P; current_x[1] *= centered_rand(; scale=2e-2, center=1) end
                current_x[end-1] = mod2pi(current_x[end-1] + centered_rand(; scale=4e-1))
                current_x[end] = 0.05 * rand()
            end
            # println(current_x)
            result = optimize(f, g!, current_x, LBFGS(alphaguess=LineSearches.InitialStatic(alpha=fit_alpha)), Optim.Options(callback=optim_cb_local)) # 27s
            current_x = copy(result.minimizer)
            ks = fit_kepler_wright_linear_step(data, times, covariance, buffer.ks.P, buffer.ks.M0, buffer.ks.e; data_unit=data_unit)
            if print_stuff; println("fit attempt $attempts: "kep_parms_str(ks)) end
            # println(∇∇nlogL_kep(data, times, covariance, ks; data_unit=data_unit))
            new_det = det(∇∇nlogL_kep(data, times, covariance, kep_signal(ks.K, ks.P, ks.M0, ks.e, ks.ω, ks.γ); data_unit=data_unit))
            if print_stuff; println("determinant: ", new_det) end
            in_saddle = new_det <= 0
            if !in_saddle; return ks end
        end
        @error "no non-saddle point soln found"
    else
        try
            result = optimize(f, g!, current_x, LBFGS(alphaguess=LineSearches.InitialStatic(alpha=fit_alpha)), Optim.Options(callback=optim_cb_local)) # 27s
            return fit_kepler_wright_linear_step(data, times, covariance, buffer.ks.P, buffer.ks.M0, buffer.ks.e; data_unit=data_unit)
        catch
            # print("Kepler fit failed")
            return nothing
        end
    end

end
fit_kepler(data, times, covariance, init_ks::kep_signal_wright; kwargs...) =
    fit_kepler_wright(data, times, covariance, init_ks; kwargs...)


"Convert the solar phase information from SOAP 2.0 into days"
function convert_SOAP_phases(new_unit::Unitful.FreeUnits, phase::Real; P_rot::Unitful.Time=25.05u"d")
    # default P_rot is the solar rotation period used by SOAP 2.0 in days
    assert_positive(P_rot)
    return uconvert(new_unit, phase * P_rot)
end


function remove_kepler(
    data::Vector{<:Real},
    times::Vector{T2} where T2<:Unitful.Time,
    ks::KeplerSignal;
    data_unit::Unitful.Velocity=1u"m/s",
    d::Vector{<:Integer}=zeros(Int,n_kep_parms))

    validate_kepler_dorder(d)
    n_out = Int(length(data) / length(times))
    if findfirst(!iszero, d) == nothing
        y = copy(data)
        if ustrip(ks.K) != 0; y[1:n_out:end] -= uconvert.(unit(data_unit), ks.(times)) ./ data_unit end
    else  # this uses h and k not e and ω
        @assert typeof(ks) == kep_signal
        y = zeros(length(data))
        kep_deriv_simple(t) = kep_deriv(ks, t, d)
        y[1:n_out:end] -= ustrip.(kep_deriv_simple.(times)) ./ convert_and_strip_units(unit(ks.K), data_unit)
    end
    return y

end
remove_kepler(
    data::Vector{<:Unitful.Velocity},
    times::Vector{T2} where T2<:Unitful.Time,
    ks::KeplerSignal;
    kwargs...) =
    remove_kepler(ustrip.(data), times, ks; data_unit=1.0*unit(data[1]), kwargs...)
remove_kepler(
    glo_rv::GLO_RV,
    ks::KeplerSignal;
    kwargs...) =
    remove_kepler(glo_rv.GLO.y_obs, glo_rv.time, ks; data_unit=glo_rv.rv_factor, kwargs...)


fit_kepler(
    glo_rv::GLO_RV,
    covariance::Union{Cholesky,Diagonal},
    ks::KeplerSignal;
    kwargs...) where T<:Real =
    fit_kepler(glo_rv.GLO.y_obs, glo_rv.time, covariance, ks; data_unit=glo_rv.rv_factor, kwargs...)

kep_signal(ks::KeplerSignal) = kep_signal(ks.K, ks.P, ks.M0, ks.e, ks.ω, ks.γ)

kep_parms_str(ks::KeplerSignal) =
    "K: $(round(convert_and_strip_units(u"m/s", ks.K), digits=2))" * "m/s" * "  P: $(round(convert_and_strip_units(u"d", ks.P), digits=2))" * "d" * "  M0: $(round(ks.M0,digits=2))  e: $(round(ks.e,digits=2))  ω: $(round(ks.ω,digits=2)) γ: $(round(convert_and_strip_units(u"m/s", ks.γ), digits=2))" * "m/s"
kep_parms_str_short(ks::KeplerSignal) =
    "K: $(round(convert_and_strip_units(u"m/s", ks.K), digits=2))" * "m/s" * "  P: $(round(convert_and_strip_units(u"d", ks.P), digits=2))" * "d" * "  e: $(round(ks.e,digits=2))"


function ∇∇nlogL_kep(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky,Diagonal},
    ks::kep_signal;
    data_unit::Unitful.Velocity=1u"m/s",
    include_priors::Bool=false) where T<:Real

    y = remove_kepler(data, times, ks; data_unit=data_unit)
    α = covariance \ y
    H = zeros(n_kep_parms, n_kep_parms)
    for i in 1:n_kep_parms
        for j in 1:n_kep_parms
            if i <= j
                d = zeros(Int, n_kep_parms)
                d[j] += 1
                y2 = remove_kepler(data, times, ks; data_unit=data_unit, d=d)
                d[i] += 1
                y12 = remove_kepler(data, times, ks; data_unit=data_unit, d=d)
                d[j] -= 1
                y1 = remove_kepler(data, times, ks; data_unit=data_unit, d=d)
                H[i, j] = GLOM.d2nlogLdθ(y2, y12, α, covariance \ y1)
            end
        end
    end

    if include_priors
        H[:,:] -= logprior_kepler_tot(ks; d_tot=2, use_hk=true)
    end

    return Symmetric(H)
end


function ∇∇nlogL_GLOM_and_planet!(
    workspace::GLOM.nlogL_matrix_workspace,
    glo_rv::GLO_RV,
    total_hyperparameters::Vector{<:Real},
    ks::kep_signal;
    include_kepler_priors::Bool=false)

    GLOM.calculate_shared_∇nlogL_matrices!(workspace, glo_rv.GLO, total_hyperparameters)

    non_zero_inds = copy(glo_rv.GLO.non_zero_hyper_inds)
    n_hyper = length(non_zero_inds)
    full_H = zeros(n_kep_parms + n_hyper, n_kep_parms + n_hyper)
    full_H[1:n_hyper, 1:n_hyper] = GLOM.∇∇nlogL_GLOM(
        glo_rv.GLO, total_hyperparameters; Σ_obs=workspace.Σ_obs, y_obs=remove_kepler(glo_rv, ks))

    full_H[n_hyper+1:end,n_hyper+1:end] = ∇∇nlogL_kep(glo_rv.GLO.y_obs, glo_rv.time, workspace.Σ_obs, ks; data_unit=glo_rv.rv_factor, include_priors=include_kepler_priors)

    # TODO allow y and α to be passed to ∇∇nlogL_kep
    y = remove_kepler(glo_rv, ks)
    α = workspace.Σ_obs \ y
    for (i, nzind1) in enumerate(non_zero_inds)
        for j in 1:n_kep_parms
            d = zeros(Int, n_kep_parms)
            d[j] += 1
            y1 = remove_kepler(glo_rv, ks; d=d)
            full_H[i, j + n_hyper] = GLOM.d2nlogLdθ(y, y1, α, workspace.Σ_obs \ y1, workspace.βs[i])
        end
    end

    return Symmetric(full_H)
end
∇∇nlogL_GLOM_and_planet(glo_rv, total_hyperparameters, ks; kwargs...) =
    ∇∇nlogL_GLOM_and_planet!(
        GLOM.nlogL_matrix_workspace(glo_rv.GLO, total_hyperparameters),
        glo_rv, total_hyperparameters, ks; kwargs...)

function validate_kepler_dorder(d::Vector{<:Integer})
	@assert sum(d) < 3
	@assert minimum(d) == 0
	@assert length(d) == n_kep_parms
end

function validate_kepler_wright_dorder(d::Vector{<:Integer})
	@assert sum(d) < 2
	@assert minimum(d) == 0
	@assert length(d) == 3
end

function ks_from_vec(vector::Vector{<:Real}, K_u::Unitful.VelocityFreeUnits, P_u::Unitful.TimeFreeUnits, γ_u::Unitful.VelocityFreeUnits; use_hk::Bool=false)
    @assert length(vector) == n_kep_parms
    return kep_signal(vector[1] * K_u, vector[2] * P_u, vector[3], vector[4], vector[5], vector[6] * γ_u; use_hk=use_hk)
end
