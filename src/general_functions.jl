# these are all general purpose functions that aren't specifically related to
# radial velocity or GP calculations
using LinearAlgebra
using Random
using Statistics
using Unitful

"""
Nyquist frequency is half of the sampling rate of a discrete signal processing system
(https://en.wikipedia.org/wiki/Nyquist_frequency)
divide by another factor of 4 for uneven spacing
"""
nyquist_frequency(time_span::Union{Real,Quantity}, n_meas::Integer; nyquist_factor::Real=1) = n_meas / time_span / 2 * nyquist_factor
function nyquist_frequency(times::Vector{T}; nyquist_factor::Real=1) where {T<:Union{Real,Quantity}}
    time_span = times[end] - times[1]
    return nyquist_frequency(time_span, length(times), nyquist_factor=nyquist_factor)
end
uneven_nyquist_frequency(times; nyquist_factor=5) = nyquist_frequency(times; nyquist_factor=nyquist_factor)


"""
shamelessly crimped from JuliaAstro.jl
used to calculate range of frequencies to look at in a periodogram
"""
function autofrequency(times::Vector{T} where {T<:Union{Real,Quantity}};
                       samples_per_peak::Integer=5,
                       nyquist_factor::Integer=5,
                       minimum_frequency::Real=NaN,
                       maximum_frequency::Real=NaN)
    time_span = maximum(times) - minimum(times)
    δf = inv(samples_per_peak * time_span)
    f_min = isfinite(minimum_frequency) ? minimum_frequency : (δf / 2)
    if isfinite(maximum_frequency)
        return f_min:δf:maximum_frequency
    else
        return f_min:δf:nyquist_frequency(time_span, length(times); nyquist_factor=nyquist_factor)
    end
end


"assert all passed variables are positive"
function assert_positive(vars...)
    for i in vars
        @assert all(ustrip.(i) .> 0) "passed a negative/0 variable that needs to be positive"
    end
end

"""
Solve a linear system of equations (optionally with variance values at each point or covariance array)
see (https://en.wikipedia.org/wiki/Generalized_least_squares#Method_outline)
"""
function general_lst_sq(
    dm::Matrix{T},
    data::Vector;
    Σ::Union{Cholesky,Diagonal}=Diagonal(ones(length(data))),
    return_ϵ_inv::Bool=false) where {T<:Real}
    @assert ndims(Σ) < 3 "the Σ variable needs to be a 1D or 2D array"

    if return_ϵ_inv
        ϵ_int = Σ \ dm
        ϵ_inv = dm' * ϵ_int
        return ϵ_inv \ (dm' * (Σ \ data)), ϵ_int, ϵ_inv
    else
        return (dm' * (Σ \ dm)) \ (dm' * (Σ \ data))
    end
    # end
end


"Return an amount of indices of local maxima of a data array"
function find_modes(data::Vector{T}; amount::Integer=3) where {T<:Real}

    # creating index list for inds at modes
    mode_inds = [i for i in 2:(length(data)-1) if (data[i]>=data[i-1]) && (data[i]>=data[i+1])]
    if data[1] > data[2]; prepend!(mode_inds, 1) end
    if data[end] > data[end-1]; append!(mode_inds, length(data)) end

    # return highest mode indices
    return mode_inds[partialsortperm(-data[mode_inds], 1:amount)]

end

centered_rand(; rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...) = centered_rand(rng; kwargs...)
centered_rand(rng::AbstractRNG; center::Real=0, scale::Real=1) = (scale * (rand(rng) - 0.5)) + center
centered_rand(d::Integer; rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...) = centered_rand(rng, d; kwargs...)
centered_rand(rng::AbstractRNG, d; center::Real=0, scale::Real=1) = (scale .* (rand(rng, d) .- 0.5)) .+ center

function remove_mean!(vec::AbstractVector)
    vec .-= mean(vec)
end
