# these are all general purpose functions that aren't specifically related to
# radial velocity or GP calculations
using LinearAlgebra
using Distributed
using Unitful
using Random

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


"""
Automatically adds as many workers as there are CPU threads minus 2 if none are
active and no number of procs to add is given
"""
function auto_addprocs(;add_procs::Integer=0)
    # only add as any processors as possible if we are on a consumer chip
    if (add_procs==0) && (nworkers()==1) && (length(Sys.cpu_info())<=16)
        add_procs = length(Sys.cpu_info()) - 2
    end
    addprocs(add_procs)
    println("added $add_procs workers")
end


"""
Automatically adds as many workers as there are CPU threads minus 2 if none are
active and no number of procs to add is given
Also includes all basic functions for analysis
"""
function prep_parallel(; add_procs::Integer=0)
    auto_addprocs(;add_procs=add_procs)
    @everywhere include("src/all_functions.jl")
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
    Σ::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}}=ones(1),
    return_ϵ_inv::Bool=false) where {T<:Real}
    @assert ndims(Σ) < 3 "the Σ variable needs to be a 1D or 2D array"

    # if Σ == ones(1)
    #     return dm \ data
    # else
    if ndims(Σ) == 1
        Σ = Diagonal(Σ)
    else
        Σ = GLOM.ridge_chol(Σ)
    end
    if return_ϵ_inv
        ϵ_int = Σ \ dm
        ϵ_inv = dm' * ϵ_int
        return ϵ_inv \ (dm' * (Σ \ data)), ϵ_int, ϵ_inv
    else
        return (dm' * (Σ \ dm)) \ (dm' * (Σ \ data))
    end
    # end
end

"""
For distributed computing. Send a variable to a worker
stolen shamelessly from ParallelDataTransfer.jl
e.g.
sendto([1, 2], x=100, y=rand(2, 3))
z = randn(10, 10); sendto(workers(), z=z)
"""
function sendto(workers::Union{T,Vector{T}}; args...) where {T<:Integer}
    for worker in workers
        for (var_name, var_value) in args
            @spawnat(worker, Core.eval(Main, Expr(:(=), var_name, var_value)))
        end
    end
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


centered_rand(; rng::AbstractRNG=Random.GLOBAL_RNG, center::Real=0, scale::Real=1) = centered_rand(rng; center=center, scale=scale)
centered_rand(rng::AbstractRNG; center::Real=0, scale::Real=1) = (scale * (rand(rng) - 0.5)) + center
centered_rand(d::Integer; rng::AbstractRNG=Random.GLOBAL_RNG, center::Real=0, scale::Real=1) = centered_rand(rng, d; center=center, scale=scale)
centered_rand(rng::AbstractRNG, d; center::Real=0, scale::Real=1) = (scale .* (rand(rng, d) .- 0.5)) .+ center
