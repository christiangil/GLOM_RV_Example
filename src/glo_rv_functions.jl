# glo_functions.jl
using Dates
using Unitful
using UnitfulAstro
import GPLinearODEMaker

"""
A structure that holds all of the relevant information for constructing the
model used in the Jones et al. 2017+ paper (https://arxiv.org/pdf/1711.01318.pdf).
"""
struct GLO_RV
	GLO::GPLinearODEMaker.GLO  # kernel function
	time::Vector{T} where T<:Unitful.Time
	rv_factor::Unitful.Velocity

	function GLO_RV(
		glo::GPLinearODEMaker.GLO,
		time_unit::Unitful.Time,
		rv_factor::Unitful.Velocity)
		return new(glo, glo.x_obs .* time_unit, rv_factor)
	end
end

get_rv(glo::GLO_RV) = glo.GLO.y_obs[1:glo.GLO.n_out:end] .* glo.rv_factor
get_rv_noise(glo::GLO_RV) = glo.GLO.noise[1:glo.GLO.n_out:end] .* glo.rv_factor
