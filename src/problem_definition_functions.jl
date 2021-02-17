# problem_definition_functions.jl
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
		prob_def::GPLinearODEMaker.GLO,
		time_unit::Unitful.Time,
		rv_factor::Unitful.Velocity)
		return new(prob_def, prob_def.x_obs .* time_unit, rv_factor)
	end
end

get_rv(prob_def::GLO_RV) = prob_def.GLO.y_obs[1:prob_def.GLO.n_out:end] .* prob_def.rv_factor
get_rv_noise(prob_def::GLO_RV) = prob_def.GLO.noise[1:prob_def.GLO.n_out:end] .* prob_def.rv_factor
