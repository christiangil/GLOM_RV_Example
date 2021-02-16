## This script was used to create keplerian_derivatives.jl

using Pkg
Pkg.activate("examples")
Pkg.instantiate()

using GLOM_RV_Example; GLOM_RV = GLOM_RV_Example
using SymEngine
using UnitfulAstro; using Unitful

@vars K, P, M0, h, k, γ, π_sym, t, cosEA, sinEA
@funs EA
e = sqrt(h^2+k^2)
EAf = EA(P, M0, h, k, t)
sinE = sin(EAf)
cosE = cos(EAf)
sinω = h/e
cosω = k/e
e_mod = 1 - e ^ 2
factor = 1 - e * cosE

function kep_EA_simplify(kep_sym::Basic)
    kep_sym = kep_sym(diff(EAf, P) => -2 * π_sym * t / (P * P * factor))
    kep_sym = kep_sym(diff(EAf, M0) => -1 / factor)
    kep_sym = kep_sym(diff(EAf, h) => sinω * sinE / factor)
    kep_sym = kep_sym(diff(EAf, k) => cosω * sinE / factor)
    return kep_sym
end
syms = [K, P, M0, h, k, γ]
n_kep = length(syms)

kep = K / factor * (e_mod * cosE * cosω - sqrt(e_mod) * sinE * sinω) + γ
dkeps = [kep_EA_simplify(diff(kep, sym)) for sym in syms]
ddkeps = [[kep_EA_simplify(diff(dkep, sym)) for dkep in dkeps] for sym in syms]

function kep_simplify!(kep::Basic, to_be_replaced::Basic, to_replace::Basic)
	kep = kep(to_be_replaced => to_replace)
	for i in 1:n_kep
		dkeps[i] = dkeps[i](to_be_replaced => to_replace)
		for j in 1:n_kep
			ddkeps[i][j] = ddkeps[i][j](to_be_replaced => to_replace)
		end
	end
	return kep
end

@vars esq, e, jsq, j, qmod

kep = kep_simplify!(kep, cosE, cosEA)
kep = kep_simplify!(kep, sinE, sinEA)
kep = kep_simplify!(kep, h^2+k^2, esq)
kep = kep_simplify!(kep, sqrt(esq), e)
kep = kep_simplify!(kep, 1 - esq, jsq)
kep = kep_simplify!(kep, sqrt(jsq), j)
kep = kep_simplify!(kep, 1 - e * cosEA, qmod)

begin
	file_loc = "src/keplerian_derivatives.jl"
	io = open(file_loc, "w")

	# begin to write the function including assertions that the amount of hyperparameters are correct
	write(io, """using UnitfulAstro; using Unitful

	\"\"\"
	kep_deriv() function created by examples/create_keplerian_derivatives.jl.
	Derivative of a Keplerian radial velocity signal using h and k instead of
	e and ω

	Parameters:
	K (Unitful.Velocity): velocity semi-amplitude
	P (Unitful.Time): period
	M0 (real): initial mean anomaly
	h (real): eccentricity * sin(argument of periastron)
	k (real): eccentricity * cos(argument of periastron)
	γ (Unitful.Velocity): velocity offset
	t (Unitful.Time): time
	dorder (vector of integers): A vector of how many partial derivatives you want
		to take with respect to each variable in the input order

	Returns:
	Unitful Quantity: The derivative specified with dorder
	\"\"\"
	function kep_deriv(
		K::Unitful.Velocity,
		P::Unitful.Time,
		M0::Real,
		h::Real,
		k::Real,
		γ::Unitful.Velocity,
		t::Unitful.Time,
		dorder::Vector{<:Integer})

		validate_kepler_dorder(dorder)

		esq = h * h + k * k
		e = sqrt(esq)
		jsq = 1 - esq
		j = sqrt(jsq)
		EAval = ecc_anomaly(t, P, M0, e)
		cosEA = cos(EAval)
		sinEA = sin(EAval)
		qmod = 1 - e * cosEA

	""")
	max_derivs = 3

	tot_inds =[[i] for i in 0:n_kep]
	append!(tot_inds, collect(Iterators.flatten([[[i,j] for i in 1:j] for j in 1:n_kep])))

	for inds in tot_inds
		dorder = zeros(Int, n_kep)
		if inds[1] != 0
			dorder[inds[1]] = 1
			func = dkeps[inds[1]]
		else
			func = kep
		end
		if length(inds) == 2
			dorder[inds[2]] += 1
			func = ddkeps[inds[1]][inds[2]]
		end

		func_str = SymEngine.toString(func)
		func_str = replace(func_str, "π_sym"=>"π")
		write(io, "    if dorder ==" * string(dorder) * "\n")
		write(io, "        func = " * func_str * "\n    end\n\n")
	end

	write(io, "    return float(func)\n\nend\n")

	write(io, """kep_deriv(ks::kep_signal, t::Unitful.Time, dorder::Vector{<:Integer}) =\n    kep_deriv(ks.K, ks.P, ks.M0, ks.h, ks.k, ks.γ, t, dorder)""")

	close(io)
end
