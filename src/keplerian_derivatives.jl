using UnitfulAstro; using Unitful

"""
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
"""
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

    if dorder ==[0, 0, 0, 0, 0, 0]
        func = γ + K*(-h*j*sinEA/e + k*jsq*cosEA/e)/qmod
    end

    if dorder ==[1, 0, 0, 0, 0, 0]
        func = (-h*j*sinEA/e + k*jsq*cosEA/e)/qmod
    end

    if dorder ==[0, 1, 0, 0, 0, 0]
        func = K*(2*h*j*t*cosEA*π_sym/(P^2*e*qmod) + 2*k*t*jsq*sinEA*π_sym/(P^2*e*qmod))/qmod + 2*K*e*t*sinEA*π_sym*(-h*j*sinEA/e + k*jsq*cosEA/e)/(P^2*qmod^3)
    end

    if dorder ==[0, 0, 1, 0, 0, 0]
        func = K*(h*j*cosEA/(e*qmod) + k*jsq*sinEA/(e*qmod))/qmod + K*e*sinEA*(-h*j*sinEA/e + k*jsq*cosEA/e)/qmod^3
    end

    if dorder ==[0, 0, 0, 1, 0, 0]
        func = K*(-j*sinEA/e + h^2*j*sinEA/e^3 - 2*h*k*cosEA/e + h^2*sinEA/(e*j) - h*k*jsq*cosEA/e^3 - h*k*jsq*sinEA^2/(e^2*qmod) - h^2*j*sinEA*cosEA/(e^2*qmod))/qmod - K*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-h*cosEA/e + h*sinEA^2/qmod)/qmod^2
    end

    if dorder ==[0, 0, 0, 0, 1, 0]
        func = K*(jsq*cosEA/e - 2*k^2*cosEA/e - k^2*jsq*cosEA/e^3 + h*j*k*sinEA/e^3 - k^2*jsq*sinEA^2/(e^2*qmod) + h*k*sinEA/(e*j) - h*j*k*sinEA*cosEA/(e^2*qmod))/qmod - K*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-k*cosEA/e + k*sinEA^2/qmod)/qmod^2
    end

    if dorder ==[0, 0, 0, 0, 0, 1]
        func = 1
    end

    if dorder ==[2, 0, 0, 0, 0, 0]
        func = 0
    end

    if dorder ==[1, 1, 0, 0, 0, 0]
        func = (2*h*j*t*cosEA*π_sym/(P^2*e*qmod) + 2*k*t*jsq*sinEA*π_sym/(P^2*e*qmod))/qmod + 2*e*t*sinEA*π_sym*(-h*j*sinEA/e + k*jsq*cosEA/e)/(P^2*qmod^3)
    end

    if dorder ==[0, 2, 0, 0, 0, 0]
        func = K*(4*k*t^2*jsq*sinEA^2*π_sym^2/(P^4*qmod^3) + 4*h*j*t^2*sinEA*π_sym^2/(P^4*e*qmod^2) - 4*k*t^2*jsq*cosEA*π_sym^2/(P^4*e*qmod^2) + 4*h*j*t^2*sinEA*cosEA*π_sym^2/(P^4*qmod^3) - 4*h*j*t*cosEA*π_sym/(P^3*e*qmod) - 4*k*t*jsq*sinEA*π_sym/(P^3*e*qmod))/qmod - 4*K*e*t^2*cosEA*π_sym^2*(-h*j*sinEA/e + k*jsq*cosEA/e)/(P^4*qmod^4) + 12*K*t^2*esq*sinEA^2*π_sym^2*(-h*j*sinEA/e + k*jsq*cosEA/e)/(P^4*qmod^5) - 4*K*e*t*sinEA*π_sym*(-h*j*sinEA/e + k*jsq*cosEA/e)/(P^3*qmod^3) + 4*K*e*t*sinEA*π_sym*(2*h*j*t*cosEA*π_sym/(P^2*e*qmod) + 2*k*t*jsq*sinEA*π_sym/(P^2*e*qmod))/(P^2*qmod^3)
    end

    if dorder ==[1, 0, 1, 0, 0, 0]
        func = (h*j*cosEA/(e*qmod) + k*jsq*sinEA/(e*qmod))/qmod + e*sinEA*(-h*j*sinEA/e + k*jsq*cosEA/e)/qmod^3
    end

    if dorder ==[0, 1, 1, 0, 0, 0]
        func = K*(2*k*t*jsq*sinEA^2*π_sym/(P^2*qmod^3) + 2*h*j*t*sinEA*π_sym/(P^2*e*qmod^2) - 2*k*t*jsq*cosEA*π_sym/(P^2*e*qmod^2) + 2*h*j*t*sinEA*cosEA*π_sym/(P^2*qmod^3))/qmod + K*e*sinEA*(2*h*j*t*cosEA*π_sym/(P^2*e*qmod) + 2*k*t*jsq*sinEA*π_sym/(P^2*e*qmod))/qmod^3 - 2*K*e*t*cosEA*π_sym*(-h*j*sinEA/e + k*jsq*cosEA/e)/(P^2*qmod^4) + 2*K*e*t*sinEA*π_sym*(h*j*cosEA/(e*qmod) + k*jsq*sinEA/(e*qmod))/(P^2*qmod^3) + 6*K*t*esq*sinEA^2*π_sym*(-h*j*sinEA/e + k*jsq*cosEA/e)/(P^2*qmod^5)
    end

    if dorder ==[0, 0, 2, 0, 0, 0]
        func = K*(k*jsq*sinEA^2/qmod^3 + h*j*sinEA/(e*qmod^2) - k*jsq*cosEA/(e*qmod^2) + h*j*sinEA*cosEA/qmod^3)/qmod - K*e*cosEA*(-h*j*sinEA/e + k*jsq*cosEA/e)/qmod^4 + 2*K*e*sinEA*(h*j*cosEA/(e*qmod) + k*jsq*sinEA/(e*qmod))/qmod^3 + 3*K*esq*sinEA^2*(-h*j*sinEA/e + k*jsq*cosEA/e)/qmod^5
    end

    if dorder ==[1, 0, 0, 1, 0, 0]
        func = (-j*sinEA/e + h^2*j*sinEA/e^3 - 2*h*k*cosEA/e + h^2*sinEA/(e*j) - h*k*jsq*cosEA/e^3 - h*k*jsq*sinEA^2/(e^2*qmod) - h^2*j*sinEA*cosEA/(e^2*qmod))/qmod - (-h*j*sinEA/e + k*jsq*cosEA/e)*(-h*cosEA/e + h*sinEA^2/qmod)/qmod^2
    end

    if dorder ==[0, 1, 0, 1, 0, 0]
        func = K*(2*j*t*cosEA*π_sym/(P^2*e*qmod) - 2*h^2*j*t*cosEA*π_sym/(P^2*e^3*qmod) + 2*h^2*j*t*cosEA^2*π_sym/(P^2*e^2*qmod^2) - 2*h^2*j*t*sinEA^2*π_sym/(P^2*e^2*qmod^2) - 4*h*k*t*sinEA*π_sym/(P^2*e*qmod) - 2*h^2*t*cosEA*π_sym/(P^2*e*j*qmod) - 2*h*k*t*jsq*sinEA*π_sym/(P^2*e^3*qmod) - 2*h*k*t*jsq*sinEA^3*π_sym/(P^2*e*qmod^3) - 2*h^2*j*t*sinEA^2*cosEA*π_sym/(P^2*e*qmod^3) + 4*h*k*t*jsq*sinEA*cosEA*π_sym/(P^2*e^2*qmod^2))/qmod - K*(-h*cosEA/e + h*sinEA^2/qmod)*(2*h*j*t*cosEA*π_sym/(P^2*e*qmod) + 2*k*t*jsq*sinEA*π_sym/(P^2*e*qmod))/qmod^2 - K*(-2*h*t*sinEA*π_sym/(P^2*e*qmod) + 2*e*h*t*sinEA^3*π_sym/(P^2*qmod^3) - 4*h*t*sinEA*cosEA*π_sym/(P^2*qmod^2))*(-h*j*sinEA/e + k*jsq*cosEA/e)/qmod^2 + 2*K*e*t*sinEA*π_sym*(-j*sinEA/e + h^2*j*sinEA/e^3 - 2*h*k*cosEA/e + h^2*sinEA/(e*j) - h*k*jsq*cosEA/e^3 - h*k*jsq*sinEA^2/(e^2*qmod) - h^2*j*sinEA*cosEA/(e^2*qmod))/(P^2*qmod^3) - 4*K*e*t*sinEA*π_sym*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-h*cosEA/e + h*sinEA^2/qmod)/(P^2*qmod^4)
    end

    if dorder ==[0, 0, 1, 1, 0, 0]
        func = K*(j*cosEA/(e*qmod) - h^2*j*cosEA/(e^3*qmod) + h^2*j*cosEA^2/(e^2*qmod^2) - h^2*j*sinEA^2/(e^2*qmod^2) - 2*h*k*sinEA/(e*qmod) - h^2*cosEA/(e*j*qmod) - h*k*jsq*sinEA/(e^3*qmod) - h*k*jsq*sinEA^3/(e*qmod^3) - h^2*j*sinEA^2*cosEA/(e*qmod^3) + 2*h*k*jsq*sinEA*cosEA/(e^2*qmod^2))/qmod - K*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-h*sinEA/(e*qmod) + e*h*sinEA^3/qmod^3 - 2*h*sinEA*cosEA/qmod^2)/qmod^2 - K*(h*j*cosEA/(e*qmod) + k*jsq*sinEA/(e*qmod))*(-h*cosEA/e + h*sinEA^2/qmod)/qmod^2 + K*e*sinEA*(-j*sinEA/e + h^2*j*sinEA/e^3 - 2*h*k*cosEA/e + h^2*sinEA/(e*j) - h*k*jsq*cosEA/e^3 - h*k*jsq*sinEA^2/(e^2*qmod) - h^2*j*sinEA*cosEA/(e^2*qmod))/qmod^3 - 2*K*e*sinEA*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-h*cosEA/e + h*sinEA^2/qmod)/qmod^4
    end

    if dorder ==[0, 0, 0, 2, 0, 0]
        func = K*(-2*k*cosEA/e - 3*h^3*j*sinEA/e^5 + 3*h*j*sinEA/e^3 + 4*h^2*k*cosEA/e^3 - 2*h^3*sinEA/(e^3*j) - k*jsq*cosEA/e^3 + 3*h*sinEA/(e*j) + h^3*sinEA/(e*j^3) + 3*h^2*k*jsq*cosEA/e^5 + h^3*j*sinEA^3/(e^3*qmod^2) + 4*h^2*k*sinEA^2/(e^2*qmod) - k*jsq*sinEA^2/(e^2*qmod) + 3*h^2*k*jsq*sinEA^2/(e^4*qmod) + 3*h^3*j*sinEA*cosEA/(e^4*qmod) - h^3*j*sinEA*cosEA^2/(e^3*qmod^2) - 3*h*j*sinEA*cosEA/(e^2*qmod) + 2*h^3*sinEA*cosEA/(e^2*j*qmod) - 2*h^2*k*jsq*sinEA^2*cosEA/(e^3*qmod^2) + h*k*jsq*sinEA^2*(-h*cosEA/e + h*sinEA^2/qmod)/(e^2*qmod^2) + h^2*j*sinEA*cosEA*(-h*cosEA/e + h*sinEA^2/qmod)/(e^2*qmod^2))/qmod + 2*K*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-h*cosEA/e + h*sinEA^2/qmod)^2/qmod^3 - K*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-cosEA/e + sinEA^2/qmod + h^2*cosEA/e^3 + h^2*sinEA^2/(e^2*qmod) - h*sinEA^2*(-h*cosEA/e + h*sinEA^2/qmod)/qmod^2 + 2*h^2*sinEA^2*cosEA/(e*qmod^2))/qmod^2 - 2*K*(-j*sinEA/e + h^2*j*sinEA/e^3 - 2*h*k*cosEA/e + h^2*sinEA/(e*j) - h*k*jsq*cosEA/e^3 - h*k*jsq*sinEA^2/(e^2*qmod) - h^2*j*sinEA*cosEA/(e^2*qmod))*(-h*cosEA/e + h*sinEA^2/qmod)/qmod^2
    end

    if dorder ==[1, 0, 0, 0, 1, 0]
        func = (jsq*cosEA/e - 2*k^2*cosEA/e - k^2*jsq*cosEA/e^3 + h*j*k*sinEA/e^3 - k^2*jsq*sinEA^2/(e^2*qmod) + h*k*sinEA/(e*j) - h*j*k*sinEA*cosEA/(e^2*qmod))/qmod - (-h*j*sinEA/e + k*jsq*cosEA/e)*(-k*cosEA/e + k*sinEA^2/qmod)/qmod^2
    end

    if dorder ==[0, 1, 0, 0, 1, 0]
        func = K*(-4*k^2*t*sinEA*π_sym/(P^2*e*qmod) + 2*t*jsq*sinEA*π_sym/(P^2*e*qmod) - 2*k^2*t*jsq*sinEA*π_sym/(P^2*e^3*qmod) - 2*k^2*t*jsq*sinEA^3*π_sym/(P^2*e*qmod^3) - 2*h*j*k*t*cosEA*π_sym/(P^2*e^3*qmod) + 2*h*j*k*t*cosEA^2*π_sym/(P^2*e^2*qmod^2) - 2*h*j*k*t*sinEA^2*π_sym/(P^2*e^2*qmod^2) + 4*k^2*t*jsq*sinEA*cosEA*π_sym/(P^2*e^2*qmod^2) - 2*h*k*t*cosEA*π_sym/(P^2*e*j*qmod) - 2*h*j*k*t*sinEA^2*cosEA*π_sym/(P^2*e*qmod^3))/qmod - K*(-k*cosEA/e + k*sinEA^2/qmod)*(2*h*j*t*cosEA*π_sym/(P^2*e*qmod) + 2*k*t*jsq*sinEA*π_sym/(P^2*e*qmod))/qmod^2 - K*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-2*k*t*sinEA*π_sym/(P^2*e*qmod) + 2*e*k*t*sinEA^3*π_sym/(P^2*qmod^3) - 4*k*t*sinEA*cosEA*π_sym/(P^2*qmod^2))/qmod^2 + 2*K*e*t*sinEA*(jsq*cosEA/e - 2*k^2*cosEA/e - k^2*jsq*cosEA/e^3 + h*j*k*sinEA/e^3 - k^2*jsq*sinEA^2/(e^2*qmod) + h*k*sinEA/(e*j) - h*j*k*sinEA*cosEA/(e^2*qmod))*π_sym/(P^2*qmod^3) - 4*K*e*t*sinEA*π_sym*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-k*cosEA/e + k*sinEA^2/qmod)/(P^2*qmod^4)
    end

    if dorder ==[0, 0, 1, 0, 1, 0]
        func = K*(jsq*sinEA/(e*qmod) - 2*k^2*sinEA/(e*qmod) - k^2*jsq*sinEA/(e^3*qmod) - k^2*jsq*sinEA^3/(e*qmod^3) - h*j*k*cosEA/(e^3*qmod) + h*j*k*cosEA^2/(e^2*qmod^2) - h*j*k*sinEA^2/(e^2*qmod^2) + 2*k^2*jsq*sinEA*cosEA/(e^2*qmod^2) - h*k*cosEA/(e*j*qmod) - h*j*k*sinEA^2*cosEA/(e*qmod^3))/qmod - K*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-k*sinEA/(e*qmod) + e*k*sinEA^3/qmod^3 - 2*k*sinEA*cosEA/qmod^2)/qmod^2 - K*(h*j*cosEA/(e*qmod) + k*jsq*sinEA/(e*qmod))*(-k*cosEA/e + k*sinEA^2/qmod)/qmod^2 + K*e*sinEA*(jsq*cosEA/e - 2*k^2*cosEA/e - k^2*jsq*cosEA/e^3 + h*j*k*sinEA/e^3 - k^2*jsq*sinEA^2/(e^2*qmod) + h*k*sinEA/(e*j) - h*j*k*sinEA*cosEA/(e^2*qmod))/qmod^3 - 2*K*e*sinEA*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-k*cosEA/e + k*sinEA^2/qmod)/qmod^4
    end

    if dorder ==[0, 0, 0, 1, 1, 0]
        func = K*(-2*h*cosEA/e - h*jsq*cosEA/e^3 + 4*h*k^2*cosEA/e^3 + j*k*sinEA/e^3 + k*sinEA/(e*j) + 3*h*k^2*jsq*cosEA/e^5 - 3*h^2*j*k*sinEA/e^5 - 2*h^2*k*sinEA/(e^3*j) - h*jsq*sinEA^2/(e^2*qmod) + 4*h*k^2*sinEA^2/(e^2*qmod) + h^2*k*sinEA/(e*j^3) + 3*h*k^2*jsq*sinEA^2/(e^4*qmod) + h^2*j*k*sinEA^3/(e^3*qmod^2) - j*k*sinEA*cosEA/(e^2*qmod) + k^2*jsq*sinEA^2*(-h*cosEA/e + h*sinEA^2/qmod)/(e^2*qmod^2) + 3*h^2*j*k*sinEA*cosEA/(e^4*qmod) - 2*h*k^2*jsq*sinEA^2*cosEA/(e^3*qmod^2) - h^2*j*k*sinEA*cosEA^2/(e^3*qmod^2) + 2*h^2*k*sinEA*cosEA/(e^2*j*qmod) + h*j*k*sinEA*cosEA*(-h*cosEA/e + h*sinEA^2/qmod)/(e^2*qmod^2))/qmod - K*(h*k*cosEA/e^3 - k*sinEA^2*(-h*cosEA/e + h*sinEA^2/qmod)/qmod^2 + h*k*sinEA^2/(e^2*qmod) + 2*h*k*sinEA^2*cosEA/(e*qmod^2))*(-h*j*sinEA/e + k*jsq*cosEA/e)/qmod^2 - K*(-j*sinEA/e + h^2*j*sinEA/e^3 - 2*h*k*cosEA/e + h^2*sinEA/(e*j) - h*k*jsq*cosEA/e^3 - h*k*jsq*sinEA^2/(e^2*qmod) - h^2*j*sinEA*cosEA/(e^2*qmod))*(-k*cosEA/e + k*sinEA^2/qmod)/qmod^2 - K*(jsq*cosEA/e - 2*k^2*cosEA/e - k^2*jsq*cosEA/e^3 + h*j*k*sinEA/e^3 - k^2*jsq*sinEA^2/(e^2*qmod) + h*k*sinEA/(e*j) - h*j*k*sinEA*cosEA/(e^2*qmod))*(-h*cosEA/e + h*sinEA^2/qmod)/qmod^2 + 2*K*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-k*cosEA/e + k*sinEA^2/qmod)*(-h*cosEA/e + h*sinEA^2/qmod)/qmod^3
    end

    if dorder ==[0, 0, 0, 0, 2, 0]
        func = K*(4*k^3*cosEA/e^3 - 6*k*cosEA/e + 3*k^3*jsq*cosEA/e^5 + h*j*sinEA/e^3 - 3*k*jsq*cosEA/e^3 + 4*k^3*sinEA^2/(e^2*qmod) + h*sinEA/(e*j) - 3*h*j*k^2*sinEA/e^5 + 3*k^3*jsq*sinEA^2/(e^4*qmod) - 2*h*k^2*sinEA/(e^3*j) - 3*k*jsq*sinEA^2/(e^2*qmod) + h*k^2*sinEA/(e*j^3) + h*j*k^2*sinEA^3/(e^3*qmod^2) - 2*k^3*jsq*sinEA^2*cosEA/(e^3*qmod^2) - h*j*sinEA*cosEA/(e^2*qmod) + k^2*jsq*sinEA^2*(-k*cosEA/e + k*sinEA^2/qmod)/(e^2*qmod^2) + 3*h*j*k^2*sinEA*cosEA/(e^4*qmod) - h*j*k^2*sinEA*cosEA^2/(e^3*qmod^2) + 2*h*k^2*sinEA*cosEA/(e^2*j*qmod) + h*j*k*sinEA*cosEA*(-k*cosEA/e + k*sinEA^2/qmod)/(e^2*qmod^2))/qmod + 2*K*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-k*cosEA/e + k*sinEA^2/qmod)^2/qmod^3 - K*(-h*j*sinEA/e + k*jsq*cosEA/e)*(-cosEA/e + sinEA^2/qmod + k^2*cosEA/e^3 + k^2*sinEA^2/(e^2*qmod) - k*sinEA^2*(-k*cosEA/e + k*sinEA^2/qmod)/qmod^2 + 2*k^2*sinEA^2*cosEA/(e*qmod^2))/qmod^2 - 2*K*(jsq*cosEA/e - 2*k^2*cosEA/e - k^2*jsq*cosEA/e^3 + h*j*k*sinEA/e^3 - k^2*jsq*sinEA^2/(e^2*qmod) + h*k*sinEA/(e*j) - h*j*k*sinEA*cosEA/(e^2*qmod))*(-k*cosEA/e + k*sinEA^2/qmod)/qmod^2
    end

    if dorder ==[1, 0, 0, 0, 0, 1]
        func = 0
    end

    if dorder ==[0, 1, 0, 0, 0, 1]
        func = 0
    end

    if dorder ==[0, 0, 1, 0, 0, 1]
        func = 0
    end

    if dorder ==[0, 0, 0, 1, 0, 1]
        func = 0
    end

    if dorder ==[0, 0, 0, 0, 1, 1]
        func = 0
    end

    if dorder ==[0, 0, 0, 0, 0, 2]
        func = 0
    end

    return float(func)

end
kep_deriv(ks::kep_signal, t::Unitful.Time, dorder::Vector{<:Integer}) =
    kep_deriv(ks.K, ks.P, ks.M0, ks.h, ks.k, ks.γ, t, dorder)