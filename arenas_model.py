## Main script to run the Arenas epidemic model of [1,2] aggregating spatial and age couplings.

import numpy as np
# from helper_functions import *

def runtest():
    '''
    Runs the test.py.
    '''
    from test import main
    main()

# Define one markov step
def markov_step(x, M):
    '''
    Computes one step in the markovian model. We assume that x_{t+1} = M x_t, where M = M(x_t, t).

    Inputs:
    `x`: state variables (S,E,A,I,H,R,D)
    `M`: list of non-zero elements of the transition matrix M
    '''
    # state variables
    S,E,A,I,H,R,D = x
    # interaction terms
    M_SS, M_ES, M_EE, M_AE, M_AA, M_IA, M_II, M_HI, M_HH, M_RI, M_RH, M_RR, M_DH, M_DD = M

    return np.array([
        M_SS * S,
        M_ES * S + M_EE * E,
        M_AE * E + M_AA * A,
        M_IA * A + M_II * I,
        M_HI * I + M_HH * H,
        M_RI * I + M_RH * H + M_RR * R,
        M_DH * H + M_DD * D
    ])

def iterate_model(x0, T, params):
    '''
    Solves the markovian model for `T` time steps (days) for the initial conditions `x0` and the set of parameters `params` and `ext_params`.

    Inputs:
    `x0`: list with the initial compartiment densities (S0, E0, A0, I0, H0, R0, D0)
    `params`': list of parameters in the same order than in Arenas report [2]: (β, kg, ηg, αg, ν, μg, γg, ωg, ψg, χg, n_ig, σ, κ0, ϕ, tc, tf)

    Output:
    `flow`: 7-dimensional time series. Each dimension corresponds to S(t), E(t), A(t), I(t), H(t), R(t), D(t) respectively.
    '''

    ## READING ##

    # Read parameters (1-D treatment. In the general treatment, suffix `g` indicates an NG-sized vector)
    β = params[0]
    k = params[1]
    η = params[2]
    α = params[3]
    ν = params[4]
    μ = params[5]
    γ = params[6]
    ω = params[7]
    ψ = params[8]
    χ = params[9]
    n = params[10] # population
    # containtment params
    σ  = params[11]
    κ0 = params[12]
    ϕ  = params[13]
    tc = params[14]
    tf = params[15]

    # Compute transmission probability
    Π_t = Π_1D( x0[2]+ ν*x0[3], β, k)

    # Compute interaction terms
    M_SS = 1 - Π_t
    M_ES = Π_t
    M_EE = 1 - η
    M_AE = η
    M_AA = 1 - α
    M_IA = α
    M_II = 1 - μ
    M_HI = μ * γ
    M_HH = ω * (1 - ψ) + (1 - ω)*(1 - χ)
    M_RI = μ * (1 - γ)
    M_RH = (1 - ω) * χ
    M_RR = 1
    M_DH = ω * ψ
    M_DD = 1

    ## Non-zero interactions for transition-like matrix
    M = [M_SS, M_ES, M_EE, M_AE, M_AA, M_IA, M_II, M_HI, M_HH, M_RI, M_RH, M_RR, M_DH, M_DD]

    ## PREALLOCATION
    flow = np.zeros( [T+1, *x0.shape] )
    flow[0,:] = x0

    x_old = x0

    ## MODEL DYNAMICS
    for t in range(T):

        # Take markov step
        x_new = markov_step(x_old, M)
        # Update flow vector
        flow[t+1,:] = x_new

        # Containtment
        if t+1 == tc:
            # Lower avg. number of contacts as a function of containtment
            k = (1-κ0)*k + κ0*(σ-1)

            # 1-D treatment
            Π_t = Π_1D( x_new[2]+ν*x_new[3], β, k )

            ## Contained people (susceptible + recovered)
            C_tc = ( x_new[0]+x_new[5] )**σ

            # update dynamic interaction terms
            M[0] = (1 - Π_t)*(1 - (1 - ϕ)*κ0*C_tc)
            M[1] = Π_t*(1 - (1 - ϕ)*κ0*C_tc)
        else:

            # Update probability of transmission
            Π_t = Π_1D( x_new[2]+ν*x_new[3], β, k )

            # update dynamic interaction terms
            M[0] = (1 - Π_t)
            M[1] = Π_t

        # end of containtment
        if t+1 == tc+tf:
            # mid-agers (1-D treatment)
            k = ( k - κ0*(σ-1) ) / (1 - κ0)

            # 1-D treatment
            Π_t = Π_1D( x_new[2]+ν*x_new[3], β, k )

            # update dynamic interaction terms
            M[0] = (1 - Π_t)*(1 + (1 - ϕ)*κ0*C_tc)
            M[1] = Π_t*(1 + (1 - ϕ)*κ0*C_tc)

        x_old = x_new

    return flow

## Helper functions
# Probability of infection for 1D treatment of the model
def Π_1D(ρ, β, k):
    '''
    Returns the probability of infection per patch per age strata per day considering the effective mobility patterns.
    This function is designed for the aggregate model, where NP = NG = 1.
    The output is a scalar.
    '''
    return 1 - (1 - β)**(k*ρ)
