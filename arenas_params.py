## Parámetros utilizados en el modelo epidemiológico de COVID-19 de Arenas et al. en [1] y [2]
# [1]: Arenas, Alex, et al. "A mathematical model for the spatiotemporal epidemic spreading of COVID19." medRxiv (2020).
# [2]: Arenas, Alex, et al. "Derivation of the effective reproduction number R for COVID-19 in relation to mobility restrictions and confinement." medRxiv (2020).

import numpy as np

# Este script define los parámetros escalares necesarios para el modelo sin acoplamiento que se usarán mientras no haya datos disponibles.

# Parámetros en [2]
R_0 = 2.3 # mentioned in the abstract of [1]

β  = 0.06 # infectivity of the desease
η  = 1/2.34 # η^-1 + α^-1 = 1/5.2 # exposed latent rate
αg = 1/2.86 # np.array( [1/8.86, 1/2.86, 1/2.86] ) # asymptomatic infectious rate by age group
ωg = 0.42 # fatality rate of ICU patients
ψg = 1/7  # death rate
χg = 1/20 # ICU discharge rate (1/10 in [2])
ξ  = 0.01 # km^-2. density weight factor
pg = 1 # np.array( [0,1,0] ) # mobility factor by age strata
μg = 1/7 # np.array( [1/1,1/7,1/7] ) # escape (from Infected) rate by age group
# Cambiar:
γg = 0.05 # np.array( [0.002, 0.05, 0.36] ) # fraction of cases requiring ICU by age group
# Cambiar:
kg = 13.3 # np.array( [11.8, 13.3, 6.6] ) # average contacts per day by age group
Cgh  = 1 # np.array( [[0.5980, 0.3849, 0.0171],
            #   [0.2440, 0.7210, 0.0350],
            #   [0.1919, 0.5705, 0.2376]] ) # contacts-by-age-strata matrix
# Cambiar:
σ  = 2.5 # average household size in Spain
ν = 0.6 # isolation factor
κ0 = 0.7 # confinement factor: adjustable for containment (κ0 = 0.7 mentioned in [2])
ϕ = 0.2 # permeability factor: adjustable for containment
tc = np.inf # days from t0 for containtment
tf = np.inf # days from tc to finalize containtment

# Parámetros diferentes en [1]
# χg = 1/10
# μg = np.array([1/1,1/3.2,1/3.2]) # escape (from Infected) rate by age group
# αg = np.array([1/5.06, 1/2.86, 1/2.86]) # asymptomatic infectious rate by age group
