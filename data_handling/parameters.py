#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Este módulo tiene todas las funciones pertinentes para obtener y/o calcular los diferentes parámetros del modelo a partir de datos y/o parámetros del artículo de Arenas.

    Convención del vector de parámetros siguiendo la Supplementary Table 1 de [1]:

    # Parámetros usuales
    β = params[0]
    k = params[1]
    η = params[2]
    α = params[3]
    ν = params[4]
    μ = params[5]
    γ = params[6]
    ω = params[7]
    ψ = params[8]
    χᴵ = params[9]
    χᴴ = params[10]

    # Población
    N = params[11]

    Paramáteros de contención
    σ  = params[12]
    κ0 = params[13]
    ϕ  = params[14]
    tc = params[15]
    tf = params[16]

    [1]: Arenas, Alex, et al. "Derivation of the effective reproduction number R for COVID-19 in relation to mobility restrictions and confinement." medRxiv (2020).
'''

# Standard libraries
import pandas as pd
import numpy as np
import datetime as dt
import math

# Libraries for fitting
from sklearn.linear_model import LinearRegression
import numpy as np
from numpy.linalg import eigvals

# Módulo de manejo de datos
from data_handling.data_processing import *

# Módulos específicos del modelo
import arenas_params as ap


def get_params_arenas():
    '''
    Da el vector de parámetros del artículo de Arenas et al.
    A este vector aún hay que agregarle la población (params[11]) y cambiar los parámetros calculados por los datos.

    Outputs:
        - params: Vector de parámteros del modelo según Arenas et al.

    Nota: Los parámetros están en el mismo orden que la Supplementary Table 1 en [1].

    [1]: Arenas, Alex, et al. "Derivation of the effective reproduction number R for COVID-19 in relation to mobility restrictions and confinement." medRxiv (2020).
    '''

    # Asumimos import arenas_params as ap
    params = [
        ap.β,  # 0 : β
        ap.kg, # 1 : k
        ap.η,  # 2 : η
        ap.αg, # 3 : α
        ap.ν,  # 4 : ν
        ap.μg, # 5 : μ
        ap.γg, # 6 : γ
        ap.ωg, # 7 : ω
        ap.ψg, # 8 : ψ
        ap.χg, # 9 : χᴵ
        ap.χg, # 10: χᴴ
        0,     # 11: N
        ap.σ,  # 12: σ
        ap.κ0, # 13: κ0
        ap.ϕ,  # 14: ϕ
        ap.tc, # 15: días desde t0 para confinamiento
        ap.tf, # 16: días desde confinamiento para reactivación
         ]

    return params

# Parámetros a partir de datos
def get_parametros(series, estado='Nacional'):
    '''
    Calcula los parámetros relevantes del modelo para `estado` que no vienen de los parámetros de Arenas.
    En particular, se calcula
        - χ: Tasa de recuperación
        - γ: Fracción de casos confirmados que son hospitalizados
        - ω: Fracción de de casos hospitalizados que fallecen
        - σ: Promedio de ocupantes en viviendas particulares habitadas

    Inputs:
        - series: Dataframe con las series de tiempo de todos los estados
        - estado=Nacional: Entidad federativa a considerar

    Outputs:
        - γ, ω, χᴵ, σ

    Nota: Si `estado` = 'Nacional', calcula los parámetros a nivel nacional. Esto se recomienda para estados con menor volumen de datos.
    '''

    if estado == 'Nacional':
        series = get_serie_nacional( series )
    else:
        series = get_serie_estatal( series, estado )

    # fracción de casos que van a hospital
    γ = series['hospitalizados_acumulados'][-1] / series['confirmados_acumulados'][-1]

    # fracción de hospitalizados que mueren (fallecidos_por_hospitalizacion_acumulados)
    ω = series['fallecidos_acumulados'][-1] / series['hospitalizados_acumulados'][-1]

    # tamaño de habitantes por casa promedio [1]
    # [1]: https://www.inegi.org.mx/temas/vivienda/
    σ = 3.7

    # Tasa de recuperación usada por el gobierno: 14 días [2]
    # [2]: https://twitter.com/HLGatell/status/1258189863326830592/photo/1
    χᴵ = 1/7 # 1/14

    return γ, ω, χᴵ, σ

def get_poblacion(estado):
    '''
    Población (según Wikipedia) para `estado`.
    '''

    path = './data/poblaciones_y_superficies_por_estado.csv'
    return pd.read_csv( path, index_col='ENTIDAD' )['POBLACIONES'][estado]

### FUNCIONES DE FIT ###

## TO-DO: Recognize if t0_fit < t_JNSD (tc) or not. If it is, k_optim = <k>, else k_optim = <k_c>
def get_fit_param(series, estado, umbral=25, t0_fit=None):
    '''
    Calcula la tasa de crecimiento de hospitalizados haciendo el ajusta a 1 semana a partir de los casos determinados por `umbral`.

    Input:
        - series: Dataframe con las series de tiempo de todos los estados
        - estado: Entidad federativa a considerar
        - umbral=25: Corte de casos de hospitalización
        - t0_fit=None: Fecha inicial para determinar la tasa de crecimiento. Por default toma la fecha respecto a `umbral`
    Output:
        - tasa de crecimiento + 1

    Nota: Para el crecimiento exponencial en tiempo discreto: y_t = (λ + 1)^t * y_0
    '''

    if t0_fit == None:
        t0_fit = get_t0(series, estado, umbral=umbral)
#         t0_fit = dias_desde_t0(t0_fit, 8)
        tf_fit = dias_desde_t0(t0_fit, n_dias=8)

    series = get_serie_estatal(series, estado)

    series = series.loc[t0_fit:tf_fit,'hospitalizados_acumulados']
#     series = series.loc[t0_fit:tf_fit,'fallecidos_acumulados']
#     series = series.loc[t0_fit:tf_fit,'fallecidos_diarios']
#     series = series.loc[t0_fit:tf_fit,'hospitalizados_diarios']

    ydata = np.log(series)
    xdata = np.array( range(len(ydata)) ).reshape(-1,1)


    regressor = LinearRegression()
    regressor.fit(xdata, ydata) #training the algorithm

    # in the discrete case, the parameter is the exponent plus one.
    return regressor.coef_[0] + 1

def get_matriz_transicion_linealizada(k, params):
    '''
    Construye la matriz de transición de las ecuaciones linealizadas del modelo en función de `k`.

    Inputs:
        - k: número de contactos promedio.
        - params: parámetros del modelo
    Outputs:
        - M(k): Matriz de transición del modelo en función de `k`.

    '''

    # Parámetros del modelo necesarios para construir la matriz de transición
    β  = params[0]
    η  = params[2]
    α  = params[3]
    ν  = params[4]
    μ  = params[5]
    γ  = params[6]
    ω  = params[7]
    ψ  = params[8]
    χᴵ = params[9]
    χᴴ = params[10]

    # tasa de crecimiento
    b_k = -k * np.log( 1 - β )

    # Matriz de transición en funcion de k (componentes en la matriz: S,E,A,I,H)
    M = np.array([[1, 0,     -b_k,                -ν*b_k,                        0],
                  [0, 1-η,    b_k,                 ν*b_k,                        0],
                  [0, η,      1-α,                     0,                        0],
                  [0, 0,        α,  γ*(1-μ)+(1-γ)*(1-χᴵ),                        0],
                  [0, 0,        0,                   μ*γ, (1-ω)*(1-ψ)-(1-ω)*(1-χᴴ)]
                 ])

    return M


def get_k_optimo(tasa, params, k_min=5, k_max=15):
    '''
    Obtiene el numero de contactos promedio óptimo (<k>) respecto a la tasa de crecimiento de hospitalizados `λ`.

    Inputs:
        - tasa: Tasa de crecimiento de hospitalizados dadas por el fit.
        - params: Parámteros del modelo
        - k_min=5: Valor mínimo de rango de búsqueda
        - k_max=15: Valor máximo de rango de búsqueda

    Outputs:
        - k_optim: Número de contactos promedio ajustado a la tasa de crecimiento.
    '''

    # Optimización por barrido de valores en un rango de confianza
    k_range = np.linspace(k_min, k_max, num=100)

    # Sweep the eigvals of M for k in k_range. We are interested in the 3rd eigenvalue
    barrido_eigvals = np.zeros( len(k_range) )
    for (i,k) in enumerate(k_range):

        # Obtiene el 3er eigenvalor de la matriz de transición
        λ3 = eigvals( get_matriz_transicion_linealizada(k, params) )[2]

        # Obtiene la diferencia absoluta entre dicho eigenvalor y el exponente λ del fit.
        barrido_eigvals[i] = np.abs( λ3 - tasa )

#     print (barrido_eigvals)
    # Calcula el óptimo tomando el mínimo del barrido
    ix_min = np.argmin( barrido_eigvals )
    return k_range[ix_min]

## Funciones extra
def get_tiempo_duplicacion(tasa):
    '''
    Calcula el tiempo de duplicación de casos (en días) dada una `tasa` de crecimiento.
    '''

    print('Tiempo de duplicación: {} días'.format( np.round(math.log(2, tasa) , 1) ) )
    return math.log(2, tasa)
