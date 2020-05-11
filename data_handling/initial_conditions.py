# -*- coding: utf-8 -*-
'''
    Este módulo tiene todas las funciones pertinentes para obtener y/o calcular las condiciones iniciales del modelo a partir de datos y parámetros.

    Convención de condiciones iniciales:
    x = S, E, A, I, H, Rᴵ, Rᴴ, D
'''

# Standard libraries
import pandas as pd
import numpy as np
import datetime as dt

# Minimizer library
import scipy

# Módulo de manejo de datos
from data_handling.data_processing import *
from data_handling.parameters import *

# El modelo
from arenas_model import iterate_model


def get_condiciones_iniciales(series, estado, params, t0):
    '''
    Calcula las condiciones iniciales necesarias para el modelo para `estado`.

    Inputs:
        - series: Dataframe con las series de tiempo de todos los estados
        - estado: Entidad federativa a considerar
        - params: Parámetros del modelo. Aquí se utilizan η, α, χᴵ, χᴴ
        - t0: Fecha inicial

    Outputs:
        - x0_subreporte: Condiciones iniciales tomadas de los datos. No considera el subreporte de infectados.

    '''

    # Toma la series del estado de interés
    series = get_serie_estatal(series, estado)

    # Toma la población del estado de interés
    N = get_poblacion(estado)

    # Parámetros relevantes
    η  = params[2]
    α  = params[3]
    χᴵ = params[9]
    χᴴ = params[10]

    # Variables en las que confiamos
    fallecidos_t0 = series.loc[t0, 'fallecidos_acumulados']
    hospitalizados_t0 = series.loc[t0, 'hospitalizados_acumulados']

    # Variables latentes: E,A,I,R.
    # Las variables a continuación solo dan un estimado inicial que luego ajustamos mejor
    confirmados_t0 = series.loc[t0, 'confirmados_acumulados']

    expuestos_t0 = series.loc[dias_desde_t0(t0, 1):dias_desde_t0(t0, np.round(1/α)), 'confirmados_diarios'].sum()

    asintomaticos_t0 = series.loc[dias_desde_t0(t0, np.round(1/α)):dias_desde_t0(t0, np.round(1/α + 1/η)), 'confirmados_diarios'].sum()

    removidos_t0 = series.loc[ dias_desde_t0(t0, -np.round(1/χᴵ)), 'confirmados_acumulados' ] # 'ambulatorios_acumulados'
    recuperados_ambu_t0 = 0

    recuperados_hosp_t0 = series.loc[ dias_desde_t0(t0, -np.round(1/χᴴ)), 'hospitalizados_acumulados' ]


    ## initial conditions of state_i setup ##

    Rᴴ0 = recuperados_hosp_t0 / N                   # fraction of recovered from hospital. Latent
    D0  = fallecidos_t0 / N                         # fraction of deceased
    H0  = (hospitalizados_t0)/N - Rᴴ0 - D0          # fraction of hospitalized cases.
    Rᴵ0 = (removidos_t0 / N) #- H0 - Rᴴ0 - D0        # fraction of recovered from infected. Latent
    I0  = (confirmados_t0/N) - H0 - Rᴵ0 - Rᴴ0 - D0  # fraction of confirmed infected cases. Latent-ish
    E0  = expuestos_t0 / N                          # fraction of exposed non-infectious cases. Latent
    A0  = asintomaticos_t0 / N                      # fraction of asymptomatic cases. Latent
    S0  = (1 - E0 - A0 - I0 - Rᴵ0 - Rᴴ0 - D0 - H0)  # fraction of suceptible cases

    return np.array( [S0,E0,A0,I0,H0,Rᴵ0, Rᴴ0, D0] ) # x0

### FUNCIONES DE MANEJO DE VARIABLES LATENTES ###

def multiplicador_subreporte(x0, params, m=10):
    '''
    Ajusta las condiciones iniciales y la tasa de pacientes que van a hospital (γ) al multiplicador de subreporte `m`.

    Inputs:
        - x0: Condiciones iniciales oficiales
        - params: Parámetros del modelo. La fracción de casos que requieren hospialización se ajusta al multiplicador
        - m=10: Multiplicador; cuántos veces de casos latentes hay que no son medidos?

    Output:
        - x_new: Condiciones iniciales ajustadas por el multiplicador
    '''

    # Divide γ / m
    params[6] = params[6] / m

    # Escala las condiciones iniciales por el multiplicador de casos latentes
    # E, A, I, Rᴵ
    x0_latentes = np.array(x0[ [1, 2, 3, 5] ]) * m
    # Rᴴ no se escala
    x0_latentes = np.append( x0_latentes, x0[6] )

    # Copia condiciones iniciales
    x0_copy = 1*x0

    # Cambia las condiciones iniciales de acuerdo a los multiplicadores
    x0_copy[ [1, 2, 3, 5, 6] ] = x0_latentes
    x0_copy[0] = 1 - x0_copy[1:].sum()

    return x0_copy

def correccion_x0_latentes(series, estado, x0, params, t0, t_fit=20, method='nelder-mead' ):
    '''
    Modifica las variables latentes (E0, A0, I0, Rᴴ0. Rᴵ0) que mejor se ajusten a los datos dados los parámetros `params` del modelo.
    Dicho ajuste se obtiene minimizando el error cuadrático medio entre el modelo y los datos.

    Inputs:
        - series: Dataframe con las series de tiempo de todos los estados
        - estado: Entidad federativa a considerar
        - x0: condiciones iniciales. Se ajustarán las variables latentes: E0, A0, I0, Rᴴ0. Rᴵ0
        - params: Parámetros del modelo
        - t0: fecha inicial para la simulación
        - t_fit=20: días de ajuste desde t0
        - method='nelder-mead': método de minimización

    Output:
        - x0_new: Nuevas condiciones iniciales para correr el modelo ajustado. Esto da una estimación burda de las variables latentes reales.

    Nota: Las condiciones iniciales `x0` ya deben haber sido procesadas por el multiplicador de subreporte.
    '''

    # Variables latentes:  E, A, I, Rᴵ, Rᴴ
    x0_latentes = x0[ [1, 2, 3, 5, 6] ]

    # Obtiene fecha final en formato fecha
    tf = dias_desde_t0(t0, t_fit)
    # Arma las series de tiempo de datos para ajuste
    data = get_serie_estatal(series, estado).loc[t0:tf, ['hospitalizados_acumulados','fallecidos_acumulados']]

    # Minimización de función objetivo (RMSE)
    opt = scipy.optimize.minimize(fun= lambda x: RMSE(data, params, x),
                               x0=x0_latentes, method=method, options={'maxiter':500}  )

    print('Error: {}\nNúmero de iteraciones: {}'.format(opt.fun, opt.nit) )

    # Variables latentes resultado de la minimización
    x0_latentes_new = opt.x

    ## Nuevas condiciones iniciales

    x0_new = 1*x0
    x0_new[ [1, 2, 3, 5, 6] ] = x0_latentes_new
    # Compensación de hospitalizados con el nuevo valor de R0ᴴ
    x0_new[4] = x0[4] + x0[6] - x0_new[6]
    # Compensación de población susceptible
    x0_new[0] = 1 - x0_new[1:].sum()

    return x0_new

### FUNCIONES DE FIT ###

def RMSE(data, params, x0_latentes):
    '''
    Calcula el error cuadrático medio (MSE) entre los datos de fallecidos y hospitalizados y el modelo usando dichos datos como condiciones iniciales.

    Inputs:
        - data: Pandas DataFrame con datos de hospitalizados y fallecidos acumulados en el intervalo de tiempo de interés para hacer el ajuste.
        - params: Parámetros del modelo
        - x0_latentes: Condiciones iniciales que no se pueden saber directamente de los datos: E,A,I,Rᴴ,Rᴵ

    Output:
        - rmse: raíz del error cuadrático medio entre los datos y las simulaciones del modelo.
    '''

    # Población Total
    N = params[11]

    # Condiciones iniciales latentes
    E0, A0, I0, Rᴵ0, Rᴴ0 = x0_latentes

    # Datos duros
    # Fallecidos
    D0 = data['fallecidos_acumulados'][0]/N
    # Hospitalizados
    H0 = data['hospitalizados_acumulados'][0]/N - D0 - Rᴴ0


    S0 = (1 - E0 - A0 - I0 - H0 - Rᴵ0 - Rᴴ0 - D0)
    x0 = np.array([S0, E0, A0, I0, H0, Rᴵ0, Rᴴ0, D0])

    # Dias de simulación: Tantos como haya datos
    T = len(data) - 1
    # Corre el modelo
    flow = iterate_model(x0, T, params)
    # Convierte densidades en número de casos
    flow *= N

    # Error cuadrático medio
    mse = 0
    # Por fallecidos
    mse += ( np.square( data['fallecidos_acumulados']      - flow[:,7] ).mean() )
    # Por hospitalizados acumulados
    mse += ( np.square( (data['hospitalizados_acumulados'] - data['fallecidos_acumulados'])
                    - (flow[:,4] + flow[:,6] ) ).mean() )

    # Condiciones de frontera: todas las condiciones iniciales deben ser positivas
    if any(x_i < 0 for x_i in x0):
        return 1e100
    else:
        # rmse = sqrt(mse)
        return np.sqrt(mse)
