# -*- coding: utf-8 -*-

'''
    Este módulo tiene todas las funciones pertinentes para procesar los datos abiertos de la DGE [1].
    [1]: https://www.gob.mx/salud/documentos/datos-abiertos-152127
'''

import numpy as np
import pandas as pd
import datetime as dt

## Función principal de procesamiento de datos abiertos
def series_panel_por_estado(datos_abiertos):
    """
    Genera una tabla de panel con los siguientes datos, donde cada panel es un estado:
        - pruebas_diarias
        - confirmados_diarios
        - hospitalizados_diarios
        - fallecidos_diarios
        - pruebas_acumuladas
        - confirmados_acumulados
        - hospitalizados_acumulados
        - fallecidos_acumulados

    Input:
    - datos_abiertos: datos abiertos de COVID-19 en México disponibles en [1].

    Output:
    - series: Series de tiempo para cada estado para cada columna mencionada arriba.

    [1]: https://www.gob.mx/salud/documentos/datos_abiertos-abiertos-152127
    """

    df = pd.DataFrame()

    datos_abiertos['FECHA_INGRESO'] = pd.to_datetime(datos_abiertos['FECHA_INGRESO'])
    datos_abiertos['FECHA_SINTOMAS'] = pd.to_datetime(datos_abiertos['FECHA_SINTOMAS'])

    # Cleaning faulty dates (20200507 had one 1969 date -_-)
    datos_abiertos = datos_abiertos[(datos_abiertos['FECHA_INGRESO'] >= '2020-01-01') & (datos_abiertos['FECHA_SINTOMAS'] >= '2020-01-01')]

    pruebas = (datos_abiertos
              .groupby(['ENTIDAD_UM', 'FECHA_INGRESO'])
              .count()['ORIGEN'])

    confirmados = (datos_abiertos[ (datos_abiertos['RESULTADO'] == 1) ]
              .groupby(['ENTIDAD_UM', 'FECHA_SINTOMAS']) # 'FECHA_INGRESO'
              .count()['ORIGEN'])

    # incluyendo uci
    hospitalizados = (datos_abiertos[ (datos_abiertos['RESULTADO'] == 1) & (datos_abiertos['TIPO_PACIENTE'] == 2) ]
              .groupby(['ENTIDAD_UM', 'FECHA_INGRESO'])
              .count()['ORIGEN'])

    fallecidos_por_hospitalizacion = (datos_abiertos[ (datos_abiertos['RESULTADO'] == 1) & (datos_abiertos['TIPO_PACIENTE'] == 2) & (datos_abiertos['FECHA_DEF'] != '9999-99-99') ]
              .groupby(['ENTIDAD_UM', 'FECHA_DEF'])
              .count()['ORIGEN'])

    fallecidos = (datos_abiertos[ (datos_abiertos['RESULTADO'] == 1) & (datos_abiertos['FECHA_DEF'] != '9999-99-99') ]
              .groupby(['ENTIDAD_UM', 'FECHA_DEF'])
              .count()['ORIGEN'])

    # Convierte las fechas de defunción de str a fecha
    fallecidos.index.set_levels( pd.to_datetime(fallecidos.index.levels[1]), level=1, inplace=True )
    fallecidos_por_hospitalizacion.index.set_levels( pd.to_datetime(fallecidos_por_hospitalizacion.index.levels[1]), level=1, inplace=True )


    df.loc[:,'pruebas_diarias']                        = pruebas
    df.loc[:,'confirmados_diarios']                    = confirmados
    df.loc[:,'hospitalizados_diarios']                 = hospitalizados
    df.loc[:,'fallecidos_diarios']                     = fallecidos
    df.loc[:,'fallecidos_por_hospitalizacion_diarios'] = fallecidos_por_hospitalizacion

    # Convierte str a fechas y llena hoyos de fechas con ceros
    df.index.set_levels( pd.to_datetime(df.index.levels[1]), level=1, inplace=True )
    idx = pd.date_range(df.index.levels[1].min(), df.index.levels[1].max())
    df.index.set_levels( idx, level=1, inplace=True )
    df.index.levels[1].name = 'Fecha'
    df = df.fillna(0)

    # Lista de entidades federativas con nombres oficiales
    entidades = get_lista_entidades()
    df.index.set_levels( entidades, level=0, inplace=True )

    # Genera series acumuladas para cada tipo de caso
    df.loc[:,'pruebas_acumuladas'] = df.loc[:,'pruebas_diarias'].groupby(level=0).cumsum()
    df.loc[:,'confirmados_acumulados'] = df.loc[:,'confirmados_diarios'].groupby(level=0).cumsum()
    df.loc[:,'hospitalizados_acumulados'] = df.loc[:,'hospitalizados_diarios'].groupby(level=0).cumsum()
    df.loc[:,'fallecidos_acumulados']   = df.loc[:,'fallecidos_diarios'].groupby(level=0).cumsum()
    df.loc[:,'fallecidos_por_hospitalizacion_acumulados']   = df.loc[:,'fallecidos_por_hospitalizacion_diarios'].groupby(level=0).cumsum()

    return df.astype('int')

## Funciones útiles
def get_serie_nacional(series): return series.groupby(level=1).sum()
def get_serie_estatal(series, estado): return series.loc[estado]

## Funciones de ayuda
def get_lista_entidades(path_catalogos='./data/diccionario_datos_covid19/Catalogos_0412.xlsx'):
    return pd.read_excel(path_catalogos,
              sheet_name='Catálogo de ENTIDADES')['ENTIDAD_FEDERATIVA'].values

def dias_desde_t0(t0, n_dias=0):
    '''
    Fecha `n_dias` después de `t0`.
    Nota: n_dias puede ser negativo.
    '''
    return pd.to_datetime(t0) + dt.timedelta(days=n_dias)
