import numpy as np
import pandas as pd

### DATOS POR ESTADO ###
def series_panel_por_estado(datos_abiertos, entidades):
    """
    Genera una tabla de panel con los siguientes datos, donde cada panel es un estado:
        - pruebas_diarias
        - confirmados_diarios
        - sospechosos_diarios
        - negativos_diarios
        - hospitalizados_diarios
        - uci_diarios
        - ambulatorios diarios
        - fallecidos_diarios
        - pruebas_acumuladas
        - confirmados_acumulados
        - sospechosos_acumulados
        - negativos_acumulados
        - uci_acumulados
        - ambulatorios_acumulados
        - fallecidos_acumulados

    Inputs:
    - datos_abiertos: datos_abiertos abiertos de COVID-19 en México disponibles en [1].
    - entidades: diccionario {nombre_entidad: clave_entidad}

    Output:
    - series: Series de tiempo para cada estado para cada columna mencionada arriba.

    [1]: https://www.gob.mx/salud/documentos/datos_abiertos-abiertos-152127
    """

    df = pd.DataFrame()

    datos_abiertos['FECHA_INGRESO'] = pd.to_datetime(datos_abiertos['FECHA_INGRESO'])
    datos_abiertos['FECHA_SINTOMAS'] = pd.to_datetime(datos_abiertos['FECHA_SINTOMAS'])

    pruebas = (datos_abiertos
              .groupby(['ENTIDAD_UM', 'FECHA_INGRESO'])
              .count()['ORIGEN'])

    confirmados = (datos_abiertos[ (datos_abiertos['RESULTADO'] == 1) ]
              .groupby(['ENTIDAD_UM', 'FECHA_INGRESO'])                   
              .count()['ORIGEN'])

    sospechosos = (datos_abiertos[ (datos_abiertos['RESULTADO'] == 3) ]
              .groupby(['ENTIDAD_UM', 'FECHA_INGRESO'])
              .count()['ORIGEN'])

    negativos = (datos_abiertos[ (datos_abiertos['RESULTADO'] == 2) ]
              .groupby(['ENTIDAD_UM', 'FECHA_INGRESO'])                 
              .count()['ORIGEN'])

    # incluyendo uci
    hospitalizados = (datos_abiertos[ (datos_abiertos['RESULTADO'] == 1) & (datos_abiertos['TIPO_PACIENTE'] == 2) ]
              .groupby(['ENTIDAD_UM', 'FECHA_INGRESO'])
              .count()['ORIGEN'])

    uci = (datos_abiertos[ (datos_abiertos['RESULTADO'] == 1) & (datos_abiertos['UCI'] == 1) ]
              .groupby(['ENTIDAD_UM', 'FECHA_INGRESO'])
              .count()['ORIGEN'])

    ambulatorios = (datos_abiertos[ (datos_abiertos['RESULTADO'] == 1) & (datos_abiertos['TIPO_PACIENTE'] == 1) ]
              .groupby(['ENTIDAD_UM', 'FECHA_INGRESO'])
              .count()['ORIGEN'])

    fallecidos = (datos_abiertos[ (datos_abiertos['RESULTADO'] == 1) & (datos_abiertos['FECHA_DEF'] != '9999-99-99') ]
              .groupby(['ENTIDAD_UM', 'FECHA_DEF'])
              .count()['ORIGEN'])
    # Convierte las fechas de defunción de str a fecha
    fallecidos.index.set_levels( pd.to_datetime(fallecidos.index.levels[1]), level=1, inplace=True )
#     fallecidos.index = pd.to_datetime(fallecidos.index)

    df.loc[:,'pruebas_diarias']         = pruebas
    df.loc[:,'confirmados_diarios']     = confirmados
    df.loc[:,'sospechosos_diarios']     = sospechosos
    df.loc[:,'negativos_diarios']       = negativos
    df.loc[:,'hospitalizados_diarios']  = hospitalizados
    df.loc[:,'uci_diarios']             = uci
    df.loc[:,'ambulatorios_diarios']    = ambulatorios
    df.loc[:,'fallecidos_diarios']      = fallecidos

    # Convierte str a fechas y llena hoyos de fechas con ceros
    df.index.set_levels( pd.to_datetime(df.index.levels[1]), level=1, inplace=True )
    idx = pd.date_range(df.index.levels[1].min(), df.index.levels[1].max())
    df.index.set_levels( idx, level=1, inplace=True )
    df.index.levels[1].name = 'Fecha'
    df = df.fillna(0)
        
    # Cambia claves de entidades federativas por sus respectivos nombres  
    df.index.set_levels( entidades.keys, level=0, inplace=True )
    
    # Genera series acumuladas para cada tipo de caso
    df.loc[:,'pruebas_acumuladas'] = df.loc[:,'pruebas_diarias'].groupby(level=0).cumsum()
    df.loc[:,'confirmados_acumulados'] = df.loc[:,'confirmados_diarios'].groupby(level=0).cumsum()
    df.loc[:,'sospechosos_acumulados'] = df.loc[:,'sospechosos_diarios'].groupby(level=0).cumsum()
    df.loc[:,'negativos_acumulados'] = df.loc[:,'negativos_diarios'].groupby(level=0).cumsum()
    df.loc[:,'hospitalizados_acumulados'] = df.loc[:,'hospitalizados_diarios'].groupby(level=0).cumsum()
    df.loc[:,'uci_acumulados'] = df.loc[:,'uci_diarios'].groupby(level=0).cumsum()
    df.loc[:,'ambulatorios_acumulados'] = df.loc[:,'ambulatorios_diarios'].groupby(level=0).cumsum()
    df.loc[:,'fallecidos_acumulados']   = df.loc[:,'fallecidos_diarios'].groupby(level=0).cumsum()

    return df.astype('int')

## HELPER FUNCTIONS 
def get_serie_nacional(series_panel): return series.groupby(level=1).sum()
def get_serie_estatal(series_panel, estado): return series.loc[estado]