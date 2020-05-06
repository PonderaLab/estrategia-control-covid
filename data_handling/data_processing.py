import numpy as np
import pandas as pd

### DATOS POR ESTADO ###
def dataframe_estado(datos, entidades, estado='QUERÉTARO'):
    """
    Genera una tabla con los siguientes datos a nivel `estado`*:
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
    - datos: datos abiertos de COVID-19 en México disponibles en [1].
    - entidades: diccionario {nombre_entidad: clave_entidad}
    - estado: Default='NACIONAL'. Nombre del estado para realizar la tabla

    Output:
    - series: Series de tiempo para cada una de las columnas mencionadas arriba.

    [1]: https://www.gob.mx/salud/documentos/datos-abiertos-152127

    *Si `estado = 'NACIONAL'`, genera la tabla agregada de todo México
    """


    df = pd.DataFrame()


    if estado != 'NACIONAL':
        # Toma datos del estado únicamente
        cve_entidad = entidades[estado]
        datos = datos[ datos['ENTIDAD_UM'] == cve_entidad ]

    datos['FECHA_INGRESO'] = pd.to_datetime(datos['FECHA_INGRESO'])

    pruebas = (datos
              .groupby('FECHA_INGRESO')
              .count()['ORIGEN'])

    confirmados = (datos[ (datos['RESULTADO'] == 1) ]
              .groupby('FECHA_INGRESO')
              .count()['ORIGEN'])

    sospechosos = (datos[ (datos['RESULTADO'] == 3) ]
              .groupby('FECHA_INGRESO')
              .count()['ORIGEN'])

    negativos = (datos[ (datos['RESULTADO'] == 2) ]
              .groupby('FECHA_INGRESO')
              .count()['ORIGEN'])

    # incluyendo uci
    hospitalizados = (datos[ (datos['RESULTADO'] == 1) & (datos['TIPO_PACIENTE'] == 2) ]
              .groupby('FECHA_INGRESO')
              .count()['ORIGEN'])

    uci = (datos[ (datos['RESULTADO'] == 1) & (datos['UCI'] == 1) ]
              .groupby('FECHA_INGRESO')
              .count()['ORIGEN'])

    ambulatorios = (datos[ (datos['RESULTADO'] == 1) & (datos['TIPO_PACIENTE'] == 1) ]
              .groupby('FECHA_INGRESO')
              .count()['ORIGEN'])

    fallecidos = (datos[ (datos['RESULTADO'] == 1) & (datos['FECHA_DEF'] != '9999-99-99') ]
              .groupby('FECHA_DEF')
              .count()['ORIGEN'])
    fallecidos.index = pd.to_datetime(fallecidos.index)

    df.loc[:,'pruebas_diarias']         = pruebas
    df.loc[:,'confirmados_diarios']     = confirmados
    df.loc[:,'sospechosos_diarios']     = sospechosos
    df.loc[:,'negativos_diarios']       = negativos
    df.loc[:,'hospitalizados_diarios']  = hospitalizados
    df.loc[:,'uci_diarios']             = uci
    df.loc[:,'ambulatorios_diarios']    = ambulatorios
    df.loc[:,'fallecidos_diarios']      = fallecidos


    df.index = pd.to_datetime(df.index)
    idx = pd.date_range(df.index.min(), df.index.max())
    df = df.reindex(idx, fill_value=0)
    df.index.name = 'Fecha'
    df = df.fillna(0)

    df.loc[:,'pruebas_acumuladas'] = df.loc[:,'pruebas_diarias'].cumsum()
    df.loc[:,'confirmados_acumulados'] = df.loc[:,'confirmados_diarios'].cumsum()
    df.loc[:,'sospechosos_acumulados'] = df.loc[:,'sospechosos_diarios'].cumsum()
    df.loc[:,'negativos_acumulados'] = df.loc[:,'negativos_diarios'].cumsum()
    df.loc[:,'hospitalizados_acumulados'] = df.loc[:,'hospitalizados_diarios'].cumsum()
    df.loc[:,'uci_acumulados'] = df.loc[:,'uci_diarios'].cumsum()
    df.loc[:,'ambulatorios_acumulados'] = df.loc[:,'ambulatorios_diarios'].cumsum()
    df.loc[:,'fallecidos_acumulados']   = df.loc[:,'fallecidos_diarios'].cumsum()

    return df.astype('int')

    # TODO
    ### DATOS NACIONAL CON PANEL POR ESTADO ###
    def dataframe_nacional(datos, entidades):
        """
        Genera una tabla con los siguientes datos a nivel `estado`*:
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
        - datos: datos abiertos de COVID-19 en México disponibles en [1].
        - entidades: diccionario {nombre_entidad: clave_entidad}
        - estado: Default='NACIONAL'. Nombre del estado para realizar la tabla

        Output:
        - series: Series de tiempo para cada una de las columnas mencionadas arriba.

        [1]: https://www.gob.mx/salud/documentos/datos-abiertos-152127

        *Si `estado = 'NACIONAL'`, genera la tabla agregada de todo México
        """


        df = pd.DataFrame()


        if estado != 'NACIONAL':
            # Toma datos del estado únicamente
            cve_entidad = entidades[estado]
            datos = datos[ datos['ENTIDAD_UM'] == cve_entidad ]

        datos['FECHA_INGRESO'] = pd.to_datetime(datos['FECHA_INGRESO'])

        pruebas = (datos
                  .groupby('FECHA_INGRESO')
                  .count()['ORIGEN'])

        confirmados = (datos[ (datos['RESULTADO'] == 1) ]
                  .groupby('FECHA_INGRESO')
                  .count()['ORIGEN'])

        sospechosos = (datos[ (datos['RESULTADO'] == 3) ]
                  .groupby('FECHA_INGRESO')
                  .count()['ORIGEN'])

        negativos = (datos[ (datos['RESULTADO'] == 2) ]
                  .groupby('FECHA_INGRESO')
                  .count()['ORIGEN'])

        # incluyendo uci
        hospitalizados = (datos[ (datos['RESULTADO'] == 1) & (datos['TIPO_PACIENTE'] == 2) ]
                  .groupby('FECHA_INGRESO')
                  .count()['ORIGEN'])

        uci = (datos[ (datos['RESULTADO'] == 1) & (datos['UCI'] == 1) ]
                  .groupby('FECHA_INGRESO')
                  .count()['ORIGEN'])

        ambulatorios = (datos[ (datos['RESULTADO'] == 1) & (datos['TIPO_PACIENTE'] == 1) ]
                  .groupby('FECHA_INGRESO')
                  .count()['ORIGEN'])

        fallecidos = (datos[ (datos['RESULTADO'] == 1) & (datos['FECHA_DEF'] != '9999-99-99') ]
                  .groupby('FECHA_DEF')
                  .count()['ORIGEN'])
        fallecidos.index = pd.to_datetime(fallecidos.index)

        df.loc[:,'pruebas_diarias']         = pruebas
        df.loc[:,'confirmados_diarios']     = confirmados
        df.loc[:,'sospechosos_diarios']     = sospechosos
        df.loc[:,'negativos_diarios']       = negativos
        df.loc[:,'hospitalizados_diarios']  = hospitalizados
        df.loc[:,'uci_diarios']             = uci
        df.loc[:,'ambulatorios_diarios']    = ambulatorios
        df.loc[:,'fallecidos_diarios']      = fallecidos


        df.index = pd.to_datetime(df.index)
        idx = pd.date_range(df.index.min(), df.index.max())
        df = df.reindex(idx, fill_value=0)
        df.index.name = 'Fecha'
        df = df.fillna(0)

        df.loc[:,'pruebas_acumuladas'] = df.loc[:,'pruebas_diarias'].cumsum()
        df.loc[:,'confirmados_acumulados'] = df.loc[:,'confirmados_diarios'].cumsum()
        df.loc[:,'sospechosos_acumulados'] = df.loc[:,'sospechosos_diarios'].cumsum()
        df.loc[:,'negativos_acumulados'] = df.loc[:,'negativos_diarios'].cumsum()
        df.loc[:,'hospitalizados_acumulados'] = df.loc[:,'hospitalizados_diarios'].cumsum()
        df.loc[:,'uci_acumulados'] = df.loc[:,'uci_diarios'].cumsum()
        df.loc[:,'ambulatorios_acumulados'] = df.loc[:,'ambulatorios_diarios'].cumsum()
        df.loc[:,'fallecidos_acumulados']   = df.loc[:,'fallecidos_diarios'].cumsum()

        return df.astype('int')
