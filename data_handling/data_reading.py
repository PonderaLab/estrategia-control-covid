import pandas as pd

if __name__ == '__main__':
    ## Hay que automatizar esta variable para que descargue los datos más actuales
    datos_abiertos_fecha = '20200505' # YYYYMMDD

    # Path de nuestro repo Mexicovid19 para leer los datos abiertos
    datos_abiertos_path = 'https://raw.githubusercontent.com/mexicovid19/Mexico-datos/master/datos_abiertos/raw/datos_abiertos_{}.csv'.format(datos_abiertos_fecha)

    # Lee base de datos
    datos_abiertos = pd.read_csv( datos_abiertos_path )
