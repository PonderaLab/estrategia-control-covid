import pandas as pd

'''
    Este módulo lee los datos abiertos de la DGE [1]. El formato de fecha es YYYYMMDD
    [1]: https://www.gob.mx/salud/documentos/datos-abiertos-152127
'''

## Hay que automatizar el argumento de main que descargue los datos más actuales
def main(datos_abiertos_fecha='20200509'):
    # Path de nuestro repo Mexicovid19 para leer los datos abiertos
    datos_abiertos_path = 'https://raw.githubusercontent.com/mexicovid19/Mexico-datos/master/datos_abiertos/raw/datos_abiertos_{}.zip'.format(datos_abiertos_fecha)

    # Lee base de datos
    datos_abiertos = pd.read_csv( datos_abiertos_path )
    return datos_abiertos

if __name__ == '__main__':
    main()
