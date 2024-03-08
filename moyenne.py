def calcular_media_fps(nome_arquivo):
    # Inicializar uma lista para armazenar os valores de FPS
    valores = []
    
    # Tentar abrir o arquivo para leitura
    try:
        with open(nome_arquivo, 'r') as arquivo:
            # Ler cada linha do arquivo
            for linha in arquivo:
                # Encontrar e extrair o valor de FPS da linha
                partes = linha.split(',')
                for parte in partes:
                    if "FPS" in parte:
                        # Extrair o valor numérico de FPS e convertê-lo para float
                        temps = float(parte.split(':')[1].strip())
                        valores.append(temps)
    
        # Calcular a média dos valores de temps, se houver algum
        if valores:
            temps_moyen = sum(valores) / len(valores)
            print(f"Temps_moyen: {temps_moyen:.2f}")
        else:
            print("Não foram encontrados dados de temps no arquivo.")
    
    except FileNotFoundError:
        print(f"Arquivo {nome_arquivo} não encontrado.")

# Chamar a função com o caminho do arquivo de texto
calcular_media_fps('results_serial.txt')
