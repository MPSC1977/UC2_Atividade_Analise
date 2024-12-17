import polars as pl
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt

ENDERECO_DADOS = r'./dados/'

# LENDO OS DADOS DO ARQUIVO PARQUET
try:
    print('\nIniciando leitura do arquivo parquet...')

    # Pega o tempo inicial
    inicio = datetime.now()

    # Scan_parquet: Cria um plano de execução preguiçoso para a leitura do parquet
    df_bolsa_familia_plan = pl.scan_parquet(ENDERECO_DADOS + 'bolsa_familia.parquet')

    # Executa as operações lazys e coleta os resultados
    df_bolsa_familia = df_bolsa_familia_plan.collect()

    print(df_bolsa_familia)

    # Pega o tempo final
    fim = datetime.now()

    print(f'Tempo de execução para leitura do parquet: {fim - inicio}')
    print('\nArquivo parquet lido com sucesso!')

except ImportError as e: 
    print(f'Erro ao ler os dados do parquet: {e}')

try:
    print('Calculando medidas de posição...')
    
    array_valor_parcela = np.array(df_bolsa_familia['VALOR PARCELA'])

    media_valor_parcela = np.mean(array_valor_parcela)
    mediana_valor_parcela = np.median(array_valor_parcela)
    distancia_media_mediana = abs((media_valor_parcela - mediana_valor_parcela) / mediana_valor_parcela) * 100

    maximo = np.max(array_valor_parcela)
    minimo = np.min(array_valor_parcela)
    amplitude_total = maximo - minimo

    q1 = np.quantile(array_valor_parcela, 0.25, method='weibull')
    q2 = np.quantile(array_valor_parcela, 0.50, method='weibull')
    q3 = np.quantile(array_valor_parcela, 0.75, method='weibull')

    iqr = q3 - q1

    limite_superior = q3 + (1.5 * iqr)
    limite_inferior = q1 - (1.5 * iqr)

    print('\nMEDIDAS DE TENDÊNCIA CENTRAL:')
    print(30*'-')
    print(f'Média: {media_valor_parcela:.2f}')
    print(f'Mediana: {mediana_valor_parcela:.2f}')
    print(f'Distância: {distancia_media_mediana:.2f}')

    print('\nMEDIDAS DE DISPERSÃO:')
    print(30*'-')
    print('Máximo: ', maximo)
    print('Mínimo: ', minimo)
    print('Amplitude Total: ', amplitude_total)

    print('\nMEDIDAS DE POSIÇÃO:')
    print(30*'-')
    print('Mínimo: ', minimo)
    print(f'Limite Inferior: {limite_inferior:.2f}')
    print('Q1 (25%): ', q1)
    print('Q2 (50%): ', q2)
    print('Q3 (75%): ', q3)
    print(f'IQR: {iqr:.2f}')
    print(f'Limite Superior: {limite_superior:.2f}')
    print('Máximo: ', maximo)

except ImportError as e:
    print(f'Erro ao obter informações sobre medidas estatísticas: {e}')
    exit()


# Processamento e visualização 
# (12 estados com maior valor de parcela)
try:
    print('Calculando os 12 estados com maior valor de parcelas e gerando gráfico de barras e boxplot...')

    # Agrupar por UF e somar o valor das parcelas
    df_estado_parcelas = df_bolsa_familia.group_by('UF').agg(pl.col('VALOR PARCELA').sum().alias('TOTAL PARCELA'))

    # Ordenar de forma decrescente e pegar os 12 primeiros
    df_estado_parcelas = df_estado_parcelas.sort('TOTAL PARCELA', descending=True).head(12)

    # Exibir o DataFrame resultante
    print(df_estado_parcelas)

    print('\nA observação das medidas estatísticas de distância entre a média e mediana (1%) e o valor de IQR abaixo do limite inferior indicam que há uma tendência de homogeneidade dos dados, podendo ser a média, uma medida confiável para a análise. No entanto, pode-se dizer também que, apesar da tendência de pouca dispersão dos dados, observa-se, ao avaliar a amplitude total (R$4.639,00), resultante da diferença entre os valores máximo e mínimo, a existência de valores muito discrepantes (outliers), tanto para cima como para baixo, o que pode ser facilmente observado no gráfico. Portanto, diante da análise apresentada, considera-se viável uma revisão dos valores distribuídos, a fim de mitigar ou avaliar mais a fundo as discrepâncias verificadas, se há situações especiais por exemplo que justifiquem tais valores, e assim tornar mais equitativa e transparente a distribuição do benefício aos seus beneficiários. Somente a título de informações, como pode-se observar no gráfico, entre os 12 maiores estados distribuidores do benefício destacam-se São Paulo e Bahia, sendo os menores distrbuidores entre os 12, Paraná e Rio Grande do Sul')

    # Criar a figura com dois subgráficos lado a lado
    plt.subplots(1, 2, figsize=(15, 6))

    # Gerar gráfico de barras
    plt.subplot(1, 2, 1)

    plt.bar(df_estado_parcelas['UF'], df_estado_parcelas['TOTAL PARCELA'])
    plt.xlabel('Estado (UF)', fontsize=12)
    plt.ylabel('Valor das Parcelas', fontsize=12)
    plt.title('Top 12 Estados com Maior Total de Parcelas', fontsize=14)
    plt.xticks(df_estado_parcelas['UF'], rotation=45, ha='right')

    # Gerar boxplot
    plt.subplot(1, 2, 2)
    array_valor_parcela = np.array(df_bolsa_familia['VALOR PARCELA'])
    plt.boxplot(array_valor_parcela, vert=False)
    plt.title('Distribuição dos Valores das Parcelas', fontsize=14)

    # Ajustar layout
    plt.tight_layout()

    # Exibir gráficos
    plt.show()

    print('Gráficos gerados com sucesso!')

except ImportError as e:
    print(f'Erro ao visualizar os dados: {e}')

