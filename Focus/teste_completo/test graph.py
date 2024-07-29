import mne
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt


# abrindo os dados filtrados salvos em arquivos .fif (MNE)
mne_ga = mne.io.read_raw_fif('dataset/Daniella/ga.fif')
mne_ga_test = mne.io.read_raw_fif('dataset/Daniella/ga_test.fif')
mne_it = mne.io.read_raw_fif('dataset/Daniella/it.fif')
mne_it_test = mne.io.read_raw_fif('dataset/Daniella/it_test.fif')
mne_test = mne.io.read_raw_fif('dataset/Daniella/test.fif')

def calcular_potencias(segmento):
    nperseg = 128  # Número de pontos por segmento
    noverlap = nperseg // 2  # Quantidade de sobreposição entre segmentos
    
    freqs, psd = welch(segmento, fs=sr, nperseg=nperseg, noverlap=noverlap)
    X = np.average(psd, axis=0)

    # Definir os limites das bandas de frequência (em Hz)
    theta_band = (4, 8)       # Theta: 4 - 8 Hz
    alpha_band = (8, 13)      # Alpha: 8 - 13 Hz
    beta_band = (13, 30)      # Beta: 13 - 30 Hz
    gamma_band = (30, 100)    # Gamma: 30 - 100 Hz

    # Encontrar os índices correspondentes às frequências de interesse
    theta_idxs = np.where((freqs >= theta_band[0]) & (freqs <= theta_band[1]))[0]
    alpha_idxs = np.where((freqs >= alpha_band[0]) & (freqs <= alpha_band[1]))[0]
    beta_idxs = np.where((freqs >= beta_band[0]) & (freqs <= beta_band[1]))[0]
    gamma_idxs = np.where((freqs >= gamma_band[0]) & (freqs <= gamma_band[1]))[0]

    # Calcular a potência em cada banda de frequência
    theta_power = np.sum(X[theta_idxs])
    alpha_power = np.sum(X[alpha_idxs])
    beta_power = np.sum(X[beta_idxs])
    gamma_power = np.sum(X[gamma_idxs])

    return theta_power, alpha_power, beta_power, gamma_power

sr = 250
jump = 5
size = sr * jump

bandas_por_tempo = {'Theta': [], 'Alpha': [], 'Beta': [], 'Gamma': []}
datas = [mne_ga, mne_ga_test, mne_it, mne_it_test, mne_test]
for data in datas:
    for i in range(0, len(data)-1, size):
        cut = data.get_data(start=i, stop=i+size)
        theta_power, alpha_power, beta_power, gamma_power = calcular_potencias(cut)
        bandas_por_tempo['Theta'].append(theta_power)
        bandas_por_tempo['Alpha'].append(alpha_power)
        bandas_por_tempo['Beta'].append(beta_power)
        bandas_por_tempo['Gamma'].append(gamma_power)

# Calcular média de cada banda de frequência ao longo do tempo
media_por_tempo = {banda: np.mean(potencias) for banda, potencias in bandas_por_tempo.items()}

# Plotar o gráfico de área empilhada
plt.figure(figsize=(10, 6))
plt.stackplot(range(int(media_por_tempo['Theta'])), media_por_tempo['Theta'], media_por_tempo['Alpha'],
              media_por_tempo['Beta'], media_por_tempo['Gamma'], labels=['Theta', 'Alpha', 'Beta', 'Gamma'])
plt.title('Timeline de Potência em cada Banda de Frequência')
plt.xlabel('Tempo (segundos)')
plt.ylabel('Potência')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()