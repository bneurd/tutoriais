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

sr = 250
jump = 5
size = sr * jump

datas = [mne_ga, mne_ga_test, mne_it, mne_it_test, mne_test]
for data in datas:
    results = [0, 0, 0, 0]
    for i in range(0, len(data)-1, sr):
        cut = data.get_data(start=i, stop=i+size)
        nperseg = 128  # Número de pontos por segmento
        noverlap = nperseg // 2  # Quantidade de sobreposição entre segmentos
        
        freqs, psd = welch(cut, fs=sr, nperseg=nperseg, noverlap=noverlap)
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
        bands = [np.sum(X[theta_idxs]), np.sum(X[alpha_idxs]), np.sum(X[beta_idxs]), np.sum(X[gamma_idxs])]
        results[np.argmax(bands)] += 1

    total = sum(results)
    percentages = [round((count/total) * 100, 2) for count in results]
    print(data.__str__())
    print(percentages)