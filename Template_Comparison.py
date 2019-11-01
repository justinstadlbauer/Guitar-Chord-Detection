# Template Comparison

# %matplotlib inline
from scipy import signal
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import librosa
import IPython.display as ipd

fret_0_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_0_2.wav')
fret_1_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_1_2.wav')
fret_2_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_2_2.wav')
fret_3_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_3_2.wav')
fret_4_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_4_2.wav')
fret_5_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_5_2.wav')
fret_6_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_6_2.wav')
fret_7_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_7_2.wav')
fret_8_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_8_2.wav')
fret_9_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_9_2.wav')
fret_10_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_10_2.wav')
fret_11_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_11_2.wav')
fret_12_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_12_2.wav')
fret_13_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_13_2.wav')
fret_14_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates - Second Data Set/Fret_14_2.wav')

hop_length = 512  
n_fft = 2048 
fmin = librosa.midi_to_hz(36)

template_files = np.array([fret_0_2, fret_1_2, fret_2_2, fret_3_2, fret_4_2, fret_5_2, fret_6_2, fret_7_2,
                           fret_8_2, fret_9_2, fret_10_2, fret_11_2, fret_12_2, fret_13_2, fret_14_2])

template_2 = np.zeros((6L,15L,1025L))
for i in range(15):
    X = librosa.stft(template_files[i], n_fft=n_fft, hop_length=hop_length)
    S = librosa.amplitude_to_db(abs(X))

    template_2[0][i] = np.mean(S[:,25:50], axis=1)
    template_2[1][i] = np.mean(S[:,200:225], axis=1)
    template_2[2][i] = np.mean(S[:,375:400], axis=1)
    template_2[3][i] = np.mean(S[:,550:575], axis=1)
    template_2[4][i] = np.mean(S[:,725:750], axis=1)
    template_2[5][i] = np.mean(S[:,900:925], axis=1)

###############################################################################################################

A, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic Mic Only 60 BPM/A_mic_only.wav')
# A, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/E_A_D_Chord_Strum.wav') #, duration=3.0)

# print(A.shape)

X_A = librosa.stft(A, n_fft=n_fft, hop_length=hop_length)
# X_A = librosa.cqt(A, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_A = librosa.amplitude_to_db(abs(X_A))
# print(S_A.shape)
# plt.rcParams.update({'font.size': 8})
# plt.figure(figsize=(20, 4))
# librosa.display.specshow(S_A, sr=sr, x_axis='frames', y_axis='linear')
# plt.xticks(np.arange(0, 1895, step=100))
# plt.colorbar(format='%+2.0f dB')
# plt.ylim(0,2000)
# plt.show()

A_template = np.mean(S_A[:,1400:1600], axis=1)

chord_corr_A = np.zeros((1,90))
chord_corr_A_matrix = np.zeros((6,15))

k=0
for i in range(6):
    for j in range(15):
        # print(np.corrcoef(A_template, template_2[i][j])[1,0])
        chord_corr_A[0][k] = np.corrcoef(A_template, template_2[i][j])[1,0]
        chord_corr_A_matrix[i][j] = np.corrcoef(A_template, template_2[i][j])[1,0]
        k+=1

templates = np.zeros((11,6,5))
k = 5
for i in range(11):
    templates[i] = chord_corr_A_matrix[0:6, i:k]
    k=k+1

A_frets_location = np.zeros((11, 6)).astype(int)
A_fret_max = np.zeros((11, 6)).astype(int)
for i in range(11):
    A_fret_max[i] = np.argmax(templates[i], axis = 1)
    A_frets_location[i] = np.argmax(templates[i], axis = 1) + i

A_largest_corr = np.zeros((11, 6))
for i in range(11):
    for j in range(6):
        A_largest_corr[i][j] = templates[i][j][A_fret_max[i][j]]

A_scores = np.sum(A_largest_corr,axis=1).reshape(11,1)
A_chord_final = np.append(A_frets_location,A_scores,axis=1)

# 'x' out strings that were not played in the chord
chord_result = np.zeros((0,6))
chord_result = A_chord_final[np.argmax(A_chord_final[:,6]),:6].astype(int)

cqt_bins = np.array([[4,9,14,19,23,28],     # Fret 0
                     [5,10,15,20,24,29],    # Fret 1
                     [6,11,16,21,25,30],    # Fret 2
                     [7,12,17,22,26,31],    # Fret 3
                     [8,13,18,23,27,32],    # Fret 4
                     [9,14,19,24,28,33],    # Fret 5
                     [10,15,20,25,29,34],   # Fret 6
                     [11,16,21,26,30,35],   # Fret 7
                     [12,17,22,27,31,36],   # Fret 8
                     [13,18,23,28,32,37],   # Fret 9
                     [14,19,24,29,33,38],   # Fret 10
                     [15,20,25,30,34,39],   # Fret 11
                     [16,21,26,31,35,40],   # Fret 12
                     [17,22,27,32,36,41],   # Fret 13
                     [18,23,28,33,37,42]])  # Fret 14

cqt_values = np.zeros(6)
for i in range(6):
    cqt_values[i] = cqt_bins[chord_result[i],i]

cqt_values = cqt_values.astype(int)

X_cqt = librosa.cqt(A, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_cqt = librosa.amplitude_to_db(abs(X_cqt))
# plt.figure(figsize=(15, 5))
# librosa.display.specshow(S_cqt, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
# plt.xticks(np.arange(0, 1895, step=50))

decibel_values = np.zeros(6)
for i in range(6):
    decibel_values[i] = np.mean(np.abs(S_cqt[cqt_values[i],1400:1600]))

average_db = np.average(decibel_values, axis=0)
std_db = np.std(decibel_values, axis=0)
three_std_db = std_db*3

tab_result = np.zeros(6).astype(str)

for i in range(6):
    if (decibel_values[i] < three_std_db):
        tab_result[i] = chord_result[i]
    else:
        tab_result[i] = "x"

print(tab_result)

#import pandas as pd
#df = pd.DataFrame(A_chord_final)
#df.to_csv('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic Mic Only - Tremelo Effect/E7_Trem.csv', index=False)

################################################################################################################
