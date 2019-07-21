# Template Comparison

# %matplotlib inline
from scipy import signal
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import librosa
import IPython.display as ipd

fret_0, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_0.wav/')
fret_1, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_1.wav')
fret_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_2.wav')
fret_3, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_3.wav')
fret_4, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_4.wav')
fret_5, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_5.wav')
fret_6, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_6.wav')
fret_7, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_7.wav')
fret_8, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_8.wav')
fret_9, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_9.wav')
fret_10, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_10.wav')
fret_11, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_11.wav')
fret_12, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_12.wav')
fret_13, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_13.wav')
fret_14, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic 14 Fret Templates/Fret_14.wav')

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

# ###### FRET 0 ###### #
  
X_0 = librosa.stft(fret_0, n_fft=n_fft, hop_length=hop_length)
#X_0 = librosa.cqt(fret_0, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_0 = librosa.amplitude_to_db(abs(X_0))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_0, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_0, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1034, step=25))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0, 2000)
plt.show()

X_0_2 = librosa.stft(fret_0_2, n_fft=n_fft, hop_length=hop_length)
#X_0_2 = librosa.cqt(fret_0_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_0_2 = librosa.amplitude_to_db(abs(X_0_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_0_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_0_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_0_dB = np.mean(S_0[:,75:100], axis=1)
A_0_dB = np.mean(S_0[:,250:275], axis=1)
D_0_dB = np.mean(S_0[:,425:450], axis=1)
G_0_dB = np.mean(S_0[:,600:625], axis=1)
B_0_dB = np.mean(S_0[:,775:800], axis=1)
e_0_dB = np.mean(S_0[:,950:975], axis=1)

E_0_dB_2 = np.mean(S_0_2[:,25:50], axis=1)
A_0_dB_2 = np.mean(S_0_2[:,200:225], axis=1)
D_0_dB_2 = np.mean(S_0_2[:,375:400], axis=1)
G_0_dB_2 = np.mean(S_0_2[:,550:575], axis=1)
B_0_dB_2 = np.mean(S_0_2[:,725:750], axis=1)
e_0_dB_2 = np.mean(S_0_2[:,900:925], axis=1)

# ###### FRET 1 ###### #

X_1 = librosa.stft(fret_1, n_fft=n_fft, hop_length=hop_length)
#X_1 = librosa.cqt(fret_1, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_1 = librosa.amplitude_to_db(abs(X_1))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_1, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_1, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_1_2 = librosa.stft(fret_1_2, n_fft=n_fft, hop_length=hop_length)
#X_1_2 = librosa.cqt(fret_1_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_1_2 = librosa.amplitude_to_db(abs(X_1_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_1_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_1_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_1_dB = np.mean(S_1[:,75:100], axis=1)
A_1_dB = np.mean(S_1[:,250:275], axis=1)
D_1_dB = np.mean(S_1[:,425:450], axis=1)
G_1_dB = np.mean(S_1[:,600:625], axis=1)
B_1_dB = np.mean(S_1[:,775:800], axis=1)
e_1_dB = np.mean(S_1[:,950:975], axis=1)

E_1_dB_2 = np.mean(S_1_2[:,25:50], axis=1)
A_1_dB_2 = np.mean(S_1_2[:,200:225], axis=1)
D_1_dB_2 = np.mean(S_1_2[:,375:400], axis=1)
G_1_dB_2 = np.mean(S_1_2[:,550:575], axis=1)
B_1_dB_2 = np.mean(S_1_2[:,725:750], axis=1)
e_1_dB_2 = np.mean(S_1_2[:,900:925], axis=1)

# ###### FRET 2 ###### #

X_2 = librosa.stft(fret_2, n_fft=n_fft, hop_length=hop_length)
#X_2 = librosa.cqt(fret_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_2 = librosa.amplitude_to_db(abs(X_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_2_2 = librosa.stft(fret_2_2, n_fft=n_fft, hop_length=hop_length)
#X_2_2 = librosa.cqt(fret_2_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_2_2 = librosa.amplitude_to_db(abs(X_2_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_2_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_2_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_2_dB = np.mean(S_2[:,75:100], axis=1)
A_2_dB = np.mean(S_2[:,250:275], axis=1)
D_2_dB = np.mean(S_2[:,425:450], axis=1)
G_2_dB = np.mean(S_2[:,600:625], axis=1)
B_2_dB = np.mean(S_2[:,775:800], axis=1)
e_2_dB = np.mean(S_2[:,950:975], axis=1)

E_2_dB_2 = np.mean(S_2_2[:,25:50], axis=1)
A_2_dB_2 = np.mean(S_2_2[:,200:225], axis=1)
D_2_dB_2 = np.mean(S_2_2[:,375:400], axis=1)
G_2_dB_2 = np.mean(S_2_2[:,550:575], axis=1)
B_2_dB_2 = np.mean(S_2_2[:,725:750], axis=1)
e_2_dB_2 = np.mean(S_2_2[:,900:925], axis=1)

# ###### FRET 3 ###### #

X_3 = librosa.stft(fret_3, n_fft=n_fft, hop_length=hop_length)
#X_3 = librosa.cqt(fret_3, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_3 = librosa.amplitude_to_db(abs(X_3))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_3, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_3, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_3_2 = librosa.stft(fret_3_2, n_fft=n_fft, hop_length=hop_length)
#X_3_2= librosa.cqt(fret_3_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_3_2 = librosa.amplitude_to_db(abs(X_3_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_3_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_3_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_3_dB = np.mean(S_3[:,75:100], axis=1)
A_3_dB = np.mean(S_3[:,250:275], axis=1)
D_3_dB = np.mean(S_3[:,425:450], axis=1)
G_3_dB = np.mean(S_3[:,600:625], axis=1)
B_3_dB = np.mean(S_3[:,775:800], axis=1)
e_3_dB = np.mean(S_3[:,950:975], axis=1)

E_3_dB_2 = np.mean(S_3_2[:,25:50], axis=1)
A_3_dB_2 = np.mean(S_3_2[:,200:225], axis=1)
D_3_dB_2 = np.mean(S_3_2[:,375:400], axis=1)
G_3_dB_2 = np.mean(S_3_2[:,550:575], axis=1)
B_3_dB_2 = np.mean(S_3_2[:,725:750], axis=1)
e_3_dB_2 = np.mean(S_3_2[:,900:925], axis=1)

# ###### FRET 4 ###### #

X_4 = librosa.stft(fret_4, n_fft=n_fft, hop_length=hop_length)
#X_4 = librosa.cqt(fret_4, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_4 = librosa.amplitude_to_db(abs(X_4))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_4, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_4, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_4_2 = librosa.stft(fret_4_2, n_fft=n_fft, hop_length=hop_length)
#X_4_2 = librosa.cqt(fret_4_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_4_2 = librosa.amplitude_to_db(abs(X_4_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_4_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_4_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_4_dB = np.mean(S_4[:,75:100], axis=1)
A_4_dB = np.mean(S_4[:,250:275], axis=1)
D_4_dB = np.mean(S_4[:,425:450], axis=1)
G_4_dB = np.mean(S_4[:,600:625], axis=1)
B_4_dB = np.mean(S_4[:,775:800], axis=1)
e_4_dB = np.mean(S_4[:,950:975], axis=1)

E_4_dB_2 = np.mean(S_4_2[:,25:50], axis=1)
A_4_dB_2 = np.mean(S_4_2[:,200:225], axis=1)
D_4_dB_2 = np.mean(S_4_2[:,375:400], axis=1)
G_4_dB_2 = np.mean(S_4_2[:,550:575], axis=1)
B_4_dB_2 = np.mean(S_4_2[:,725:750], axis=1)
e_4_dB_2 = np.mean(S_4_2[:,900:925], axis=1)

# ###### FRET 5 ###### #

X_5 = librosa.stft(fret_5, n_fft=n_fft, hop_length=hop_length)
#X_5 = librosa.cqt(fret_5, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_5 = librosa.amplitude_to_db(abs(X_5))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_5, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_5, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_5_2 = librosa.stft(fret_5_2, n_fft=n_fft, hop_length=hop_length)
#X_5_2 = librosa.cqt(fret_5_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_5_2 = librosa.amplitude_to_db(abs(X_5_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_5_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_5_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_5_dB = np.mean(S_5[:,75:100], axis=1)
A_5_dB = np.mean(S_5[:,250:275], axis=1)
D_5_dB = np.mean(S_5[:,425:450], axis=1)
G_5_dB = np.mean(S_5[:,600:625], axis=1)
B_5_dB = np.mean(S_5[:,775:800], axis=1)
e_5_dB = np.mean(S_5[:,950:975], axis=1)

E_5_dB_2 = np.mean(S_5_2[:,25:50], axis=1)
A_5_dB_2 = np.mean(S_5_2[:,200:225], axis=1)
D_5_dB_2 = np.mean(S_5_2[:,375:400], axis=1)
G_5_dB_2 = np.mean(S_5_2[:,550:575], axis=1)
B_5_dB_2 = np.mean(S_5_2[:,725:750], axis=1)
e_5_dB_2 = np.mean(S_5_2[:,900:925], axis=1)

# ###### FRET 6 ###### #

X_6 = librosa.stft(fret_6, n_fft=n_fft, hop_length=hop_length)
#X_6 = librosa.cqt(fret_6, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_6 = librosa.amplitude_to_db(abs(X_6))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_6, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_6, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_6_2 = librosa.stft(fret_6_2, n_fft=n_fft, hop_length=hop_length)
#X_6_2 = librosa.cqt(fret_6_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_6_2 = librosa.amplitude_to_db(abs(X_6_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_6_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_6_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_6_dB = np.mean(S_6[:,75:100], axis=1)
A_6_dB = np.mean(S_6[:,250:275], axis=1)
D_6_dB = np.mean(S_6[:,425:450], axis=1)
G_6_dB = np.mean(S_6[:,600:625], axis=1)
B_6_dB = np.mean(S_6[:,775:800], axis=1)
e_6_dB = np.mean(S_6[:,950:975], axis=1)

E_6_dB_2 = np.mean(S_6_2[:,25:50], axis=1)
A_6_dB_2 = np.mean(S_6_2[:,200:225], axis=1)
D_6_dB_2 = np.mean(S_6_2[:,375:400], axis=1)
G_6_dB_2 = np.mean(S_6_2[:,550:575], axis=1)
B_6_dB_2 = np.mean(S_6_2[:,725:750], axis=1)
e_6_dB_2 = np.mean(S_6_2[:,900:925], axis=1)

# ###### FRET 7 ###### #

X_7 = librosa.stft(fret_7, n_fft=n_fft, hop_length=hop_length)
#X_7 = librosa.cqt(fret_7, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_7 = librosa.amplitude_to_db(abs(X_7))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_7, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_7, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_7_2 = librosa.stft(fret_7_2, n_fft=n_fft, hop_length=hop_length)
#X_7_2 = librosa.cqt(fret_7_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_7_2 = librosa.amplitude_to_db(abs(X_7_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_7_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_7_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_7_dB = np.mean(S_7[:,75:100], axis=1)
A_7_dB = np.mean(S_7[:,250:275], axis=1)
D_7_dB = np.mean(S_7[:,425:450], axis=1)
G_7_dB = np.mean(S_7[:,600:625], axis=1)
B_7_dB = np.mean(S_7[:,775:800], axis=1)
e_7_dB = np.mean(S_7[:,950:975], axis=1)

E_7_dB_2 = np.mean(S_7_2[:,25:50], axis=1)
A_7_dB_2 = np.mean(S_7_2[:,200:225], axis=1)
D_7_dB_2 = np.mean(S_7_2[:,375:400], axis=1)
G_7_dB_2 = np.mean(S_7_2[:,550:575], axis=1)
B_7_dB_2 = np.mean(S_7_2[:,725:750], axis=1)
e_7_dB_2 = np.mean(S_7_2[:,900:925], axis=1)

# ###### FRET 8 ###### #

X_8 = librosa.stft(fret_8, n_fft=n_fft, hop_length=hop_length)
#X_8 = librosa.cqt(fret_8, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_8 = librosa.amplitude_to_db(abs(X_8))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_8, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_8, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_8_2 = librosa.stft(fret_8_2, n_fft=n_fft, hop_length=hop_length)
#X_8_2 = librosa.cqt(fret_8_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_8_2 = librosa.amplitude_to_db(abs(X_8_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_8_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_8_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_8_dB = np.mean(S_8[:,75:100], axis=1)
A_8_dB = np.mean(S_8[:,250:275], axis=1)
D_8_dB = np.mean(S_8[:,425:450], axis=1)
G_8_dB = np.mean(S_8[:,600:625], axis=1)
B_8_dB = np.mean(S_8[:,775:800], axis=1)
e_8_dB = np.mean(S_8[:,950:975], axis=1)

E_8_dB_2 = np.mean(S_8_2[:,25:50], axis=1)
A_8_dB_2 = np.mean(S_8_2[:,200:225], axis=1)
D_8_dB_2 = np.mean(S_8_2[:,375:400], axis=1)
G_8_dB_2 = np.mean(S_8_2[:,550:575], axis=1)
B_8_dB_2 = np.mean(S_8_2[:,725:750], axis=1)
e_8_dB_2 = np.mean(S_8_2[:,900:925], axis=1)

# ###### FRET 9 ###### #

X_9 = librosa.stft(fret_9, n_fft=n_fft, hop_length=hop_length)
#X_9 = librosa.cqt(fret_9, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_9 = librosa.amplitude_to_db(abs(X_9))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_9, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_9, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_9_2 = librosa.stft(fret_9_2, n_fft=n_fft, hop_length=hop_length)
#X_9_2 = librosa.cqt(fret_9_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_9_2 = librosa.amplitude_to_db(abs(X_9_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_9_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_9_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_9_dB = np.mean(S_9[:,75:100], axis=1)
A_9_dB = np.mean(S_9[:,250:275], axis=1)
D_9_dB = np.mean(S_9[:,425:450], axis=1)
G_9_dB = np.mean(S_9[:,600:625], axis=1)
B_9_dB = np.mean(S_9[:,775:800], axis=1)
e_9_dB = np.mean(S_9[:,950:975], axis=1)

E_9_dB_2 = np.mean(S_9_2[:,25:50], axis=1)
A_9_dB_2 = np.mean(S_9_2[:,200:225], axis=1)
D_9_dB_2 = np.mean(S_9_2[:,375:400], axis=1)
G_9_dB_2 = np.mean(S_9_2[:,550:575], axis=1)
B_9_dB_2 = np.mean(S_9_2[:,725:750], axis=1)
e_9_dB_2 = np.mean(S_9_2[:,900:925], axis=1)

# ###### FRET 10 ###### #

X_10 = librosa.stft(fret_10, n_fft=n_fft, hop_length=hop_length)
#X_10 = librosa.cqt(fret_10, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_10 = librosa.amplitude_to_db(abs(X_10))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_10, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_10, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_10_2 = librosa.stft(fret_10_2, n_fft=n_fft, hop_length=hop_length)
#X_10_2 = librosa.cqt(fret_10_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_10_2 = librosa.amplitude_to_db(abs(X_10_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_10_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_10_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_10_dB = np.mean(S_10[:,75:100], axis=1)
A_10_dB = np.mean(S_10[:,250:275], axis=1)
D_10_dB = np.mean(S_10[:,425:450], axis=1)
G_10_dB = np.mean(S_10[:,600:625], axis=1)
B_10_dB = np.mean(S_10[:,775:800], axis=1)
e_10_dB = np.mean(S_10[:,950:975], axis=1)

E_10_dB_2 = np.mean(S_10_2[:,25:50], axis=1)
A_10_dB_2 = np.mean(S_10_2[:,200:225], axis=1)
D_10_dB_2 = np.mean(S_10_2[:,375:400], axis=1)
G_10_dB_2 = np.mean(S_10_2[:,550:575], axis=1)
B_10_dB_2 = np.mean(S_10_2[:,725:750], axis=1)
e_10_dB_2 = np.mean(S_10_2[:,900:925], axis=1)

# ###### FRET 11 ###### #

X_11 = librosa.stft(fret_11, n_fft=n_fft, hop_length=hop_length)
#X_11 = librosa.cqt(fret_11, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_11 = librosa.amplitude_to_db(abs(X_11))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_11, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_11, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_11_2 = librosa.stft(fret_11_2, n_fft=n_fft, hop_length=hop_length)
#X_11_2 = librosa.cqt(fret_11_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_11_2 = librosa.amplitude_to_db(abs(X_11_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_11_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_11_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_11_dB = np.mean(S_11[:,75:100], axis=1)
A_11_dB = np.mean(S_11[:,250:275], axis=1)
D_11_dB = np.mean(S_11[:,425:450], axis=1)
G_11_dB = np.mean(S_11[:,600:625], axis=1)
B_11_dB = np.mean(S_11[:,775:800], axis=1)
e_11_dB = np.mean(S_11[:,950:975], axis=1)

E_11_dB_2 = np.mean(S_11_2[:,25:50], axis=1)
A_11_dB_2 = np.mean(S_11_2[:,200:225], axis=1)
D_11_dB_2 = np.mean(S_11_2[:,375:400], axis=1)
G_11_dB_2 = np.mean(S_11_2[:,550:575], axis=1)
B_11_dB_2 = np.mean(S_11_2[:,725:750], axis=1)
e_11_dB_2 = np.mean(S_11_2[:,900:925], axis=1)

# ###### FRET 12 ###### #

X_12 = librosa.stft(fret_12, n_fft=n_fft, hop_length=hop_length)
#X_12 = librosa.cqt(fret_12, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_12 = librosa.amplitude_to_db(abs(X_12))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_12, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_12, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_12_2 = librosa.stft(fret_12_2, n_fft=n_fft, hop_length=hop_length)
#X_12_2 = librosa.cqt(fret_12_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_12_2 = librosa.amplitude_to_db(abs(X_12_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_12_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_12_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_12_dB = np.mean(S_12[:,75:100], axis=1)
A_12_dB = np.mean(S_12[:,250:275], axis=1)
D_12_dB = np.mean(S_12[:,425:450], axis=1)
G_12_dB = np.mean(S_12[:,600:625], axis=1)
B_12_dB = np.mean(S_12[:,775:800], axis=1)
e_12_dB = np.mean(S_12[:,950:975], axis=1)

E_12_dB_2 = np.mean(S_12_2[:,25:50], axis=1)
A_12_dB_2 = np.mean(S_12_2[:,200:225], axis=1)
D_12_dB_2 = np.mean(S_12_2[:,375:400], axis=1)
G_12_dB_2 = np.mean(S_12_2[:,550:575], axis=1)
B_12_dB_2 = np.mean(S_12_2[:,725:750], axis=1)
e_12_dB_2 = np.mean(S_12_2[:,900:925], axis=1)

# ###### FRET 13 ###### #

X_13 = librosa.stft(fret_13, n_fft=n_fft, hop_length=hop_length)
#X_13 = librosa.cqt(fret_13, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_13 = librosa.amplitude_to_db(abs(X_13))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_13, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_13, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_13_2 = librosa.stft(fret_13_2, n_fft=n_fft, hop_length=hop_length)
#X_13_2 = librosa.cqt(fret_13_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_13_2 = librosa.amplitude_to_db(abs(X_13_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_13_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_13_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_13_dB = np.mean(S_13[:,75:100], axis=1)
A_13_dB = np.mean(S_13[:,250:275], axis=1)
D_13_dB = np.mean(S_13[:,425:450], axis=1)
G_13_dB = np.mean(S_13[:,600:625], axis=1)
B_13_dB = np.mean(S_13[:,775:800], axis=1)
e_13_dB = np.mean(S_13[:,950:975], axis=1)

E_13_dB_2 = np.mean(S_13_2[:,25:50], axis=1)
A_13_dB_2 = np.mean(S_13_2[:,200:225], axis=1)
D_13_dB_2 = np.mean(S_13_2[:,375:400], axis=1)
G_13_dB_2 = np.mean(S_13_2[:,550:575], axis=1)
B_13_dB_2 = np.mean(S_13_2[:,725:750], axis=1)
e_13_dB_2 = np.mean(S_13_2[:,900:925], axis=1)

# ###### FRET 14 ###### #

X_14 = librosa.stft(fret_14, n_fft=n_fft, hop_length=hop_length)
#X_14 = librosa.cqt(fret_14, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_14 = librosa.amplitude_to_db(abs(X_14))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_14, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_14, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

X_14_2 = librosa.stft(fret_14_2, n_fft=n_fft, hop_length=hop_length)
#X_14_2 = librosa.cqt(fret_14_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_14_2 = librosa.amplitude_to_db(abs(X_14_2))
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_14_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#plt.figure(figsize=(20, 4))
#librosa.display.specshow(S_14_2, sr=sr, x_axis='frames', y_axis='linear')
#plt.xticks(np.arange(0, 1034, step=25))
#plt.colorbar(format='%+2.0f dB')
#plt.ylim(0,2000)
#plt.show()

E_14_dB = np.mean(S_14[:,75:100], axis=1)
A_14_dB = np.mean(S_14[:,250:275], axis=1)
D_14_dB = np.mean(S_14[:,425:450], axis=1)
G_14_dB = np.mean(S_14[:,600:625], axis=1)
B_14_dB = np.mean(S_14[:,775:800], axis=1)
e_14_dB = np.mean(S_14[:,950:975], axis=1)

E_14_dB_2 = np.mean(S_14_2[:,25:50], axis=1)
A_14_dB_2 = np.mean(S_14_2[:,200:225], axis=1)
D_14_dB_2 = np.mean(S_14_2[:,375:400], axis=1)
G_14_dB_2 = np.mean(S_14_2[:,550:575], axis=1)
B_14_dB_2 = np.mean(S_14_2[:,725:750], axis=1)
e_14_dB_2 = np.mean(S_14_2[:,900:925], axis=1)

# Arrange templates from each data set in matrices

# First data set
template = np.array([[E_0_dB, E_1_dB, E_2_dB, E_3_dB, E_4_dB, E_5_dB, E_6_dB, E_7_dB, E_7_dB, E_9_dB, E_10_dB, E_11_dB, E_12_dB, E_13_dB, E_14_dB], 
                    [A_0_dB, A_1_dB, A_2_dB, A_3_dB, A_4_dB, A_5_dB, A_6_dB, A_7_dB, A_7_dB, A_9_dB, A_10_dB, A_11_dB, A_12_dB, A_13_dB, A_14_dB],
                    [D_0_dB, D_1_dB, D_2_dB, D_3_dB, D_4_dB, D_5_dB, D_6_dB, D_7_dB, D_7_dB, D_9_dB, D_10_dB, D_11_dB, D_12_dB, D_13_dB, D_14_dB],
                    [G_0_dB, G_1_dB, G_2_dB, G_3_dB, G_4_dB, G_5_dB, G_6_dB, G_7_dB, G_7_dB, G_9_dB, G_10_dB, G_11_dB, G_12_dB, G_13_dB, G_14_dB],
                    [B_0_dB, B_1_dB, B_2_dB, B_3_dB, B_4_dB, B_5_dB, B_6_dB, B_7_dB, B_7_dB, B_9_dB, B_10_dB, B_11_dB, B_12_dB, B_13_dB, B_14_dB],
                    [e_0_dB, e_1_dB, e_2_dB, e_3_dB, e_4_dB, e_5_dB, e_6_dB, e_7_dB, e_7_dB, e_9_dB, e_10_dB, e_11_dB, e_12_dB, e_13_dB, e_14_dB]])

# Second data set
template_2 = np.array([[E_0_dB_2, E_1_dB_2, E_2_dB_2, E_3_dB_2, E_4_dB_2, E_5_dB_2, E_6_dB_2, E_7_dB_2, E_7_dB_2, E_9_dB_2, E_10_dB_2, E_11_dB_2, E_12_dB_2, E_13_dB_2, E_14_dB_2], 
                    [A_0_dB_2, A_1_dB_2, A_2_dB_2, A_3_dB_2, A_4_dB_2, A_5_dB_2, A_6_dB_2, A_7_dB_2, A_7_dB_2, A_9_dB_2, A_10_dB_2, A_11_dB_2, A_12_dB_2, A_13_dB_2, A_14_dB_2],
                    [D_0_dB_2, D_1_dB_2, D_2_dB_2, D_3_dB_2, D_4_dB_2, D_5_dB_2, D_6_dB_2, D_7_dB_2, D_7_dB_2, D_9_dB_2, D_10_dB_2, D_11_dB_2, D_12_dB_2, D_13_dB_2, D_14_dB_2],
                    [G_0_dB_2, G_1_dB_2, G_2_dB_2, G_3_dB_2, G_4_dB_2, G_5_dB_2, G_6_dB_2, G_7_dB_2, G_7_dB_2, G_9_dB_2, G_10_dB_2, G_11_dB_2, G_12_dB_2, G_13_dB_2, G_14_dB_2],
                    [B_0_dB_2, B_1_dB_2, B_2_dB_2, B_3_dB_2, B_4_dB_2, B_5_dB_2, B_6_dB_2, B_7_dB_2, B_7_dB_2, B_9_dB_2, B_10_dB_2, B_11_dB_2, B_12_dB_2, B_13_dB_2, B_14_dB_2],
                    [e_0_dB_2, e_1_dB_2, e_2_dB_2, e_3_dB_2, e_4_dB_2, e_5_dB_2, e_6_dB_2, e_7_dB_2, e_7_dB_2, e_9_dB_2, e_10_dB_2, e_11_dB_2, e_12_dB_2, e_13_dB_2, e_14_dB_2]])

corr_results = np.zeros((6,15))
print(corr_results.shape)

for i in range(6):
    for j in range(15):
        print(np.corrcoef(template[i][j], template_2[i][j])[1,0])
        corr_results[i][j] = np.corrcoef(template[i][j], template_2[i][j])[1,0]

import pandas as pd
df = pd.DataFrame(corr_results)
df.to_csv('Compare_Two_Data_Sets_CQT.csv', index=False)

# Correlate templates for Em chord

Em, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic Mic Only 60 BPM/Em_mic_only.wav')

print(Em.shape)

X_Em = librosa.stft(Em, n_fft=n_fft, hop_length=hop_length)
#X_Em = librosa.cqt(Em, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_Em = librosa.amplitude_to_db(abs(X_Em))
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_Em, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1895, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

Em_template = np.mean(S_Em[:,1550:1600], axis=1)

chord_corr_Em = np.zeros((1,90))
chord_corr_Em_matrix = np.zeros((6,15))
k=0
for i in range(6):
    for j in range(15):
        print(np.corrcoef(Em_template, template_2[i][j])[1,0])
        chord_corr_Em[0][k] = np.corrcoef(Em_template, template_2[i][j])[1,0]
        chord_corr_Em_matrix[i][j] = np.corrcoef(Em_template, template_2[i][j])[1,0]
        k+=1

df2 = pd.DataFrame(chord_corr_Em)
df2.to_csv('Em_Corr_STFT.csv', index=False)

x = ('E0','E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12','E13','E14',
     'A0','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14',
     'D0','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14',
     'G0','G1','G2','G3','G4','G5','G6','G7','G8','G9','G10','G11','G12','G13','G14',
     'B0','B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12','B13','B14',
     'e0','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11','e12','e13','e14')

x_Em = np.arange(len(x))
plt.figure(figsize=(30, 6))
plt.rcParams.update({'font.size': 8})
plt.bar(x_Em, chord_corr_Em[0],align='center',width=0.3)
plt.xticks(x_Em, x)
plt.rcParams.update({'font.size': 14})
plt.title('Template Correlation for Em (STFT)')
plt.xlabel('String/Fret')
plt.ylabel('Correlation')
plt.savefig('Em_Corr_STFT.png')
plt.show()

# Correlate templates for A chord

A, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic Mic Only 60 BPM/A_mic_only.wav')

print(A.shape)

X_A = librosa.stft(A, n_fft=n_fft, hop_length=hop_length)
#X_A = librosa.cqt(A, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_A = librosa.amplitude_to_db(abs(X_A))
plt.rcParams.update({'font.size': 8})
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_A, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1895, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

A_template = np.mean(S_A[:,1550:1600], axis=1)

chord_corr_A = np.zeros((1,90))
chord_corr_A_matrix = np.zeros((6,15))

k=0
for i in range(6):
    for j in range(15):
        print(np.corrcoef(A_template, template_2[i][j])[1,0])
        chord_corr_A[0][k] = np.corrcoef(A_template, template_2[i][j])[1,0]
        chord_corr_A_matrix[i][j] = np.corrcoef(A_template, template_2[i][j])[1,0]
        k+=1

df3 = pd.DataFrame(chord_corr_A)
df3.to_csv('A_Corr_CQT.csv', index=False)

x_A = np.arange(len(x))
plt.figure(figsize=(30, 6))
plt.rcParams.update({'font.size': 8})
plt.bar(x_A, chord_corr_A[0],align='center',width=0.3)
plt.xticks(x_A, x)
plt.rcParams.update({'font.size': 14})
plt.title('Template Correlation for A (STFT)')
plt.xlabel('String/Fret')
plt.ylabel('Correlation')
plt.savefig('A_Corr_STFT.png')
plt.show()

# Correlate templates for D7 chord

D7, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic Mic Only 60 BPM/D7_mic_only.wav')

print(D7.shape)

X_D7 = librosa.stft(D7, n_fft=n_fft, hop_length=hop_length)
#X_D7 = librosa.cqt(D7, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_D7 = librosa.amplitude_to_db(abs(X_D7))
plt.rcParams.update({'font.size': 8})
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_D7, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1895, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

D7_template = np.mean(S_D7[:,1550:1600], axis=1)

chord_corr_D7 = np.zeros((1,90))
chord_corr_D7_matrix = np.zeros((6,15))

k=0
for i in range(6):
    for j in range(15):
        print(np.corrcoef(D7_template, template_2[i][j])[1,0])
        chord_corr_D7[0][k] = np.corrcoef(D7_template, template_2[i][j])[1,0]
        chord_corr_D7_matrix[i][j] = np.corrcoef(D7_template, template_2[i][j])[1,0]
        k+=1

df4 = pd.DataFrame(chord_corr_D7)
df4.to_csv('D7_Corr_STFT.csv', index=False)

x_D7 = np.arange(len(x))
plt.figure(figsize=(30, 6))
plt.rcParams.update({'font.size': 8})
plt.bar(x_D7, chord_corr_D7[0],align='center',width=0.3)
plt.xticks(x_D7, x)
plt.rcParams.update({'font.size': 14})
plt.title('Template Correlation for D7 (STFT)')
plt.xlabel('String/Fret')
plt.ylabel('Correlation')
plt.savefig('D7_Corr_STFT.png')
plt.show()

# ###### Algorithm ###### #

# Em
Em_chord_0_4 = chord_corr_Em_matrix[0:6, 0:5]
print(Em_chord_0_4)
Em_chord_5_9 = chord_corr_Em_matrix[0:6, 6:10]
Em_chord_10_14 = chord_corr_Em_matrix[0:6, 11:15]

Em_fret_max = np.zeros((3, 6))
Em_frets_0_4_location = np.argmax(Em_chord_0_4, axis=1)
Em_frets_5_9_location = np.argmax(Em_chord_5_9, axis=1)+5
Em_frets_10_14_location = np.argmax(Em_chord_10_14, axis=1)+10
Em_fret_max = np.append(Em_fret_max, [
                        Em_frets_0_4_location, Em_frets_5_9_location, Em_frets_10_14_location], axis=0)
Em_fret_max = Em_fret_max[3:6, 0:6]

print(Em_frets_0_4_location)
print(Em_frets_5_9_location)
print(Em_frets_10_14_location)

Em_frets_0_4 = np.argmax(Em_chord_0_4, axis=1)
Em_frets_5_9 = np.argmax(Em_chord_5_9, axis=1)
Em_frets_10_14 = np.argmax(Em_chord_10_14, axis=1)

print(Em_frets_5_9)

Em_largest_corr = np.zeros((3, 6))
for j in range(6):
    Em_largest_corr[0][j] = Em_chord_0_4[j][Em_frets_0_4[j]]
    Em_largest_corr[1][j] = Em_chord_5_9[j][Em_frets_5_9[j]]
    Em_largest_corr[2][j] = Em_chord_10_14[j][Em_frets_10_14[j]]

Em_strings = np.argmax(Em_largest_corr, axis=0)

print(Em_strings)
Em_chord_final = np.zeros(6)

for i in range(6):
    Em_chord_final[i] = Em_fret_max[Em_strings[i]][i]

print(Em_strings)
print("\n")
print(Em_fret_max)
print("\n")
print(Em_chord_final)
print("\n")
print(Em_largest_corr)

###############################################################################################################
A, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Acoustic Mic Only 60 BPM/E_mic_only.wav')

print(A.shape)

X_A = librosa.stft(A, n_fft=n_fft, hop_length=hop_length)
#X_A = librosa.cqt(A, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_A = librosa.amplitude_to_db(abs(X_A))
plt.rcParams.update({'font.size': 8})
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_A, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1895, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

A_template = np.mean(S_A[:,1550:1600], axis=1)

chord_corr_A = np.zeros((1,90))
chord_corr_A_matrix = np.zeros((6,15))

k=0
for i in range(6):
    for j in range(15):
        print(np.corrcoef(A_template, template_2[i][j])[1,0])
        chord_corr_A[0][k] = np.corrcoef(A_template, template_2[i][j])[1,0]
        chord_corr_A_matrix[i][j] = np.corrcoef(A_template, template_2[i][j])[1,0]
        k+=1
       
# A
A_chord_1_4 = chord_corr_A_matrix[0:6, 0:5] # includes open string
A_chord_2_5 = chord_corr_A_matrix[0:6, 1:6]
A_chord_3_6 = chord_corr_A_matrix[0:6, 2:7]
A_chord_4_7 = chord_corr_A_matrix[0:6, 3:8]
A_chord_5_8 = chord_corr_A_matrix[0:6, 4:9]
A_chord_6_9 = chord_corr_A_matrix[0:6, 5:10]
A_chord_7_10 = chord_corr_A_matrix[0:6, 6:11]
A_chord_8_11 = chord_corr_A_matrix[0:6, 7:12]
A_chord_9_12 = chord_corr_A_matrix[0:6, 8:13]
A_chord_10_13 = chord_corr_A_matrix[0:6, 9:14]
A_chord_11_14 = chord_corr_A_matrix[0:6, 10:15]

a = chord_corr_A_matrix[0:6, 0]
b = a.reshape(-1,1)

A_chord_1_4_aug = np.concatenate((b, A_chord_1_4), axis = 1)
A_chord_2_5_aug = np.concatenate((b, A_chord_2_5), axis = 1)
A_chord_3_6_aug = np.concatenate((b, A_chord_3_6), axis = 1)
A_chord_4_7_aug = np.concatenate((b, A_chord_4_7), axis = 1)
A_chord_5_8_aug = np.concatenate((b, A_chord_5_8), axis = 1)
A_chord_6_9_aug = np.concatenate((b, A_chord_6_9), axis = 1)
A_chord_7_10_aug = np.concatenate((b, A_chord_7_10), axis = 1)
A_chord_8_11_aug = np.concatenate((b, A_chord_8_11), axis = 1)
A_chord_9_12_aug = np.concatenate((b, A_chord_9_12), axis = 1)
A_chord_10_13_aug = np.concatenate((b, A_chord_10_13), axis = 1)
A_chord_11_14_aug = np.concatenate((b, A_chord_11_14), axis = 1)

A_chord_templates = np.array([A_chord_1_4, A_chord_2_5, A_chord_3_6, A_chord_4_7, A_chord_5_8, A_chord_6_9, 
                              A_chord_7_10, A_chord_8_11, A_chord_9_12, A_chord_10_13, A_chord_11_14])

A_chord_templates_aug = np.array([A_chord_1_4_aug, A_chord_2_5_aug, A_chord_3_6_aug, A_chord_4_7_aug, A_chord_5_8_aug, A_chord_6_9_aug, A_chord_7_10_aug, A_chord_8_11_aug, A_chord_9_12_aug, A_chord_10_13_aug, A_chord_11_14_aug])

A_fret_max = np.zeros((11, 6))
#for i in range(12):
#   A_fret_max = np.append(A_fret_max, [[np.argmax(A_chord_templates[i]+1, axis=1)]], axis=0)
A_frets_1_4_location = np.argmax(A_chord_1_4, axis=1) # REMOVE "AUG" TO GO BACK TO THE OLD WAY!
A_frets_2_5_location = np.argmax(A_chord_2_5, axis=1)#+1
A_frets_3_6_location = np.argmax(A_chord_3_6, axis=1)#+2
A_frets_4_7_location = np.argmax(A_chord_4_7, axis=1)#+3
A_frets_5_8_location = np.argmax(A_chord_5_8, axis=1)#+4
A_frets_6_9_location = np.argmax(A_chord_6_9, axis=1)#+5
A_frets_7_10_location = np.argmax(A_chord_7_10, axis=1)#+6
A_frets_8_11_location = np.argmax(A_chord_8_11, axis=1)#+7
A_frets_9_12_location = np.argmax(A_chord_9_12, axis=1)#+8
A_frets_10_13_location = np.argmax(A_chord_10_13, axis=1)#+9
A_frets_11_14_location = np.argmax(A_chord_11_14, axis=1)#+10

A_fret_max = np.append(A_fret_max, [A_frets_1_4_location,A_frets_2_5_location,A_frets_3_6_location, A_frets_4_7_location,A_frets_5_8_location,A_frets_6_9_location,A_frets_7_10_location,A_frets_8_11_location, A_frets_9_12_location,A_frets_10_13_location,A_frets_11_14_location], axis=0)

#print(np.argmax(A_chord_7_10, axis=1))
A_fret_max = A_fret_max[11:22, 0:6]
A_fret_max = A_fret_max.astype(int)

A_frets_location = np.zeros((11, 6))
for i in range(11):
    temp = A_fret_max[i]+i
    A_frets_location = np.append(A_frets_location,[temp],axis=0)

A_frets_location = A_frets_location[11:22, 0:6]

print(A_frets_location)

A_largest_corr = np.zeros((11, 6))
for i in range(11):
    for j in range(6): # CHANGE BACK TO "6" TO GO BACK TO THE OLD WAY!
        A_largest_corr[i][j] = A_chord_templates[i][j][A_fret_max[i][j]]

A_scores = np.zeros((11,1))
for i in range(11):
    A_scores[i] = np.sum(A_largest_corr[i],axis=0)

A_chord_final = np.append(A_frets_location,A_scores,axis=1)

print(A_chord_final)

import pandas as pd
df = pd.DataFrame(A_chord_final)
df.to_csv('D7_chord_final_aug.csv', index=False)

#print(A_largest_corr)
#print(A_chord_1_4)
#print(A_chord_2_5)
#print(A_fret_max)
#print(A_chord_templates[0][1][0])
################################################################################################################
