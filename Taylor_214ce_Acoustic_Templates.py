
from scipy import signal
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import librosa
import IPython.display as ipd

# Taylor guitar templates below

fret_0_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 0#02.wav')
fret_1_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 1#01.wav')
fret_2_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 2#01.wav')
fret_3_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 3#01.wav')
fret_4_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 4#01.wav')
fret_5_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 5#01.wav')
fret_6_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 6#01.wav')
fret_7_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 7#01.wav')
fret_8_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 8#01.wav')
fret_9_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 9#01.wav')
fret_10_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 10#01.wav')
fret_11_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 11#01.wav')
fret_12_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 12#01.wav')
fret_13_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 13#01.wav')
fret_14_2, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce frets/Fret 14#01.wav')

hop_length = 512  
n_fft = 2048 
#fmin = librosa.midi_to_hz(36)

# ###### FRET 0 ###### #

X_0_2 = librosa.stft(fret_0_2, n_fft=n_fft, hop_length=hop_length)
#X_0_2 = librosa.cqt(fret_0_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_0_2 = librosa.amplitude_to_db(abs(X_0_2))
print(S_0_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_0_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_0_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1206, step=25))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_0_dB_2 = np.mean(S_0_2[:,150:175], axis=1)
A_0_dB_2 = np.mean(S_0_2[:,275:300], axis=1)
D_0_dB_2 = np.mean(S_0_2[:,375:400], axis=1)
G_0_dB_2 = np.mean(S_0_2[:,500:525], axis=1)
B_0_dB_2 = np.mean(S_0_2[:,600:625], axis=1)
e_0_dB_2 = np.mean(S_0_2[:,700:725], axis=1)

# ###### FRET 1 ###### #

X_1_2 = librosa.stft(fret_1_2, n_fft=n_fft, hop_length=hop_length)
#X_1_2 = librosa.cqt(fret_1_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_1_2 = librosa.amplitude_to_db(abs(X_1_2))
print(S_1_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_1_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_1_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1465, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_1_dB_2 = np.mean(S_1_2[:,250:275], axis=1)
A_1_dB_2 = np.mean(S_1_2[:,350:375], axis=1)
D_1_dB_2 = np.mean(S_1_2[:,475:500], axis=1)
G_1_dB_2 = np.mean(S_1_2[:,600:625], axis=1)
B_1_dB_2 = np.mean(S_1_2[:,725:750], axis=1)
e_1_dB_2 = np.mean(S_1_2[:,850:875], axis=1)

# ###### FRET 2 ###### #

X_2_2 = librosa.stft(fret_2_2, n_fft=n_fft, hop_length=hop_length)
#X_2_2 = librosa.cqt(fret_2_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_2_2 = librosa.amplitude_to_db(abs(X_2_2))
print(S_2_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_2_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_2_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1465, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_2_dB_2 = np.mean(S_2_2[:,200:225], axis=1)
A_2_dB_2 = np.mean(S_2_2[:,325:350], axis=1)
D_2_dB_2 = np.mean(S_2_2[:,425:450], axis=1)
G_2_dB_2 = np.mean(S_2_2[:,550:575], axis=1)
B_2_dB_2 = np.mean(S_2_2[:,650:675], axis=1)
e_2_dB_2 = np.mean(S_2_2[:,775:800], axis=1)

# ###### FRET 3 ###### #

X_3_2 = librosa.stft(fret_3_2, n_fft=n_fft, hop_length=hop_length)
#X_3_2= librosa.cqt(fret_3_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_3_2 = librosa.amplitude_to_db(abs(X_3_2))
print(S_3_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_3_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_3_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1120, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_3_dB_2 = np.mean(S_3_2[:,75:100], axis=1)
A_3_dB_2 = np.mean(S_3_2[:,200:225], axis=1)
D_3_dB_2 = np.mean(S_3_2[:,325:350], axis=1)
G_3_dB_2 = np.mean(S_3_2[:,425:450], axis=1)
B_3_dB_2 = np.mean(S_3_2[:,550:575], axis=1)
e_3_dB_2 = np.mean(S_3_2[:,650:675], axis=1)

# ###### FRET 4 ###### #

X_4_2 = librosa.stft(fret_4_2, n_fft=n_fft, hop_length=hop_length)
#X_4_2 = librosa.cqt(fret_4_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_4_2 = librosa.amplitude_to_db(abs(X_4_2))
print(S_4_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_4_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_4_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1379, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_4_dB_2 = np.mean(S_4_2[:,150:175], axis=1)
A_4_dB_2 = np.mean(S_4_2[:,275:300], axis=1)
D_4_dB_2 = np.mean(S_4_2[:,375:400], axis=1)
G_4_dB_2 = np.mean(S_4_2[:,500:525], axis=1)
B_4_dB_2 = np.mean(S_4_2[:,600:625], axis=1)
e_4_dB_2 = np.mean(S_4_2[:,725:750], axis=1)

# ###### FRET 5 ###### #

X_5_2 = librosa.stft(fret_5_2, n_fft=n_fft, hop_length=hop_length)
#X_5_2 = librosa.cqt(fret_5_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_5_2 = librosa.amplitude_to_db(abs(X_5_2))
print(S_5_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_5_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_5_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1379, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_5_dB_2 = np.mean(S_5_2[:,175:200], axis=1)
A_5_dB_2 = np.mean(S_5_2[:,300:325], axis=1)
D_5_dB_2 = np.mean(S_5_2[:,400:425], axis=1)
G_5_dB_2 = np.mean(S_5_2[:,525:550], axis=1)
B_5_dB_2 = np.mean(S_5_2[:,650:675], axis=1)
e_5_dB_2 = np.mean(S_5_2[:,750:775], axis=1)

# ###### FRET 6 ###### #

X_6_2 = librosa.stft(fret_6_2, n_fft=n_fft, hop_length=hop_length)
#X_6_2 = librosa.cqt(fret_6_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_6_2 = librosa.amplitude_to_db(abs(X_6_2))
print(S_6_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_6_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_6_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1292, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_6_dB_2 = np.mean(S_6_2[:,125:150], axis=1)
A_6_dB_2 = np.mean(S_6_2[:,250:275], axis=1)
D_6_dB_2 = np.mean(S_6_2[:,350:375], axis=1)
G_6_dB_2 = np.mean(S_6_2[:,475:500], axis=1)
B_6_dB_2 = np.mean(S_6_2[:,575:600], axis=1)
e_6_dB_2 = np.mean(S_6_2[:,700:725], axis=1)

# ###### FRET 7 ###### #

X_7_2 = librosa.stft(fret_7_2, n_fft=n_fft, hop_length=hop_length)
#X_7_2 = librosa.cqt(fret_7_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_7_2 = librosa.amplitude_to_db(abs(X_7_2))
print(S_7_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_7_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_7_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1206, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_7_dB_2 = np.mean(S_7_2[:,150:175], axis=1)
A_7_dB_2 = np.mean(S_7_2[:,275:300], axis=1)
D_7_dB_2 = np.mean(S_7_2[:,375:400], axis=1)
G_7_dB_2 = np.mean(S_7_2[:,475:500], axis=1)
B_7_dB_2 = np.mean(S_7_2[:,600:625], axis=1)
e_7_dB_2 = np.mean(S_7_2[:,700:725], axis=1)

# ###### FRET 8 ###### #

X_8_2 = librosa.stft(fret_8_2, n_fft=n_fft, hop_length=hop_length)
#X_8_2 = librosa.cqt(fret_8_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_8_2 = librosa.amplitude_to_db(abs(X_8_2))
print(S_8_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_8_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_8_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1206, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_8_dB_2 = np.mean(S_8_2[:,125:150], axis=1)
A_8_dB_2 = np.mean(S_8_2[:,225:250], axis=1)
D_8_dB_2 = np.mean(S_8_2[:,325:350], axis=1)
G_8_dB_2 = np.mean(S_8_2[:,450:475], axis=1)
B_8_dB_2 = np.mean(S_8_2[:,550:575], axis=1)
e_8_dB_2 = np.mean(S_8_2[:,650:675], axis=1)

# ###### FRET 9 ###### #

X_9_2 = librosa.stft(fret_9_2, n_fft=n_fft, hop_length=hop_length)
#X_9_2 = librosa.cqt(fret_9_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_9_2 = librosa.amplitude_to_db(abs(X_9_2))
print(S_9_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_9_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_9_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1292, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_9_dB_2 = np.mean(S_9_2[:,150:175], axis=1)
A_9_dB_2 = np.mean(S_9_2[:,250:275], axis=1)
D_9_dB_2 = np.mean(S_9_2[:,375:400], axis=1)
G_9_dB_2 = np.mean(S_9_2[:,500:525], axis=1)
B_9_dB_2 = np.mean(S_9_2[:,600:625], axis=1)
e_9_dB_2 = np.mean(S_9_2[:,725:750], axis=1)

# ###### FRET 10 ###### #

X_10_2 = librosa.stft(fret_10_2, n_fft=n_fft, hop_length=hop_length)
#X_10_2 = librosa.cqt(fret_10_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_10_2 = librosa.amplitude_to_db(abs(X_10_2))
print(S_10_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_10_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_10_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1206, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_10_dB_2 = np.mean(S_10_2[:,75:100], axis=1)
A_10_dB_2 = np.mean(S_10_2[:,200:225], axis=1)
D_10_dB_2 = np.mean(S_10_2[:,300:325], axis=1)
G_10_dB_2 = np.mean(S_10_2[:,400:425], axis=1)
B_10_dB_2 = np.mean(S_10_2[:,500:525], axis=1)
e_10_dB_2 = np.mean(S_10_2[:,625:650], axis=1)

# ###### FRET 11 ###### #

X_11_2 = librosa.stft(fret_11_2, n_fft=n_fft, hop_length=hop_length)
#X_11_2 = librosa.cqt(fret_11_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_11_2 = librosa.amplitude_to_db(abs(X_11_2))
print(S_11_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_11_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_11_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1292, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_11_dB_2 = np.mean(S_11_2[:,200:225], axis=1)
A_11_dB_2 = np.mean(S_11_2[:,300:325], axis=1)
D_11_dB_2 = np.mean(S_11_2[:,400:425], axis=1)
G_11_dB_2 = np.mean(S_11_2[:,500:525], axis=1)
B_11_dB_2 = np.mean(S_11_2[:,600:625], axis=1)
e_11_dB_2 = np.mean(S_11_2[:,725:750], axis=1)

# ###### FRET 12 ###### #

X_12_2 = librosa.stft(fret_12_2, n_fft=n_fft, hop_length=hop_length)
#X_12_2 = librosa.cqt(fret_12_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_12_2 = librosa.amplitude_to_db(abs(X_12_2))
print(S_12_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_12_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_12_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1292, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_12_dB_2 = np.mean(S_12_2[:,225:250], axis=1)
A_12_dB_2 = np.mean(S_12_2[:,350:375], axis=1)
D_12_dB_2 = np.mean(S_12_2[:,450:475], axis=1)
G_12_dB_2 = np.mean(S_12_2[:,550:575], axis=1)
B_12_dB_2 = np.mean(S_12_2[:,650:675], axis=1)
e_12_dB_2 = np.mean(S_12_2[:,750:775], axis=1)

# ###### FRET 13 ###### #

X_13_2 = librosa.stft(fret_13_2, n_fft=n_fft, hop_length=hop_length)
#X_13_2 = librosa.cqt(fret_13_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_13_2 = librosa.amplitude_to_db(abs(X_13_2))
print(S_13_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_13_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_13_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1120, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_13_dB_2 = np.mean(S_13_2[:,150:175], axis=1)
A_13_dB_2 = np.mean(S_13_2[:,250:275], axis=1)
D_13_dB_2 = np.mean(S_13_2[:,350:375], axis=1)
G_13_dB_2 = np.mean(S_13_2[:,450:475], axis=1)
B_13_dB_2 = np.mean(S_13_2[:,550:575], axis=1)
e_13_dB_2 = np.mean(S_13_2[:,650:675], axis=1)

# ###### FRET 14 ###### #

X_14_2 = librosa.stft(fret_14_2, n_fft=n_fft, hop_length=hop_length)
#X_14_2 = librosa.cqt(fret_14_2, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_14_2 = librosa.amplitude_to_db(abs(X_14_2))
print(S_14_2.shape)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(S_14_2, sr=sr, x_axis='frames', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_14_2, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1120, step=50))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

E_14_dB_2 = np.mean(S_14_2[:,150:175], axis=1)
A_14_dB_2 = np.mean(S_14_2[:,250:275], axis=1)
D_14_dB_2 = np.mean(S_14_2[:,350:375], axis=1)
G_14_dB_2 = np.mean(S_14_2[:,450:475], axis=1)
B_14_dB_2 = np.mean(S_14_2[:,550:575], axis=1)
e_14_dB_2 = np.mean(S_14_2[:,650:675], axis=1)

# Arrange templates from each data set in matrices

# taylor 214ce acoustic guitar templates 
template_2 = np.array([[E_0_dB_2, E_1_dB_2, E_2_dB_2, E_3_dB_2, E_4_dB_2, E_5_dB_2, E_6_dB_2, E_7_dB_2, E_7_dB_2, E_9_dB_2, E_10_dB_2, E_11_dB_2, E_12_dB_2, E_13_dB_2, E_14_dB_2], 
                    [A_0_dB_2, A_1_dB_2, A_2_dB_2, A_3_dB_2, A_4_dB_2, A_5_dB_2, A_6_dB_2, A_7_dB_2, A_7_dB_2, A_9_dB_2, A_10_dB_2, A_11_dB_2, A_12_dB_2, A_13_dB_2, A_14_dB_2],
                    [D_0_dB_2, D_1_dB_2, D_2_dB_2, D_3_dB_2, D_4_dB_2, D_5_dB_2, D_6_dB_2, D_7_dB_2, D_7_dB_2, D_9_dB_2, D_10_dB_2, D_11_dB_2, D_12_dB_2, D_13_dB_2, D_14_dB_2],
                    [G_0_dB_2, G_1_dB_2, G_2_dB_2, G_3_dB_2, G_4_dB_2, G_5_dB_2, G_6_dB_2, G_7_dB_2, G_7_dB_2, G_9_dB_2, G_10_dB_2, G_11_dB_2, G_12_dB_2, G_13_dB_2, G_14_dB_2],
                    [B_0_dB_2, B_1_dB_2, B_2_dB_2, B_3_dB_2, B_4_dB_2, B_5_dB_2, B_6_dB_2, B_7_dB_2, B_7_dB_2, B_9_dB_2, B_10_dB_2, B_11_dB_2, B_12_dB_2, B_13_dB_2, B_14_dB_2],
                    [e_0_dB_2, e_1_dB_2, e_2_dB_2, e_3_dB_2, e_4_dB_2, e_5_dB_2, e_6_dB_2, e_7_dB_2, e_7_dB_2, e_9_dB_2, e_10_dB_2, e_11_dB_2, e_12_dB_2, e_13_dB_2, e_14_dB_2]])

###########################################################################################################################################################################################

A, sr = librosa.load('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Taylor 214ce chords/GCGD progression#01.wav') #, duration=3.0)

print(A.shape)

X_A = librosa.stft(A, n_fft=n_fft, hop_length=hop_length)
#X_A = librosa.cqt(A, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
S_A = librosa.amplitude_to_db(abs(X_A))
print(S_A.shape)
plt.rcParams.update({'font.size': 8})
plt.figure(figsize=(20, 4))
librosa.display.specshow(S_A, sr=sr, x_axis='frames', y_axis='linear')
plt.xticks(np.arange(0, 1120, step=25))
plt.colorbar(format='%+2.0f dB')
plt.ylim(0,2000)
plt.show()

#A_template = np.mean(S_A[:,200:30], axis=1)
A_template = np.mean(S_A[:,175:230], axis=1)

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
df.to_csv('C:/Users/Justin Stadlbauer/Documents/RESEARCH/Taylor Templates/Code - Results/GCGD progression#01_G_test(1).csv', index=False)

###########################################################################################################################################################################################