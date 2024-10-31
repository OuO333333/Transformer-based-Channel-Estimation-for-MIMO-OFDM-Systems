from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPool2D, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from numpy import *
import numpy as np
import numpy.linalg as LA
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
import scipy.io as sio
from AttentionModule import Multiheadattention
from tensorflow.keras.layers import AveragePooling1D



Nt=32
Nr=16
SNR_dB = 20
# get command line argv
args = sys.argv
if len(args) == 2:
    try:
        SNR_dB = int(args[1])
    except ValueError:
        print("intput not valid")
SNR=10.0**(SNR_dB/10.0) # transmit power
print("SNR = ", SNR)# DFT matrix
# DFT matrix
def DFT_matrix(N):
    m, n = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1j / N )
    D = np.power( omega, m * n )
    return D

def sub_dftmtx(A, N):
    D=A[:,0:N]
    return D

F_DFT=DFT_matrix(Nt)/np.sqrt(Nt)
W_DFT=DFT_matrix(Nr)/np.sqrt(Nr)

Nt_beam=32
F_RF=F_DFT[:,0:Nt_beam]
F=F_RF
F_conj=np.conjugate(F)
F_conjtransp=np.transpose(F_conj)
FFH=np.dot(F,F_conjtransp)
Nr_beam=16
W_RF=W_DFT[:,0:Nr_beam]
W=W_RF
W_conj=np.conjugate(W)
W_conjtransp=np.transpose(W_conj)
WWH=np.dot(W,W_conjtransp)

Nt_beam1=16
F_RF1=F_DFT[:,0:Nt_beam1]
F1=F_RF1
F_conj1=np.conjugate(F1)
F_conjtransp1=np.transpose(F_conj1)
FFH1=np.dot(F1,F_conjtransp1)
Nr_beam1=4
W_RF1=W_DFT[:,0:Nr_beam1]
W1=W_RF1
W_conj1=np.conjugate(W1)
W_conjtransp1=np.transpose(W_conj1)
WWH1=np.dot(W1,W_conjtransp1)

Nt_beam2=16
F_RF2=F_DFT[:,0:Nt_beam2]
F2=F_RF2
F_conj2=np.conjugate(F2)
F_conjtransp2=np.transpose(F_conj2)
FFH2=np.dot(F1,F_conjtransp2)
Nr_beam2=4
W_RF2=W_DFT[:,0:Nr_beam2]
W2=W_RF2
W_conj2=np.conjugate(W2)
W_conjtransp2=np.transpose(W_conj2)
WWH2=np.dot(W2,W_conjtransp2)

Nt_beam3=16
F_RF3=F_DFT[:,0:Nt_beam3]
F3=F_RF3
F_conj3=np.conjugate(F3)
F_conjtransp3=np.transpose(F_conj3)
FFH3=np.dot(F3,F_conjtransp3)
Nr_beam3=4
W_RF3=W_DFT[:,0:Nr_beam3]
W3=W_RF3
W_conj3=np.conjugate(W3)
W_conjtransp3=np.transpose(W_conj3)
WWH3=np.dot(W3,W_conjtransp3)

scale=2
fre=2
time_steps=4

############## training set generation ##################
data_num_train=500
data_num_file=500
H_train=zeros((data_num_train,Nr,Nt,2*fre), dtype=float)
H_train_noisy=zeros((data_num_train,Nr,Nt,2*fre*time_steps), dtype=float)
current_directory = os.getcwd()
filedir = os.path.join(current_directory, '2fre4time_data')  # type the path of training datan=0
n=0
SNRr=0
SNR_factor=5.9  # compensate channel power gain to approximate to 1
# 生成一个接近于单位矩阵的 IRS 矩阵
H_shape = 32
IRS_scale = 0.01
IRS = np.eye(H_shape) + np.random.normal(0, IRS_scale, H_shape)  # 添加均值为0，标准差为0.01的随机扰动
for filename in os.listdir(filedir):
    newname = os.path.join(filedir, filename)
    data = sio.loadmat(newname)
    channel = data['ChannelData_fre']
    for i in range(data_num_file):
        for j in range(fre):
            a=channel[:,:,j,i]
            for t in range(time_steps):
                a1=a[:,t*Nr:t*Nr+Nr]
                H=np.transpose(a1)
                H = np.dot(H, IRS)
                H_re=np.real(H)
                H_im = np.imag(H)
                if t==3:
                    H_train[n * data_num_file + i, :, :, 2 * j] = H_re / scale
                    H_train[n * data_num_file + i, :, :, 2 * j + 1] = H_im / scale
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam))
                NFH = np.dot(N, F_conjtransp)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam1)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam1))
                NFH1=np.dot(N,F_conjtransp1)
                N1=np.dot(WWH1,NFH1)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam2)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam2))
                NFH2 = np.dot(N, F_conjtransp2)
                N2 = np.dot(WWH2, NFH2)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam3)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam3))
                NFH3 = np.dot(N, F_conjtransp3)
                N3 = np.dot(WWH3, NFH3)
                if t == 0:
                    Y = H + 1.0 / np.sqrt(SNR_factor * SNR) * NFH
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H)) ** 2 / (LA.norm(NFH)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 1:
                    HF1=np.dot(H,FFH1)
                    H1=np.dot(WWH1,HF1)
                    Y = H1 + 1.0 / np.sqrt(SNR_factor * SNR) * N1
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H1)) ** 2 / (LA.norm(N1)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 2:
                    HF2 = np.dot(H, FFH2)
                    H2 = np.dot(WWH2, HF2)
                    Y = H2 + 1.0 / np.sqrt(SNR_factor * SNR) * N2
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H2)) ** 2 / (LA.norm(N2)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 3:
                    HF3 = np.dot(H, FFH3)
                    H3 = np.dot(WWH3, HF3)
                    Y = H3 + 1.0 / np.sqrt(SNR_factor * SNR) * N3
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H3)) ** 2 / (LA.norm(N3)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
    n=n+1
print(n)
print(SNRr/(data_num_train*fre*time_steps))
print(H_train.shape,H_train_noisy.shape)
index1=np.where(abs(H_train)>1)
row_num=np.unique(index1[0])
H_train=np.delete(H_train,row_num,axis=0)
H_train_noisy=np.delete(H_train_noisy,row_num,axis=0)
print(len(row_num))
print(H_train.shape,H_train_noisy.shape)

index1=np.where(abs(H_train)>1)
row_num=np.unique(index1[0])
H_train=np.delete(H_train,row_num,axis=0)
H_train_noisy=np.delete(H_train_noisy,row_num,axis=0)

############## testing set generation ##################
data_num_test=500
data_num_file=500
H_test=zeros((data_num_test,Nr,Nt,2*fre), dtype=float)
H_test_noisy=zeros((data_num_test,Nr,Nt,2*fre*time_steps), dtype=float)
current_directory = os.getcwd()
filedir = os.path.join(current_directory, '2fre4time_data')  # type the path of training datan=0
n=0
SNRr=0
SNR_factor=5.9  # compensate channel power gain to approximate to 1
for filename in os.listdir(filedir):
    newname = os.path.join(filedir, filename)
    data = sio.loadmat(newname)
    channel = data['ChannelData_fre']
    for i in range(data_num_file):
        for j in range(fre):
            a=channel[:,:,j,i]
            for t in range(time_steps):
                a1=a[:,t*Nr:t*Nr+Nr]
                H=np.transpose(a1)
                H = np.dot(H, IRS)
                H_re=np.real(H)
                H_im = np.imag(H)
                if t==3:
                    H_test[n * data_num_file + i, :, :, 2 * j] = H_re / scale
                    H_test[n * data_num_file + i, :, :, 2 * j + 1] = H_im / scale
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam))
                NFH = np.dot(N, F_conjtransp)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam1)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam1))
                NFH1 = np.dot(N, F_conjtransp1)
                N1 = np.dot(WWH1, NFH1)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam2)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam2))
                NFH2 = np.dot(N, F_conjtransp2)
                N2 = np.dot(WWH2, NFH2)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam3)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam3))
                NFH3 = np.dot(N, F_conjtransp3)
                N3 = np.dot(WWH3, NFH3)
                if t == 0:
                    Y = H + 1.0 / np.sqrt(SNR_factor * SNR) * NFH
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H)) ** 2 / (LA.norm(NFH)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 1:
                    HF1 = np.dot(H, FFH1)
                    H1 = np.dot(WWH1, HF1)
                    Y = H1 + 1.0 / np.sqrt(SNR_factor * SNR) * N1
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H1)) ** 2 / (LA.norm(N1)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 2:
                    HF2 = np.dot(H, FFH2)
                    H2 = np.dot(WWH2, HF2)
                    Y = H2 + 1.0 / np.sqrt(SNR_factor * SNR) * N2
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H2)) ** 2 / (LA.norm(N2)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 3:
                    HF3 = np.dot(H, FFH3)
                    H3 = np.dot(WWH3, HF3)
                    Y = H3 + 1.0 / np.sqrt(SNR_factor * SNR) * N3
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H3)) ** 2 / (LA.norm(N3)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
    n = n + 1
print(n)
print(SNRr/(data_num_test*fre*time_steps))
print(H_test.shape,H_test_noisy.shape)
index3 = np.where(abs(H_test) > 1)
row_num = np.unique(index3[0])
H_test = np.delete(H_test, row_num, axis=0)
H_test_noisy = np.delete(H_test_noisy, row_num, axis=0)
print(len(row_num))
print(H_test.shape, H_test_noisy.shape)
print(((H_test)**2).mean())

index3 = np.where(abs(H_test) > 1)
row_num = np.unique(index3[0])
H_test = np.delete(H_test, row_num, axis=0)
H_test_noisy = np.delete(H_test_noisy, row_num, axis=0)

print("H_train_noisy shape = ", H_train_noisy.shape)
print("H_train shape = ", H_train.shape)
print("H_test_noisy shape = ", H_test_noisy.shape)
print("H_test shape = ", H_test.shape)

K=3
input_dim = tf.keras.Input(shape=(Nr,Nt,2*fre*time_steps))
print("input_dim shape = ", input_dim.shape)
x = Multiheadattention(64,32,4)(input_dim)
x = tf.keras.layers.Conv2D(2*fre*time_steps,(3,3),padding='same', activation='relu')(input_dim)
x = tf.keras.layers.Conv2D(32,(3,3),padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(16,(3,3),padding='same', activation='relu',  activity_regularizer=tf.keras.regularizers.l2(10e-8))(x)
x = tf.keras.layers.Conv2D(32,(3,3),padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(2*fre*time_steps,(3,3),padding='same', activation='relu')(x)
x = Multiheadattention(64,32,4)(input_dim+x)
outputs = tf.keras.layers.Conv2D(time_steps,(3,3),padding='same', activation='tanh')(x)

# Create the model
model = Model(inputs=input_dim, outputs=outputs)

# checkpoint
filepath='2fre4time_SNR20_time4_16_4_200ep.tf'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

adam=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss='mse')
model.fit(H_train_noisy, H_train, epochs=200, batch_size=32, callbacks=callbacks_list, verbose=2, shuffle=True, validation_split=0.1)
print("H_train_noisy shape = ", H_train_noisy.shape, ", H_train shape = ", H_train.shape)

# load model
CNN = load_model('2fre4time_SNR20_time4_16_4_200ep.tf')

for i in range(5):
    # start time
    start_time = time.time()
    decoded_channel = CNN.predict(H_test_noisy)

    # end time
    end_time = time.time()
    # 计算执行时间
    execution_time = end_time - start_time
    print("执行时间：", execution_time, "秒")

nmse2=zeros((data_num_test-len(row_num),1), dtype=float)
for n in range(data_num_test-len(row_num)):
    MSE = tf.reduce_sum(tf.square(H_test - decoded_channel))
    norm_real = tf.reduce_sum(tf.square(H_test))
    nmse2[n]=MSE/norm_real
print("NMSE = ", nmse2.sum()/(data_num_test-len(row_num)))  # calculate NMSE of current training stage

def Sumrate(h_test,h_est,bandwidth):
    numerator = np.sum((h_test-h_est)**2)
    denominator = np.sum((h_test-np.mean(h_test))**2)
    rate = bandwidth * np.log2(1+(2*denominator-numerator)/denominator)
    return rate
print("Sumrate(bandwidth = 10) = ", Sumrate(H_test, decoded_channel, 10))  # calculate NMSE of current training stage

def print_shape(reshape_type):
    shapes = {'Nr': 0, 'Nt': 1, 'channel': 2}
    shape_order = [None] * 3
    for shape, index in shapes.items():
        shape_order[index] = shape

    # 重新排列形状顺序
    reshaped_order = [shape_order[i] for i in reshape_type]

    # 打印结果
    print(f"({', '.join(reshaped_order)})")

# 打开文件以写入模式
with open('output.txt', 'a') as f:
    # 保存原始的标准输出
    original_stdout = sys.stdout
    
    # 将标准输出重定向到文件
    sys.stdout = f
    print("SPARSEMATTDAE")
    print("   ", '{:>3}'.format(SNR_dB), "        ", '{:>20}'.format(nmse2.sum()/(data_num_test-len(row_num))), "        ", '{:>20}'.format(Sumrate(H_test, decoded_channel, 10)))

    # 恢复原始的标准输出
    sys.stdout = original_stdout