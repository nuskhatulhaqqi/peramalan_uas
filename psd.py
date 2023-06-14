import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import pickle


Home,Implementasi = st.tabs(['Home','Implementasi'])

with Home:
   st.title("""
   Peramalan Data Time Series Pada Produksi Elektronik Menggunakan Metode Regresi Linear
   """)
   st.write('Proyek Sain Data')
   st.text("""
            1. Nuskhatul Haqqi 200411100034 
            2. Amanda Caecilia 200411100090   
            """)
   st.subheader('Tentang Dataset')
   st.write ("""
   Dataset yang digunakan adalah data time series pada produksi elektronik, datanya di dapatkan dari kaggle pada link berikut ini.
   """)
   st.write ("""
    Dataset yang digunakan berjumlah 397 data dan terdapat 2 atribut : 
    """)
   st.write('1. Date : berisi tanggal 1-12 pada bulan januari mulai dari tahun 1985-2018')
   st.write('2. IPG2211A2N : berisi jumlah produksi elektronik')
   st.subheader('Dataset')
   df = pd.read_csv('Electric_Production.csv')
   df

with Implementasi:
   # untuk mengambil data yang akan diproses
   data = df['IPG2211A2N']
   # menghitung jumlah data
   n = len(data)
   # membagi data menjadi 80% untuk data training dan 20% data testing
   sizeTrain = (round(n*0.8))
   data_train = pd.DataFrame(data[:sizeTrain])
   data_test = pd.DataFrame(data[sizeTrain:])
   # melakukan normalisasi menggunakan minMaxScaler
   scaler = MinMaxScaler()
   train_scaled = scaler.fit_transform(data_train)
   # Mengaplikasikan MinMaxScaler pada data pengujian
   test_scaled = scaler.transform(data_test)
   # reshaped_data = data.reshape(-1, 1)
   train = pd.DataFrame(train_scaled, columns = ['data'])
   train = train['data']
   test = pd.DataFrame(test_scaled, columns = ['data'])
   test = test['data']

   def split_sequence(sequence, n_steps):
      X, y = list(), list()
      for i in range(len(sequence)):
         # find the end of this pattern
         end_ix = i + n_steps
         # check if we are beyond the sequence
         if end_ix > len(sequence)-1:
            break
         # gather input and output parts of the pattern
         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
         X.append(seq_x)
         y.append(seq_y)
      return array(X), array(y)
   #memanggil fungsi untuk data training
   df_X, df_Y = split_sequence(train, 4)
   x = pd.DataFrame(df_X, columns = ['xt-4','xt-3','xt-2','xt-1'])
   y = pd.DataFrame(df_Y, columns = ['xt'])
   dataset_train = pd.concat([x, y], axis=1)
   dataset_train.to_csv('data-train.csv', index=False)
   X_train = dataset_train.iloc[:, :4].values
   Y_train = dataset_train.iloc[:, -1].values
   #memanggil fungsi untuk data testing
   test_x, test_y = split_sequence(test, 4)
   x = pd.DataFrame(test_x, columns = ['xt-4','xt-3','xt-2','xt-1'])
   y = pd.DataFrame(test_y, columns = ['xt'])
   dataset_test = pd.concat([x, y], axis=1)
   dataset_test.to_csv('data-test.csv', index=False)
   X_test = dataset_test.iloc[:, :4].values
   Y_test = dataset_test.iloc[:, -1].values
   # Model
   linear = LinearRegression()
   linear.fit(X_train,Y_train)
   y_pred=linear.predict(X_test)
   reshaped_data = y_pred.reshape(-1, 1)
   original_data = scaler.inverse_transform(reshaped_data)
   reshaped_datates = Y_test.reshape(-1, 1)
   actual_test = scaler.inverse_transform(reshaped_datates)
   akhir1 = pd.DataFrame(original_data)
   akhir1.to_csv('prediksi.csv', index=False)
   akhir = pd.DataFrame(actual_test)
   akhir.to_csv('aktual.csv', index=False)
   mape = mean_absolute_percentage_error(original_data, actual_test)
   #menyimpan model
   with open('lr','wb') as r:
      pickle.dump(linear,r)
   with open('minmax','wb') as r:
      pickle.dump(scaler,r)
   
   st.title("""Implementasi Data""")
   input_1 = st.number_input('Masukkan Data 1')
   input_2 = st.number_input('Masukkan Data 2')
   input_3 = st.number_input('Masukkan Data 3')
   input_4 = st.number_input('Masukkan Data 4')

   def submit():
      # inputs = np.array([inputan])
      with open('lr', 'rb') as r:
         model = pickle.load(r)
      with open('minmax', 'rb') as r:
         minmax = pickle.load(r)
      data1 = minmax.transform([[input_1]])
      data2 = minmax.transform([[input_2]])
      data3 = minmax.transform([[input_3]])
      data4 = minmax.transform([[input_4]])

      X_pred = model.predict([[(data1[0][0]),(data2[0][0]),(data3[0][0]),(data4[0][0])]])
      t_data1= X_pred.reshape(-1, 1)
      original = minmax.inverse_transform(t_data1)
      hasil =f"Prediksi Hasil Peramalan jumlah produksi elektronik untuk besok adalah  : {original[0][0]}"
      st.success(hasil)

   all = st.button("Submit")
   if all :
      st.balloons()
      submit()


