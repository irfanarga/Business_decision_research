#panggil library
import pandas as pd

#panggil dataset
df = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/data_retail.csv', sep=';')

#cetak lima data teratas
print(df.head())

#cetak info dataset
print(df.info())

#mengubah kolom waktu transaksi kedalam format tanggal
df['First_Transaction']=pd.to_datetime(df['First_Transaction']/1000, unit='s', origin='1970-01-01')
df['Last_Transaction']=pd.to_datetime(df['Last_Transaction']/1000, unit='s', origin='1970-01-01')

#Pengecekan transaksi terakhir
print(max(df['Last_Transaction']))

#Klasifikasi customer churn atau tidak dan dimasukkan ke kolom is_churn
df.loc[df['Last_Transaction']<='2018-08-01', 'is_churn'] = True
df.loc[df['Last_Transaction']>'2018-08-01', 'is_churn'] = False

#menghapus kolom yang tidak diperlukan
del df['no']
del df['Row_Num']
