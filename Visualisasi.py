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

import matplotlib.pyplot as plt

#Tahun transaksi
df['Year_First_Transaction'] = df['First_Transaction'].dt.year
df['Year_Last_Transaction'] = df['Last_Transaction'].dt.year

#Plot jumlah customer berdasarkan tahun transaksi pertama
df_year = df.groupby(['Year_First_Transaction'])['Customer_ID'].count()
df_year.plot(x='Year_First_Transaction', y='Customer_ID', kind='bar', title='Graph of Customer Acquisition')
plt.xlabel('Year_First_Transaction')
plt.ylabel('Num_of_Customer')
plt.tight_layout()
plt.show()

#Plot jumlah transaksi berdasarkan tahun transaksi pertama
df_year = df.groupby(['Year_First_Transaction'])['Count_Transaction'].sum()
df_year.plot(x='Year_First_Transaction', y='Count_Transaction', kind='bar', title='Graph of Transaction Customer')
plt.xlabel('Year_First_Transaction')
plt.ylabel('Num_of_Transaction')
plt.tight_layout()
plt.show()

import seaborn as sns

#Nilai transaksi rata-rata berdasarkan transaksi tahun pertama
sns.pointplot(data = df.groupby(['Product', 'Year_First_Transaction']).mean().reset_index(),
			  x='Year_First_Transaction',
			  y='Average_Transaction_Amount',
			  hue='Product')
plt.tight_layout()
plt.show()

#pivot data dengan pivot_table
df_piv = df.pivot_table(index='is_churn',
						columns='Product',
						values='Customer_ID',
						aggfunc='count',
						fill_value=0)

# Mendapatkan Proportion Churn by Product
plot_product = df_piv.count().sort_values(ascending=False).head(5).index

# Plot pie chartnya
df_piv = df_piv.reindex(columns=plot_product)
df_piv.plot.pie(subplots=True,
				figsize=(10, 7),
				layout=(-1, 2),
				autopct='%1.0f%%',
				title='Proportion Churn by Product')
plt.tight_layout()
plt.show()

# Kategorisasi jumlah transaksi
def func(row):
	if row['Count_Transaction'] == 1:
		val = '1. 1'
	elif (row['Count_Transaction'] > 1 and row['Count_Transaction'] <= 3):
		val ='2.2 - 3'
	elif (row['Count_Transaction'] > 3 and row['Count_Transaction'] <= 6):
		val ='3.4 - 6'
	elif (row['Count_Transaction'] > 6 and row['Count_Transaction'] <= 10):
		val ='4.7 - 10'
	else:
		val ='5.>10'
	return val

# menambah kolom count_transasction_group
df['Count_Transaction_Group'] = df.apply(func, axis=1)

#plot jumlah customer berdasarkan count_transasction_group
df_year = df.groupby(['Count_Transaction_Group'])['Customer_ID'].count()
df_year.plot(x='Count_Transaction_Group', y='Customer_ID', kind='bar', title='Customer Distribution by Count Transaction Group')
plt.xlabel('Count_Transaction_Group')
plt.ylabel('Num_of_Customer')
plt.tight_layout()
plt.show()

# Kategorisasi rata-rata besar nilai transaksi
def f(row):
	if (row['Average_Transaction_Amount'] >= 100000 and row['Average_Transaction_Amount'] <=200000):
	    val ='1. 100.000 - 250.000'
	elif (row['Average_Transaction_Amount'] >250000 and row['Average_Transaction_Amount'] <= 500000):
	    val ='2. >250.000 - 500.000'
	elif (row['Average_Transaction_Amount'] >500000 and row['Average_Transaction_Amount'] <= 750000):
	    val ='3. >500.000 - 750.000'
	elif (row['Average_Transaction_Amount'] >750000 and row['Average_Transaction_Amount'] <= 1000000):
	    val ='4. >750.000 - 1.000.000'
	elif (row['Average_Transaction_Amount'] >1000000 and row['Average_Transaction_Amount'] <= 2500000):
	    val ='5. >1.000.000 - 2.500.000'
	elif (row['Average_Transaction_Amount'] >2500000 and row['Average_Transaction_Amount'] <= 5000000):
	    val ='6. >2.500.000 - 5.000.000'
	elif (row['Average_Transaction_Amount'] >5000000 and row['Average_Transaction_Amount'] <= 10000000):
	    val ='7. >5.000.000 - 10.000.000'
	else:
	    val ='8. >10.000.000'
	return val
# Menambahkan kolom average_transaction_amount_group
df['Average_Transaction_Amount_Group'] = df.apply(f, axis=1)

#plot average_transaction_amount_group berdasarkan jumlah customer
df_year = df.groupby(['Average_Transaction_Amount_Group'])['Customer_ID'].count()
df_year.plot(x='Average_Transaction_Amount_Group', y='Customer_ID',kind='bar', title='Customer Distribution by Average Transaction Amount Group')
plt.xlabel('Average_Transaction_Amount_Group')
plt.ylabel('Num_of_Customer')
plt.tight_layout()
plt.show()