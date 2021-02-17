import pandas as pd

#panggil dataset
df = pd.read_csv('data_retail.csv', sep=';')

#mengubah kolom waktu transaksi kedalam format tanggal
df['First_Transaction']=pd.to_datetime(df['First_Transaction']/1000, unit='s', origin='1970-01-01')
df['Last_Transaction']=pd.to_datetime(df['Last_Transaction']/1000, unit='s', origin='1970-01-01')

#Klasifikasi customer churn atau tidak dan dimasukkan ke kolom is_churn
df.loc[df['Last_Transaction']<='2018-08-01', 'is_churn'] = True
df.loc[df['Last_Transaction']>'2018-08-01', 'is_churn'] = False

#menghapus kolom yang tidak diperlukan
del df['no']
del df['Row_Num']

#Tahun transaksi
df['Year_First_Transaction'] = df['First_Transaction'].dt.year
df['Year_Last_Transaction'] = df['Last_Transaction'].dt.year

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

from sklearn.preprocessing import LabelEncoder
df['is_churn'] = LabelEncoder().fit_transform(df['is_churn'])

# Feature column: Year_Diff
df['Year_Diff'] = df['Year_Last_Transaction'] - df['Year_First_Transaction']

# Nama-nama feature columns
feature_columns = ['Average_Transaction_Amount', 'Count_Transaction', 'Year_Diff']

# Features variable
X = df[feature_columns]

# Target variable
y = df['is_churn']

from sklearn.model_selection import train_test_split

#membagi dataset menjadi training 75% dan testing 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

########################Model training###################################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Inisiasi model log_model dan menerapkan ke data train
log_model=LogisticRegression().fit(X_train,y_train)

y_train_pred = log_model.predict(X_train)

# Evaluasi model train menggunakan confusion matrix
cnf_matrix = confusion_matrix(y_train, y_train_pred)
report = classification_report(y_train, y_train_pred)
print('Confusion Matrix for Training Model (Logistic Regression)\n', cnf_matrix)
print('Classification Report Training Model (Logistic Regression) :\n', report)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Evaluasi model testing menggunakan plot confusion matrix
class_names = [0, 1] # 0=Churn & 1=Tidak churn
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
plt.title('Confusion Matrix for Training Model\n(Logistic Regression)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

#######################Model Testing#################################
y_test_pred = log_model.predict(X_test)

# Evaluasi model train menggunakan confusion matrix
cnf_matrix = confusion_matrix(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred)
print('Confusion Matrix for Testing Model (Logistic Regression)\n', cnf_matrix)
print('Classification Report Testing Model (Logistic Regression) :\n', report)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Evaluasi model testing menggunakan plot confusion matrix
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
plt.title('Confusion Matrix for Testing Model\n(Logistic Regression)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()