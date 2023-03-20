from flask import Flask, render_template, jsonify, json, session, \
request, redirect, url_for
from preprocessing import lower, remove_punctuation, remove_stopwords, normalized_term, stem_text, preprocess_data, tokenize
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment import SentimentAnalyzer
from werkzeug.utils import secure_filename
from sklearn.pipeline import Pipeline
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import nltk
import csv
import os
import re
from io import BytesIO
import base64

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = 'H:\\myflask\\TA-Melia\\static\\uploads'
application.config['SECRET_KEY'] = '1234567890!@#$%^&*()'
application.config['MAX_CONTENT_PATH'] = 10000000

#<-------------------------------------------- DASHBOARD --------------------------------------->
@application.route('/dashboard')
def dashboard():
	data_testing = pd.read_excel('H://myflask/TA-Melia/static/uploads/hasil_data_keseluruhan.xlsx')
	data_testing.dropna()

	positif = data_testing.kelas.value_counts().Positif
	netral = data_testing.kelas.value_counts().Netral
	negatif = data_testing.kelas.value_counts().Negatif
	total = positif + netral + negatif

#<------------------- DASHBOARD CONFUSION MATRIX TRAINING -------------------->

	data_pelabelan = pd.read_excel('H://myflask/TA-Melia/static/uploads/data_training.xlsx')
	data_pelabelan.drop(['Unnamed: 1', 'Unnamed: 2','Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',
		'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10','Unnamed: 11','Unnamed: 12'], axis=1, inplace=True)

	X = data_pelabelan['tweet']
	Y = data_pelabelan['label']
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

	vect = TfidfVectorizer(analyzer = "word",min_df=0.0004,max_df=0.115, ngram_range=(1,3))
	vect.fit(X_train) 
	X_train_dtm = vect.transform(X_train)
	X_test_dtm = vect.transform(X_test)

	nbmodel = MultinomialNB(alpha=0.1)
	nbmodel = nbmodel.fit(X_train_dtm,Y_train)
	Y_pred = nbmodel.predict(X_test_dtm)

	img = BytesIO()
	cm = confusion_matrix(Y_test, Y_pred, labels=[1, -1, 0])
	ax = sns.heatmap(cm, xticklabels='PNN', yticklabels='PNN', annot=True, square=True, cmap='Blues')
	y = ax.set_ylabel('Kelas Actual')
	x = ax.set_xlabel('Kelas Predicted')
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	plt.yticks(rotation=0)
	plt.savefig(img, format='png')
	plt.close()
	img.seek(0)
	training = base64.b64encode(img.getvalue()).decode('utf8')

#<------------------- DASHBOARD CONFUSION MATRIX TESTING -------------------->
	akurasi = pd.read_excel('H://myflask/TA-Melia/static/uploads/hasil_testing.xlsx')
	akurasi.drop(['Unnamed: 0'], axis=1, inplace=True)

	X = akurasi['tweet']
	Y = akurasi['label']
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

	vect = TfidfVectorizer(analyzer = "word",min_df=0.0004,max_df=0.115, ngram_range=(1,3))
	vect.fit(X_train) 
	X_train_dtm = vect.transform(X_train)
	X_test_dtm = vect.transform(X_test)

	nbmodel = MultinomialNB(alpha=0.1)
	nbmodel = nbmodel.fit(X_train_dtm,Y_train)
	Y_pred = nbmodel.predict(X_test_dtm)
	
	img = BytesIO()
	akurasi = confusion_matrix(Y_test, Y_pred, labels=[1, -1, 0])
	ax = sns.heatmap(akurasi, xticklabels='PNN', yticklabels='PNN', annot=True, square=True, cmap='Greens')
	y = ax.set_ylabel('Kelas Actual')
	x = ax.set_xlabel('Kelas Predicted')
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	plt.yticks(rotation=0)
	plt.savefig(img, format='png')
	plt.close()
	img.seek(0)
	akurasi = base64.b64encode(img.getvalue()).decode('utf8')

	return render_template('dashboard.html', total=total, positif=positif, netral=netral, negatif=negatif, training=training, akurasi=akurasi)

#<-------------------------------------------- DATA TWEET UPLOAD --------------------------------------->
@application.route('/data_tweet', methods=['GET', 'POST'])
def data_tweet():
	data = []
	data = pd.DataFrame()
	if request.method == "POST":
		uploaded_file = request.files['filename'] 
		file = os.path.join(application.config['UPLOAD_FOLDER'], uploaded_file.filename)
		uploaded_file.save(file)
		try:
			data = pd.read_excel(file)
			data.drop(['Unnamed: 1','Unnamed: 2','Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',
		'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10','Unnamed: 11','Unnamed: 12'], axis=1, inplace=True)
			data.append(data)
		except:
			data = pd.read_csv(file,  on_bad_lines='skip')
			data.dropna()
			data.append(data)

	return render_template('data_tweet.html', data=[data.to_html(justify='center', classes=['table-striped', 'table-bordered', 'dt-responsive', 'table-style'], table_id='example')])

#<-------------------------------------------- PREPROCESSING DATA --------------------------------------->
@application.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
	if request.method == 'POST':
		data = pd.read_csv('H://myflask/TA-Melia/static/uploads/hasil_15April.csv', names=['tanggal', 'tweet'], encoding='latin-1')
		data.dropna()
		data.drop(['tanggal'], axis=1, inplace=True)

		data['tweet'] = data['tweet'].map(lambda x: lower(x))
		data['tweet'] = data['tweet'].map(lambda x: remove_punctuation(x))
		data['tweet'] = data['tweet'].map(lambda x: normalized_term(x))
		data['tweet'] = data['tweet'].map(lambda x: remove_stopwords(x))
		data['tweet'] = data['tweet'].map(lambda x: stem_text(x))
		data['tweet'] = data['tweet'].map(lambda x: tokenize(x))

		data = data.drop_duplicates(subset=['tweet'], keep=False)
		data.to_excel('H://myflask/TA-Melia/static/uploads/hasil_preprocessing.xlsx')

	data_preprocessing = pd.read_excel('H://myflask/TA-Melia/static/uploads/hasil_preprocessing.xlsx')
	data_preprocessing.dropna()
	data_preprocessing.drop(['Unnamed: 0'], axis=1, inplace=True)

	return render_template('preprocessing.html', data_preprocessing=[data_preprocessing.to_html(index=False, justify='center', classes=['table-striped', 'table-bordered', 'dt-responsive', 'table-style'], table_id='example')])
	
#<-------------------------------------------- PELABELAN MANUAL --------------------------------------->
@application.route('/pelabelan_manual', methods=['GET', 'POST'])
def pelabelan_manual():
	data_pelabelan = pd.read_excel('H://myflask/TA-Melia/static/uploads/data_training.xlsx')
	data_pelabelan.dropna()
	data_pelabelan.drop(['Unnamed: 1', 'Unnamed: 2','Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',
		'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10','Unnamed: 11','Unnamed: 12'], axis=1, inplace=True)
	data_pelabelan.drop_duplicates(['tweet'])

#<-------------------------------------------- AKURASI TRAINING DATA --------------------------------------->
	X = data_pelabelan['tweet']
	Y = data_pelabelan['label']
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

	vect = TfidfVectorizer(analyzer = "word",min_df=0.0004,max_df=0.115, ngram_range=(1,3))
	vect.fit(X_train) 
	X_train_dtm = vect.transform(X_train)
	X_test_dtm = vect.transform(X_test)

	nbmodel = MultinomialNB(alpha=0.1)
	nbmodel = nbmodel.fit(X_train_dtm,Y_train)
	Y_pred = nbmodel.predict(X_test_dtm)

#<-------------------------------------------- CONFUSION MATRIX DATA TRAINING --------------------------------------->
	akurasi = accuracy_score(Y_test, Y_pred) * 100
	f1score = f1_score(Y_test, Y_pred, average='weighted') * 100
	presision = precision_score(Y_test, Y_pred, average='weighted') * 100
	recall = recall_score(Y_test, Y_pred, average='weighted') * 100

	return render_template('pelabelan_manual.html', data_pelabelan=[data_pelabelan.to_html(index=False, justify='center', classes=['table-striped', 'table-bordered', 'table-hover', 'table-condensed', 'dt-responsive', 'table-style'], table_id='example')], accuracy_score=akurasi, f1_score=f1score, precision_score=presision, recall_score=recall)
	
#<-------------------------------------------- TRAINING DATA --------------------------------------->
@application.route('/training', methods=['GET', 'POST'])
def training():
	data_testing = pd.read_excel('H://myflask/TA-Melia/static/uploads/data_training.xlsx')
	data_testing.dropna()
	data_testing.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',
		'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10','Unnamed: 11','Unnamed: 12'], axis=1, inplace=True)

	X = data_testing['tweet']
	y = data_testing['label']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

	vectorizer = TfidfVectorizer()

	X_train = vectorizer.fit_transform(data_testing['tweet'])
	X_test = vectorizer.transform(X_test)

	return render_template('testing_tfidf.html', y=y, X=X, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

@application.route('/testing_tfidf', methods=['GET', 'POST'])
def testing_tfidf():
	data_testing = pd.read_excel('H://myflask/TA-Melia/static/uploads/data_training.xlsx')
	data_testing.dropna()
	data_testing.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',
		'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10','Unnamed: 11','Unnamed: 12'], axis=1, inplace=True)

	bow_transformer = CountVectorizer().fit(data_testing['tweet'])
	messages_bow = bow_transformer.transform(data_testing['tweet'])

	tfidf_transformer = TfidfTransformer().fit(messages_bow)

	messages_tfidf = tfidf_transformer.transform(messages_bow)

	pipeline = Pipeline([
		('bow', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('classifier', MultinomialNB())
		])

	X_train = np.asarray(data_testing['tweet'])
	pipeline = pipeline.fit(X_train, np.asarray(data_testing['label']))

	file_data = pickle.dump(pipeline, open('H://myflask/TA-Melia/static/uploads/latihan_ta.pickle', 'wb'))

	return render_template('testing_tfidf.html', pipeline=pipeline)

#<-------------------------------------------- DATA TESTING --------------------------------------->
@application.route('/testing', methods=['GET', 'POST'])
def testing():
	import pickle
	vect = pickle.load(open('H://myflask/TA-Melia/static/uploads/latihan_ta.pickle', 'rb'))

	testing = pd.read_excel('H://myflask/TA-Melia/static/uploads/testing.xlsx')
	testing.dropna()
	testing.drop(['Unnamed: 0','Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',
		'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10','Unnamed: 11','Unnamed: 12'], axis=1, inplace=True)
	
	testing = testing['tweet'].fillna(' ')
	prediction = vect.predict(testing)

	result = []

	for i in range(len(prediction)):
		if(prediction[i] == 1):
			sentiment = 'Positif'
		elif(prediction[i]==0):
			sentiment = 'Netral'
		else:
			sentiment = 'Negatif'

		result.append({'tweet':testing[i],'label':prediction[i], 'kelas':sentiment })

	testing = pd.DataFrame(result)
	testing = testing.dropna()
	testing.to_excel('H://myflask/TA-Melia/static/uploads/hasil_testing.xlsx')

#<-------------------------------------------- AKURASI DATA TESTING --------------------------------------->
	akurasi = pd.read_excel('H://myflask/TA-Melia/static/uploads/hasil_testing.xlsx')
	akurasi.drop(['Unnamed: 0'], axis=1, inplace=True)

	X = akurasi['tweet']
	Y = akurasi['label']
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

	vect = TfidfVectorizer(analyzer = "word",min_df=0.0004,max_df=0.115, ngram_range=(1,3))
	vect.fit(X_train) 
	X_train_dtm = vect.transform(X_train)
	X_test_dtm = vect.transform(X_test)

	nbmodel = MultinomialNB(alpha=0.1)
	nbmodel = nbmodel.fit(X_train_dtm,Y_train)
	Y_pred = nbmodel.predict(X_test_dtm)
	
	akurasi = accuracy_score(Y_test, Y_pred) * 100
	measure = f1_score(Y_test, Y_pred, average='weighted') * 100
	presision = precision_score(Y_test, Y_pred, average='weighted') * 100
	recall = recall_score(Y_test, Y_pred, average='weighted') * 100

	return render_template('testing.html', testing=[testing.to_html(index=False, justify='center', classes=['table-striped', 'table-bordered', 'table-hover', 'table-condensed', 'dt-responsive', 'table-style'], table_id='example')], accuracy_score=akurasi, f1_score=measure, precision_score=presision, recall_score=recall)

#<-------------------------------------------- KLASIFIKASI NAIVE BAYES --------------------------------------->
@application.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
	import pickle
	vect = pickle.load(open('H://myflask/TA-Melia/static/uploads/latihan_ta.pickle', 'rb'))

	klasifikasi = pd.read_excel('H://myflask/TA-Melia/static/uploads/data_tweet_keseluruhan.xlsx')
	klasifikasi.dropna()
	klasifikasi.drop(['Unnamed: 0','Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',
		'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13'], axis=1, inplace=True)
	
	klasifikasi = klasifikasi['tweet'].fillna(' ')
	prediction = vect.predict(klasifikasi)

	result = []

	for i in range(len(prediction)):
		if(prediction[i] == 1):
			sentiment = 'Positif'
		elif(prediction[i]==0):
			sentiment = 'Netral'
		else:
			sentiment = 'Negatif'

		result.append({'tweet':klasifikasi[i],'label':prediction[i], 'kelas':sentiment })

	klasifikasi = pd.DataFrame(result)
	klasifikasi = klasifikasi.dropna()
	klasifikasi.to_excel('H://myflask/TA-Melia/static/uploads/hasil_data_keseluruhan.xlsx')

	return render_template('klasifikasi_naive_bayes.html', klasifikasi=[klasifikasi.to_html(index=False, justify='center', classes=['table-striped', 'table-bordered', 'table-hover', 'table-condensed', 'dt-responsive', 'table-style'], table_id='example')])

if __name__ == '__main__':
	application.run(debug=True)
