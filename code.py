
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

df = pd.read_excel("train_data.xlsx",usecols= [0,1])
target_name = df['Domains+Events'].unique().tolist()
df['Id'] = df['Domains+Events'].factorize()[0]


category_id_df = df[['Domains+Events', 'Id']].drop_duplicates().sort_values('Id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['Id', 'Domains+Events']].values)


stopword = text.ENGLISH_STOP_WORDS.difference(["AI", "ai"])

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words= stopword, token_pattern = r"(?u)c\+{2}|\b\w+\b")
features = tfidf.fit_transform(df['Event Names'].values)
labels = df['Domains+Events']

for product, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    uni = [v for v in feature_names if len(v.split(' ')) == 1]
    bi = [v for v in feature_names if len(v.split(' ')) == 2]
    #print("# '{}':".format(Product))
    #print("  . Most correlated unigrams:\n. {}".format('\n. '.join(uni[-2:])))
    #print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bi[-2:])))

from sklearn.svm import LinearSVC
svc = LinearSVC()
xtrain, xtest, ytrain, ytest, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.01, random_state=0)


svc.fit(xtrain, ytrain)
y_pred = svc.predict(xtest)

svc = LinearSVC()
svc.fit(features, labels)



for Product, category_id in sorted(category_to_id.items()):
    indices = np.argsort(svc.coef_[category_id])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    uni = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:2]
    bi = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:2]
    

def query_ans(event_dom, event_type, employees):
    return employees.query("Domain == '" + event_dom + "' and (Event1 == '" + event_type +"' or Event2 == '" + event_type + "')")


def predict(inp_eve, employees):
        recommendations = []
        prediction = svc.predict(tfidf.transform(inp_eve))  
        for text, predicted in zip(inp_eve, prediction):
            print('"{}"'.format(text))
            print("  - Predicted as: '{}'".format(predicted))
            print("")
        for predictions in prediction.tolist():
            domain, event_type = predictions.split(".")
            if domain == 'Artificial_Intelligence':
                recommend_to = query_ans('Artificial Intelligence', event_type, employees)

            elif domain == 'Data_Science':
                recommend_to = query_ans('Data Science', event_type, employees)

            elif domain == 'CC':
                recommend_to = query_ans('Cloud Computing', event_type, employees)

            elif domain == 'WebDev':
                recommend_to = query_ans('Web Development', event_type, employees)
                
            elif domain == 'Mobile_Applications':
                recommend_to = query_ans('Mobile Applications', event_type, employees)

            elif domain == 'Software_Architecture':
                recommend_to = query_ans('Software Architecture', event_type, employees)

            elif domain == 'ML':
                recommend_to = query_ans('Machine Learning', event_type, employees)
                
            elif domain == 'Higher_Education':
                recommend_to = query_ans('Higher Education', event_type, employees)
                
            elif domain == 'DevOps':
                recommend_to = query_ans('Development Processes', event_type, employees)

            elif domain == 'Cpp':
                recommend_to = query_ans('C++', event_type, employees)
                
            elif domain == 'None':
                recommend_to = employees.query("Event1 == '" + event_type + "' or Event2 == '" + event_type + "'")
                
            else:
                recommend_to = query_ans(domain, event_type, employees)
                
            recommendations.append(", ".join(recommend_to['Name'].values))
            
        return recommendations

def create_excel():
    path=input("Enter name of input file")
    employees = pd.read_csv("CCMLEmployeeData.csv")
    to_pred_events = pd.read_csv(path, encoding= 'unicode_escape')
    recommendations = predict(to_pred_events.Events, employees)
    to_pred_events['Employees'] = recommendations
    to_pred_events.to_excel('result.xlsx', index=False)
create_excel()






