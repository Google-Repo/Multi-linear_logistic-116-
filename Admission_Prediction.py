import pandas as pd
import plotly.express as px

df = pd.read_csv("Admission_Predict.csv")

TOEFL_score_list = df["TOEFL Score"].tolist()
GRE_score_list = df["GRE Score"].tolist()

fig = px.scatter(x= TOEFL_score_list, y= GRE_score_list)
fig.show()

import plotly.graph_objects as go

TOEFL_score_list = df["TOEFL Score"].tolist()
GRE_score_list = df["GRE Score"].tolist()
chance_of_admission = df["Chance of admit"].tolist()

colors = []

for data in chance_of_admission:
    if data == 1:
        colors.append("green")
    else:
        colors.append("red")
    
fig = go.Figure(data=go.Scatter(
    x=TOEFL_score_list,
    y=GRE_score_list,
    mode='markers',
    marker=dict(color=colors)
))
fig.show()

scores = df[["TOEFL Score","GRE Score"]]

chance_of_admission = df['Chance of admit']

from sklearn.model_selection import train_test_split

scores_train, scores_test, chance_of_admission_train, chance_of_admission_test = train_test_split(scores, chance_of_admission, test_size =0.25, random_state=0)
print(scores_train)

from sklearn.linear_model import LogisticsRegression

classifier = LogisticsRegression(random_state=0)
classifier.fit(scores_train, chance_of_admission)

chance_of_admission_pred = classifier.predict(scores_test)

from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(chance_of_admission_test, chance_of_admission_pred))
