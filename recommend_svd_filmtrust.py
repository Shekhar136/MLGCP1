import pandas as pd
import numpy as np
import surprise
import os

os.chdir(r"C:\Training\Academy\Statistics (Python)\20. Recommender Systems\filmtrust")
ratings = pd.read_csv("ratings.txt",sep=' ',names = ['uid','iid','rating'])
ratings.head()

lowest_rating = ratings['rating'].min()
highest_rating = ratings['rating'].max()

print("Ratings range between {0} and {1}".format(lowest_rating,highest_rating))

reader = surprise.Reader(rating_scale = (lowest_rating,highest_rating))
data = surprise.Dataset.load_from_df(ratings,reader)

algo = surprise.SVD(random_state=2022)
output = algo.fit(data.build_full_trainset())

pred = algo.predict(uid='50',iid='5')
score = pred.est
print(score)

iids = ratings['iid'].unique()
iids50 = ratings.loc[ratings['uid'] == 50 ,'iid']
print("List of iid that uid={0} has rated:".format(50))
print(iids50)

iids_to_predict = np.setdiff1d(iids,iids50)
print("List of iid which uid={0} did not rate(in all {1}) :".format(50,len(iids_to_predict)))
print(iids_to_predict)

### ratings arbitrarily set to 0
testset = [[50,iid,0.] for iid in iids_to_predict]
predictions = algo.test(testset)
predictions[0]

pred_ratings = np.array([pred.est for pred in predictions])

# Finding the index of maximum predicted rating
i_max = pred_ratings.argmax()

# Recommending the item with maximum predicted rating
iid_recommend_most = iids_to_predict[i_max] 
print("Top item to be recommended for user {0} is {1} with predicted rating as {2}".format(50,iid_recommend_most,pred_ratings[i_max]))

# Getting top 10 items to be recommended for uid = 50
import heapq
i_sorted_10 = heapq.nlargest(10, range(len(pred_ratings)), pred_ratings.take)
top_10_items = iids_to_predict[i_sorted_10]
print(top_10_items)

#################### tuning ####################
from surprise.model_selection import GridSearchCV
param_grid = {'n_epochs': np.arange(5,50,10), 
              'lr_all':np.linspace(0.001,1,5),
              'reg_all': np.linspace(0.01,0.8,5)}

from surprise.model_selection.split import KFold
kfold = KFold(random_state=2022,n_splits=5,shuffle=True)
gs = GridSearchCV(surprise.SVD, param_grid,joblib_verbose =3, 
                  measures=['rmse', 'mae'], cv=kfold,n_jobs=-1)

gs.fit(data)
# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])


# We can now use the algorithm that yields the best rmse:
algo_best = gs.best_estimator['rmse']
#algo.fit(data.build_full_trainset())

########## storing the model in the form of a file #######
os.chdir(r"C:\Training\Academy\Statistics (Python)\20. Recommender Systems")
surprise.dump.dump("filmtrust_SVD_Model", 
                   algo=algo_best, verbose=0)

########## loading the stored model from the file ########
loaded = surprise.dump.load("filmtrust_SVD_Model")
model = loaded[1]

pred = model.predict(uid='50',iid='5')
score = pred.est
print(score)

### ratings arbitrarily set to 0
testset = [[50,iid,0.] for iid in iids_to_predict]
predictions = model.test(testset)
predictions[0]

iids = ratings['iid'].unique()
iids50 = ratings.loc[ratings['uid'] == 50 ,'iid']
print("List of iid that uid={0} has rated:".format(50))
print(iids50)

iids_to_predict = np.setdiff1d(iids,iids50)
print("List of iid which uid={0} did not rate(in all {1}) :".format(50,len(iids_to_predict)))
print(iids_to_predict)
pred_ratings = np.array([pred.est for pred in predictions])

# Finding the index of maximum predicted rating
i_max = pred_ratings.argmax()

# Recommending the item with maximum predicted rating
iid_recommend_most = iids_to_predict[i_max] 
print("Top item to be recommended for user {0} is {1} with predicted rating as {2}".format(50,iid_recommend_most,pred_ratings[i_max]))

