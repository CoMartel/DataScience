import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import itertools
import copy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
#from sklearn.neural_network import MLPClassifier
from sklearn import svm
import datetime
pd.set_option('display.max_columns', None)
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from operator import itemgetter
import time

############################################################################
# Function to Write result in csv file to submit 
###########################################################################

def write_to_csv(output,score):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    prediction_file_object = csv.writer(f)
    prediction_file_object.writerow(["activity_id","outcome"])  # don't forget the headers

    for i in range(len(df_test)):
        prediction_file_object.writerow([df_test["activity_id"][df_test.index[i]], (output[i])])


############################################################################
# Function to process features 
###########################################################################
def get_features(train, test):
    trainval = list(train.columns.values) # list train features
    testval = list(test.columns.values) # list test features
    output = list(set(trainval) & set(testval)) # check wich features are in common (remove the outcome column)
    output.remove('people_id') # remove non-features 
    output.remove('activity_id')
    
    # Keep only the most usefull features :
    #clf = ExtraTreesClassifier(n_estimators=200)
    #clf = clf.fit(train[output][::500], train['outcome'][::500])
    
    features = pd.DataFrame()
    features['feature'] = output
    #features['importance'] = clf.feature_importances_
    #features=features.sort_values(by='importance',ascending=False)
    #print "most important features : ",features.head(25)
    
    return features #[:25] # keep the best 25 features


def process_features(train,test,people):
    tables=[train,test]
    for i,table in enumerate(tables): 
        print("Cleaning the dataframes : {} / {}").format(i+1,len(tables))
        # clean activity id
        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)

        for i in range(1, 11):
            table['char_' + str(i)].fillna('type -999', inplace=True) # replace nan by -999
            table['char_' + str(i)] = table['char_' + str(i)].str.lstrip('type ').astype(np.int32)
        # separate date in Y , M , D and add day of week, since weekend seems important
        table['date'] = pd.to_datetime(table['date'])
        table['year'] = table['date'].dt.year # create a column for year, month and day
        table['month'] = table['date'].dt.month
        table['day'] = table['date'].dt.day
        table['day_of_week'] = table['date'].dt.dayofweek
        table.drop('date', axis=1, inplace=True) # delete the date column
    
    print "Cleaning the people dataframe"
    people['date'] = pd.to_datetime(people['date'])
    people['year'] = people['date'].dt.year # create a column for year, month and day
    people['month'] = people['date'].dt.month
    people['day'] = people['date'].dt.day
    people['day_of_week'] = people['date'].dt.dayofweek
    people.drop('date', axis=1, inplace=True) # delete the date column
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    for i in range(1, 10):
        people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    for i in range(10, 38):
        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)
    
    print("Merge with the people dataframe...")
    train = pd.merge(train, people, how='left', on='people_id', left_index=True)
    train.fillna(-999, inplace=True)
    test = pd.merge(test, people, how='left', on='people_id', left_index=True)
    test.fillna(-999, inplace=True)
    
    #titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    #combined = pd.concat([combined,titles_dummies],axis=1)
    
    # removing the title variable
    #combined.drop('Title',axis=1,inplace=True)
    
    print "Getting features..."
    features = get_features(train,test)
    

    return train,test,features

def train_and_test(train,test,features,target='outcome'): # simple xgboost
    eta_list = [0.2]
    max_depth_list = [6]
    subsample = 0.8
    colsample_bytree = 0.8
    
    num_boost_round = 100 #115 originally 
    early_stopping_rounds = 10
    test_size = 0.2 # 0.1 originally
    
    start_time = time.time()
   

    # start the training
    array_score=np.ndarray((len(eta_list)*len(max_depth_list),3)) # store score values
    i=0
    for eta,max_depth in list(itertools.product(eta_list, max_depth_list)):
        print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
        params = {
            "objective": "binary:logistic",
            "booster" : "gbtree",# default
            "eval_metric": "auc",
            "eta": eta, # shrinking parameters to prevent overfitting
            "tree_method": 'exact',
            "max_depth": max_depth,
            "subsample": subsample, # collect 80% of the data only to prevent overfitting
            "colsample_bytree": colsample_bytree,
            "silent": 1,
            "seed": 0,
        }
    
        X_train, X_valid = train_test_split(train, test_size=test_size, random_state=0) # randomly split into 90% test and 10% CV -> still has the outcome at this point
        y_train = X_train[target] # define y as the outcome column
        y_valid = X_valid[target]
        dtrain = xgb.DMatrix(X_train[features], y_train) # DMatrix are matrix for xgboost
        dvalid = xgb.DMatrix(X_valid[features], y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')] # list of things to evaluate and print
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True) # find the best score

        print("Validating...")
        check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_ntree_limit) # get the best score
        score = roc_auc_score(X_valid[target].values, check)
        print('Check error value: {:.6f}'.format(score))
        array_score[i][0]=eta
        array_score[i][1]=max_depth
        array_score[i][2]=score
        i+=1
    df_score=pd.DataFrame(array_score,columns=['eta','max_depth','score'])
    print "df_score : \n", df_score
    #create_feature_map(features)
    importance = gbm.get_fscore()
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    print('Importance array: ', importance)
    np.save("features_importance",importance)
    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_ntree_limit)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    
    #xgb.plot_tree(gbm)
    #plt.show()
    return test_prediction, score    
    
def train_and_test_Kfold(train,test,features,target='outcome'): # add Kfold
    eta_list = [0.15]
    max_depth_list = [6,8]
    subsample = 0.75
    colsample_bytree = 0.75
    
    num_boost_round = 400 #115 originally 
    early_stopping_rounds = 30
    
    start_time = time.time()
   

    # start the training
    array_score=np.ndarray((len(eta_list)*len(max_depth_list),4)) # store score values
    i=0
    for eta,max_depth in list(itertools.product(eta_list, max_depth_list)):
        print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
        params = {
            "objective": "binary:logistic",
            "booster" : "gbtree",# default
            "eval_metric": "auc",
            "eta": eta, # shrinking parameters to prevent overfitting
            "tree_method": 'exact',
            "max_depth": max_depth,
            "subsample": subsample, # collect 80% of the data only to prevent overfitting
            "colsample_bytree": colsample_bytree,
            "silent": 1,
            "seed": 0,
        }
        kf = KFold(len(train), n_folds=5)
        test_prediction=np.ndarray((5,len(test)))
        fold=0
        fold_score=[]
        for train_index, cv_index in kf:
            X_train, X_valid    = train[features].as_matrix()[train_index], train[features].as_matrix()[cv_index]
            y_train, y_valid    = train[target].as_matrix()[train_index], train[target].as_matrix()[cv_index]

            #y_train = X_train[target] # define y as the outcome column
            #y_valid = X_valid[target]
            dtrain = xgb.DMatrix(X_train, y_train) # DMatrix are matrix for xgboost
            dvalid = xgb.DMatrix(X_valid, y_valid)

            watchlist = [(dtrain, 'train'), (dvalid, 'eval')] # list of things to evaluate and print
            gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True) # find the best score

            print("Validating...")
            check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_ntree_limit) # get the best score
            score = roc_auc_score(y_valid, check)
            print('Check error value: {:.6f}'.format(score))
            fold_score.append(score)
            importance = gbm.get_fscore()
            importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
            print('Importance array for fold {} :\n {}').format(fold, importance)
            #np.save("features_importance",importance)
            print("Predict test set...")
            prediction=gbm.predict(xgb.DMatrix(test[features].as_matrix()), ntree_limit=gbm.best_ntree_limit)
            np.save("prediction_eta%s_depth%s_fold%s" %(eta,max_depth,fold),prediction)
            test_prediction[fold]=prediction
            fold = fold + 1
        mean_score=np.mean(fold_score)
        print("Mean Score : {}, eta : {}, depth : {}\n").format(mean_score,eta,max_depth)
        array_score[i][0]=eta
        array_score[i][1]=max_depth
        array_score[i][2]=mean_score
        array_score[i][3]=np.std(fold_score)
        i+=1
    final_prediction=test_prediction.mean(axis=0)
    df_score=pd.DataFrame(array_score,columns=['eta','max_depth','mean_score','std_score'])
    print "df_score : \n", df_score
    #create_feature_map(features)


    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    
    #xgb.plot_tree(gbm)
    #plt.show()
    return final_prediction, mean_score 


############################################################################
# Read files and merge  
###########################################################################


df_train = pd.read_csv("act_train.csv")
df_test = pd.read_csv("act_test.csv")

df_people =  pd.read_csv("people.csv")
#train = pd.merge(df_train, df_people, on='people_id', how='right')
#test = pd.merge(df_test, df_people, on='people_id', how='right')
train,test,features = process_features(df_train,df_test,df_people)

test_prediction,score = train_and_test(train,test,features['feature'])
features=np.load("features_importance.npy")
test_prediction,score = train_and_test_Kfold(train,test,features[:15,0])

write_to_csv(test_prediction,score)

