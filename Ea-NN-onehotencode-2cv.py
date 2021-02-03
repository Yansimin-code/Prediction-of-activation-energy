import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, hist, boxplot,  title, subplot, plot, show, axis, xlabel, ylabel, ylim, xlim, legend
import numpy as np
import xlrd
from scipy import stats
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network  import MLPRegressor
from sklearn.metrics  import mean_squared_error, mean_absolute_error
from sklearn.model_selection import learning_curve, train_test_split

doc = xlrd.open_workbook('./Eads-yanlan.xlsx').sheet_by_index(0)
#.sheet_by_index mean get the first sheet by index

#get attribute name(i.e. Ea and H)
attributeNames = doc.row_values(0,0,2)

#print(attributeNames)
#the first classlabels is the surface
surfacedata = doc.col_values(2,231,584)
surface = np.array(surfacedata).reshape(-1,1)


# onehotencode
enc = OneHotEncoder(handle_unknown='ignore')
surfacelist = enc.fit_transform(surface).toarray()
surfaceindex= enc.categories_


#define fingerprint for the surface by using its position in perodical table(row, column, d-band center)
#d band center from page 86 in Theoretical surface science and catalysis calculations and concepts

elemdict={'Co(0001)':[5,9,-1.17],
          'Ru(0001)':[5,8,-1.41],
          'Fe(110)':[4,8,-0.92],
          'Ni(111)':[4,10,-1.29],
          'Cu(111)':[4,11,-2.67],
          'Pd(111)':[5,10,-1.83],
          'Ag(111)':[5,11,-4.30],
          'Au(111)':[6,11,-3.56],
          'Rh(111)':[5,9,-1.73],
          'Pt(111)':[6,10,-2.25]}
elemlist = np.asarray([elemdict[value] for value in surfacedata])

print(elemlist)

dbanddict={'Co(0001)':[-1.17],
          'Ru(0001)':[-1.41],
          'Fe(110)':[-0.92],
          'Ni(111)':[-1.29],
          'Cu(111)':[-2.67],
          'Pd(111)':[-1.83],
          'Ag(111)':[-4.30],
          'Au(111)':[-3.56],
          'Rh(111)':[-1.73],
          'Pt(111)':[-2.25]}
dbandlist = np.asarray([dbanddict[value] for value in surfacedata])
#get the second-fours class label , which is ab*

abdata = doc.col_values(3,231,584)
ab = np.array(abdata).reshape(-1,1)
ablist = enc.fit_transform(ab).toarray()
abindex = enc.categories_

adata = doc.col_values(4,231,584)
a = np.array(adata).reshape(-1,1)
alist = enc.fit_transform(a).toarray()
aindex = enc.categories_

bdata = doc.col_values(5,231,584)
b = np.array(bdata).reshape(-1,1)
blist = enc.fit_transform(b).toarray()
bindex = enc.categories_
#
#recovered_b=  np.array([enc.active_features_[col] for col in blist.sorted_indices().indices]).reshape(n_samples, n_features)-enc.feature_indices_[:-1]
#print(recovered_b)

y = np.asarray(doc.col_values(1, 231, 584)).reshape(-1,1)

print(len(y))

x1 = np.asarray(doc.col_values(0, 231, 584)).reshape(-1,1)



x2 = np.hstack((x1, surfacelist))
print(x2.shape)

#remove surfacelist
x3 = np.hstack((x1, ablist))

x4 = np.hstack((x3, alist))

x5 = np.hstack((x4, blist))

x6 = np.hstack((x5,elemlist))

X = np.hstack((x5,dbandlist))
print(X.shape)

y=y.ravel()

print(surfaceindex, abindex, aindex, bindex)
#K-fold crossvalidation
K_inner = 10
K_outer = 10
inner_cv = model_selection.KFold(n_splits=K_inner, shuffle=True,random_state=42)
outer_cv = model_selection.KFold(n_splits=K_outer, shuffle=True,random_state=42)
K = K_inner

# Initialize variables-mean square error
Error_train_NN = np.empty((K,1))
Error_cv_NN = np.empty((K,1))
Root_mean_square_error_NN_cv = np.empty((K,1))
mean_absolute_error_train_NN = np.empty((K,1))
mean_absolute_error_cv_NN = np.empty((K,1))
mae_test_nn = np.empty((K,1))
rmse_test_nn = np.empty((K,1))
mse_test_nn = np.empty((K,1))

# # forward feature selection
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
#
# reg_NN = MLPRegressor(hidden_layer_sizes=(25), alpha=2.5, activation="tanh", solver='lbfgs',
#                               max_iter=20000, tol=0.000001, shuffle=True, warm_start=True,random_state=20)
# sfs1 = SFS(reg_NN,
#            k_features=1,
#            forward=True,
#            floating=False,
#            verbose=2,
#            scoring='neg_mean_absolute_error',
#            cv=10)
#
# sfs1 = sfs1.fit(X, y)
#
# #look at the selected feature indices at each step
# print(sfs1.subsets_)
#
# # Which features?
# feat_cols = list(sfs1.k_feature_idx_)
# print(feat_cols)
#
# fig = plot_sfs(sfs1.get_metric_dict(), kind='std_err')
#
# plt.title('Sequential Forward Selection (w. StdDev)')
# plt.rcParams.update({'font.size': 20})
# plt.xlabel('Mean Absolute error(eV)')
# plt.text(7,-1, r'NN-1',{'color':'k','fontsize':18})


# Neural network
#first split the data into train+cv and test
k_outer = 0
for train_index, test_index in outer_cv.split(X):
    print('Computing CV fold:{0}/{1}..'.format(k_outer + 1, K))
    X_train, Y_train = X[train_index, :], y[train_index]
    x_test, y_test = X[test_index, :], y[test_index]
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    x_test = scaler.transform(x_test)
    #use the following code to selcect the parameters
    # set up the possible values of parametrs to potimize over
    #p_grid_nn = {'hidden_layer_sizes':[10,15,20,22,25,28,30,35,40,45,50],'alpha':[0.01,0.1,0.2,0.5,0.8,1,1.2,1.5,1.8,
    #2,2.2,2.5,2.8,3,5,10],'activation':('tanh','relu')}
    #cv_best_params_nn: {'activation': 'tanh', 'alpha': 2, 'hidden_layer_sizes': 45} for the first fold
    # sepecify the estimator
    #nn = MLPRegressor( solver='lbfgs',max_iter=20000, tol=0.000001,
     #                  shuffle=True, warm_start=True,random_state=20)
    #clf_nn = GridSearchCV(nn,p_grid_nn,cv=5,return_train_score=True,
     #                     scoring='neg_mean_absolute_error')
    #clf_nn.fit(X_train, Y_train)
    #print('cv result_nn:{0}'.format(clf_nn.cv_results_))
    #print('cv_bestscore_nn:{0}'.format(clf_nn.best_score_))
    #print('cv_best_params_nn:{0}'.format(clf_nn.best_params_))
    k_inner = 0
    for train_index, test_index in inner_cv.split(X_train):
        #print('Computing CV fold:{0}/{1}..'.format(k+1,K))
        # extract training and test set for current CV fold
        x_train, y_train = X_train[train_index,:], Y_train[train_index]
        x_cv, y_cv = X_train[test_index,:], Y_train[test_index]

        # Train the net on training data
        reg_NN = MLPRegressor(hidden_layer_sizes=(25), alpha=2.5, activation="tanh", solver='lbfgs',
                              max_iter=20000, tol=0.000001, shuffle=True, warm_start=True,random_state=20)
        reg_NN = reg_NN.fit(x_train, y_train)

        Error_train_NN[k_inner] = np.square(y_train - reg_NN.predict(x_train)).sum() / y_train.shape[0]
        Error_cv_NN[k_inner] = np.square(y_cv - reg_NN.predict(x_cv)).sum() / y_cv.shape[0]
        Root_mean_square_error_NN_cv[k_inner] = np.sqrt(Error_cv_NN[k_inner])
        mean_absolute_error_train_NN[k_inner] = mean_absolute_error(y_train, reg_NN.predict(x_train))
        mean_absolute_error_cv_NN[k_inner] = mean_absolute_error(y_cv, reg_NN.predict(x_cv))
        k_inner += 1

    reg_NN = reg_NN.fit(X_train, Y_train)

    mae_test_nn[k_outer] = mean_absolute_error(y_test, reg_NN.predict(x_test))
    mse_test_nn[k_outer] =mean_squared_error(y_test, reg_NN.predict(x_test))
    rmse_test_nn[k_outer] = np.sqrt(mse_test_nn[k_outer])
    k_outer += 1

print('MSE_NN_train:{0}'.format(Error_train_NN.mean()))
print('MSE_NN_cv:{0}'.format(Error_cv_NN.mean()))
print('RMSE_NN_cv:{0}'.format(Root_mean_square_error_NN_cv.mean()))
print('MAE_NN_train:{0}'.format(mean_absolute_error_train_NN.mean()))
print('MAE_NN_cv:{0}'.format(mean_absolute_error_cv_NN.mean()))
print(reg_NN.score(x_train, y_train), reg_NN.score(x_cv, y_cv))
print('mae_test_nn:{0}'.format(mae_test_nn.mean()))
print('mse_test_nn:{0}'.format(mse_test_nn.mean()))
print('rmse_test_nn:{0}'.format(rmse_test_nn.mean()))

scaler = StandardScaler()
scaler = scaler.fit(X)
X = scaler.transform(X)
y_predict = reg_NN.predict(X)
print(mean_absolute_error(y, reg_NN.predict(X)))

#export the predicted y to excel
import pandas as pd
data = pd.DataFrame(y_predict)
print(data)
#datatoexcel = pd.ExcelWriter('y_NN.xlsx')
data.to_excel('yNN.xlsx', sheet_name='sheet 1')

#save the model to disk
import pickle
filename = 'regNN.sav'
pickle.dump(reg_NN,open(filename,'wb'))

#load the model from disk
loaded_model = pickle.load(open(filename,'rb'))



train_error_nn = y_train - reg_NN.predict(x_train)
ind = np.argmax(train_error_nn)
print(np.amax(train_error_nn),np.amin(train_error_nn), np.argmax(train_error_nn))

test_error_nn = y_test-reg_NN.predict(x_test)
print(np.amax(test_error_nn),np.amin(test_error_nn), np.argmax(test_error_nn))

#train error histogram
figure()
residual = reg_NN.predict(x_train)-y_train
hist(residual,10, label= 'Artifical_Neural_Network')
xlabel('error(eV)')

figure()
plt.rcParams.update({'font.size': 20})
subplot(1,2,1)
plot(y_train, reg_NN.predict(x_train), 'bo',label = 'train_set')
plot(y_test, reg_NN.predict(x_test), 'rh',label = 'test_set')
xlim(-1,5)
ylim(-1,5)
plt.text(3.5,4.5, r'NN-1',{'color':'k','fontsize':18})
plt.text(2.8,-0.5,'MAE={:.2f}'.format(mae_test_nn.mean()),{'color':'k','fontsize':18})
plt.text(2.8,-0.8,'RMSE={:.2f}'.format(rmse_test_nn.mean()),{'color':'k','fontsize':18})
xlabel('DFT calculated barrier(eV)');
ylabel('Predicted barrier(eV)');
plot([-1, 5],[-1, 5],ls="--", c=".3")

subplot(1,2,2)
residual = reg_NN.predict(x_test)-y_test
hist(residual,10, label= 'Artifical_Neural_Network')
xlabel('error(eV)')

figure()

residual = reg_NN.predict(x_train)-y_train
hist(residual,10, label= 'NN1-train')
xlabel('train error(eV)')

show()




