import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, hist, boxplot,  title, subplot, plot, show, axis, xlabel, ylabel, ylim, xlim, legend)
import numpy as np
import xlrd
from scipy import stats
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import learning_curve, train_test_split


doc = xlrd.open_workbook('./Ean.xlsx').sheet_by_index(0)
#.sheet_by_index mean get the first sheet by index

#get attribute name(i.e. Ea and H)
attributeNames = doc.row_values(0,0,2)

#print(attributeNames)
#the first classlabels is the surface
surfacedata = doc.col_values(2,231,584)
surface = np.array(surfacedata).reshape(-1,1)
print(surface)

# onehotencode
enc = OneHotEncoder(handle_unknown='ignore')
surfacelist = enc.fit_transform(surface).toarray()
print(enc.categories_)

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

#get the second-fours class label , which is ab*

abdata = doc.col_values(3,231,584)
ab = np.array(abdata).reshape(-1,1)
ablist = enc.fit_transform(ab).toarray()

adata = doc.col_values(4,231,584)
a = np.array(adata).reshape(-1,1)
alist = enc.fit_transform(a).toarray()

bdata = doc.col_values(5,231,584)
b = np.array(bdata).reshape(-1,1)
blist = enc.fit_transform(b).toarray()
print(blist)

reaction_type = doc.col_values(6,231,584)
reaction = np.array(reaction_type).reshape(-1,1)
reaction_typelist = enc.fit_transform(reaction).toarray()
print(reaction_typelist)

y = np.asarray(doc.col_values(1,231,584)).reshape(-1,1)

x = np.asarray(doc.col_values(0,231,584)).reshape(-1,1)

x1 = np.hstack((x, surfacelist))

x2 = np.hstack((x1, reaction_typelist))

# x3 involves the fingerprinter of the metals
x3 = np.hstack((x2, elemlist))

#Set the X for model, x1, x2, x3
X = x3
y=y.ravel()

#K-fold crossvalidation
K_inner = 10
K_outer = 10
inner_cv = model_selection.KFold(n_splits=K_inner, shuffle=True,random_state=42)
outer_cv = model_selection.KFold(n_splits=K_outer, shuffle=True,random_state=42)
K = K_outer

Error_train_GP = np.empty((K,1))
Error_cv_GP = np.empty((K,1))
Root_mean_square_error_GP_cv = np.empty((K,1))
mean_absolute_error_train_GP = np.empty((K,1))
mean_absolute_error_cv_GP = np.empty((K,1))
mae_test_gp = np.empty((K,1))
rmse_test_gp = np.empty((K,1))
mse_test_gp = np.empty((K,1))

# forward feature selection
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
#from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

#kernel = C(0.1, (0.001, 10)) * RBF(0.5, (1e-4, 10))
#reg_GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=0.05,
#                                         normalize_y=True, random_state=20)
#sfs1 = SFS(reg_GP,
#           k_features=1,
#           forward=True,
#           floating=False,
#          scoring='neg_mean_absolute_error',
#           cv=10)

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
# plt.text(7,-1, r'GP-1',{'color':'k','fontsize':18})
# plt.show()

#Gaussian Process
#first split the data into train+cv and test
k_outer = 0
for train_index, test_index in outer_cv.split(X):
    print('Computing CV fold:{0}/{1}..'.format(k_outer + 1, K))
    X_train, Y_train = X[train_index, :], y[train_index]
    x_test, y_test = X[test_index, :], y[test_index]
    #scaler = StandardScaler()
    #scaler = scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #x_test = scaler.transform(x_test)
    #use the following code to selcect the parameters
    # set up the possible values of parametrs to potimize over
    #p_grid_gp = {'alpha':[0.01,0.02,0.05,0.08,0.1,0.2,0.5,0.8,1,1.2,1.5,2,2.5,3,5]}
    # sepecify the estimator
    # kernel = C(0.1, (0.001, 10)) * RBF(0.5, (1e-4, 10))
    # gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30,
    #                                  normalize_y=True, random_state=20)
    # clf_gp = GridSearchCV(gp, p_grid_gp, cv=5,return_train_score=True,
    #     #                      scoring='neg_mean_absolute_error')
    # clf_gp.fit(X_train, Y_train)
    #print('cv result_gp:{0}'.format(clf_gp.cv_results_))
    #print('cv_bestscore_gp:{0}'.format(clf_gp.best_score_))
    #print('cv_best_params_gp:{0}'.format(clf_gp.best_params_))
    #cv_best_params_gp:{'alpha': 0.01,0.05,0.05,0.05,0.05,0.02,0.01,0.01,0.05,0.08}

    k_inner = 0
    for train_index, test_index in inner_cv.split(X_train):
        #print('Computing CV fold:{0}/{1}..'.format(k+1,K))
        x_train, y_train = X_train[train_index,:], Y_train[train_index]
        x_cv, y_cv = X_train[test_index,:], Y_train[test_index]

        # Train the net on training data
        kernel = C(0.1, (0.001, 10)) * RBF(0.5, (1e-4, 10))
        reg_GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=0.05,
                                          normalize_y=True, random_state=20)
        reg_GP = reg_GP.fit(x_train, y_train)
        Error_train_GP[k_inner] = np.square(y_train - reg_GP.predict(x_train)).sum() / y_train.shape[0]
        Error_cv_GP[k_inner] = np.square(y_cv - reg_GP.predict(x_cv)).sum() / y_cv.shape[0]
        Root_mean_square_error_GP_cv[k_inner] = np.sqrt(Error_cv_GP[k_inner])
        mean_absolute_error_train_GP[k_inner] = mean_absolute_error(y_train, reg_GP.predict(x_train))
        mean_absolute_error_cv_GP[k_inner] = mean_absolute_error(y_cv, reg_GP.predict(x_cv))
        k_inner += 1

    reg_GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=0.05,
                                      normalize_y=True, random_state=20)
    reg_GP = reg_GP.fit(X_train, Y_train)

    mae_test_gp[k_outer] = mean_absolute_error(y_test, reg_GP.predict(x_test))
    mse_test_gp[k_outer] = mean_squared_error(y_test, reg_GP.predict(x_test))
    rmse_test_gp[k_outer] = np.sqrt(mse_test_gp[k_outer])
    k_outer += 1

print('MSE_GP_train:{0}'.format(Error_train_GP.mean()))
print('MSE_GP_cv:{0}'.format(Error_cv_GP.mean()))
print('RMSE_GP_cv:{0}'.format(Root_mean_square_error_GP_cv.mean()))
print('MAE_GP_train:{0}'.format(mean_absolute_error_train_GP.mean()))
print('MAE_GP_cv:{0}'.format(mean_absolute_error_cv_GP.mean()))
print(reg_GP.score(x_train, y_train), reg_GP.score(x_cv, y_cv))
print('mae_test_gp:{0}'.format(mae_test_gp.mean()))
print('mse_test_gp:{0}'.format(mse_test_gp.mean()))
print('rmse_test_gp:{0}'.format(rmse_test_gp.mean()))

#save the model to disk
import pickle
filename = 'regGP2.sav'
pickle.dump(reg_GP,open(filename,'wb'))

#load the model from disk
loaded_model = pickle.load(open(filename,'rb'))

y_predict = reg_GP.predict(X)
#export the predicted y to excel
import pandas as pd
data = pd.DataFrame(y_predict)

#datatoexcel = pd.ExcelWriter('yNN.xlsx')
data.to_excel('yGP2.xlsx', sheet_name='sheet 1')

print(mean_absolute_error(y, reg_GP.predict(X)))



train_error_gp = y_train - reg_GP.predict(x_train)
ind = np.argmax(train_error_gp)
print(np.amax(train_error_gp),np.amin(train_error_gp), np.argmax(train_error_gp))

test_error_gp = y_test - reg_GP.predict(x_test)
print(np.amax(test_error_gp),np.amin(test_error_gp), np.argmax(test_error_gp))

#train error histogram
figure()
residual = reg_GP.predict(x_train)-y_train
hist(residual,10, label= 'GP2')
xlabel('Train error(eV)')

figure()
plt.rcParams.update({'font.size': 20})
subplot(1,2,1)
plot(y_train, reg_GP.predict(x_train), 'bo',label = 'train_set')
plot(y_test, reg_GP.predict(x_test), 'rh',label = 'test_set')
xlim(-1,5)
ylim(-1,5)
plt.text(3.5,4.5, r'GP-2',{'color':'k','fontsize':18})
plt.text(2.8,-0.5,'MAE={:.2f}'.format(mae_test_gp.mean()),{'color':'k','fontsize':18})
plt.text(2.8,-0.8,'RMSE={:.2f}'.format(rmse_test_gp.mean()),{'color':'k','fontsize':18})
xlabel('DFT calculated barrier(eV)');
ylabel('Predicted barrier(eV)');
plot([-1, 5],[-1,5],ls="--", c=".3")

subplot(1,2,2)
residual = reg_GP.predict(x_test)-y_test
hist(residual,10, label= 'Gaussian Process')
xlabel('error(eV)')

show()
