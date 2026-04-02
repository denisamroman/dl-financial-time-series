from src.config import *

class FeatureSelection:
    def __init__(self, X, testsize, freq, extended_features, transformer=None, first_selection=True):

        self.X_df = X
        self.X = X.drop(['Target'],axis=1)
        columns = self.X.copy()
        self.y = X['Target'].values.astype(int)
        self.testsize = testsize
        
        # split training and testing dataset
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=testsize, shuffle=False)

        if(transformer!=None):
            # Transformation
            transformer.fit(X_train) 
            # scale the training dataset
            X_train = transformer.transform(X_train)
            # scale the test dataset
            X_test = transformer.transform(X_test)

        if(first_selection == True and freq == 'hourly' and extended_features == True):
            columns['dsin']=''
            columns['dcos']=''
            columns['tsin']=''
            columns['tcos']=''
            columns = columns.drop(['days', 'hours'], axis=1)
        self.columns = columns.columns
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

    def plot_data_scaled(self):
        X = pd.DataFrame(MinMaxScaler().fit_transform(self.X))
        pd.plotting.scatter_matrix(X, alpha=0.2)
        sns.pairplot(X)
        plt.show()

    def random_forest(self, cwts):
        # Define the ranfom forest classifier
        forest = RandomForestClassifier(n_jobs=-1, 
                                class_weight=cwts, 
                                random_state=42, 
                                max_depth=5)

        # train the model
        forest.fit(self.X_train, self.y_train)  

        # print scores
        print("Accuracy Score \t\t", accuracy_score(self.y_test, forest.predict(self.X_test)))

        return forest
    
    def xgb_classifier(self):
        xgbcls = XGBClassifier(verbosity = 0, silent=True, random_state=42)
        xgbcls.fit(self.X_train, self.y_train)
        print("Accuracy Score \t\t", accuracy_score(self.y_test, xgbcls.predict(self.X_test)))
        tscv = TimeSeriesSplit(n_splits=5, gap=1)
        # Hyper parameter optimization
        param_grid = {'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                    'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
                    'min_child_weight': [1, 3, 5, 7],
                    'gamma': [0.0, 0.1, 0.2 , 0.3, 0.4],
                    'colsample_bytree': [0.3, 0.4, 0.5 , 0.7]}
                    # perform random search
        rs = RandomizedSearchCV(xgbcls, param_grid, n_iter=100, scoring='f1', cv=tscv, verbose=0)
        rs.fit(self.X_train, self.y_train, verbose=0)
        # best parameters
        print(rs.best_params_)
        print(rs.best_score_)
        # Refit the XGB Classifier with the best params
        cls = XGBClassifier(**rs.best_params_) 

        cls.fit(self.X_train, self.y_train, 
                eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                eval_metric='logloss',
                verbose=True)
        print("Accuracy Score \t\t", accuracy_score(self.y_test, cls.predict(self.X_test)))
        return cls
    
    def boruta(self, forest):
        # define Boruta feature selection method
        # perc set to 66 for daily data
        feat_selector = BorutaPy(forest, n_estimators='auto', perc=100, alpha=0.05, verbose=2, random_state=0)
        # find all relevant features
        # takes input in array format not as dataframe
        feat_selector.fit(self.X_train, self.y_train) 

        # check selected features
        print(feat_selector.support_)

        # check ranking of features
        print(feat_selector.ranking_)

        # call transform() on X to filter it down to selected features
        X_filtered = feat_selector.transform(self.X_train)

        feature_names = self.columns
        # zip my names, ranks, and decisions in a single iterable
        feature_ranks = list(zip(feature_names, 
                                feat_selector.ranking_, 
                                feat_selector.support_))

        # iterate through and print out the results
        for feat in feature_ranks:
            print(f'Feature: {feat[0]:<30} Rank: {feat[1]:<5} Keep: {feat[2]}')

        selected_rf_features = pd.DataFrame({'Feature':feature_names,
                                     'Ranking':feat_selector.ranking_})

        # selected_rf_features#.sort_values(by='Ranking') 

        print(selected_rf_features[selected_rf_features['Ranking']==1])
        print(X_filtered.shape)

        columns_filtered = selected_rf_features[selected_rf_features['Ranking']==1].iloc[:,0].values

        forest.fit(X_filtered, self.y_train)
        # first apply feature selector transform to make sure same features are selected
        X_test_filtered = feat_selector.transform(self.X_test)
        # check the shape
        print(X_test_filtered.shape)
        prediction = forest.predict(X_test_filtered)
        print("Accuracy Score \t\t", accuracy_score(self.y_test, prediction))
        # Classification Report
        print(classification_report(self.y_test, prediction))
        # plot roc
        plot_roc_curve(forest, X_test_filtered, self.y_test)
   
        return columns_filtered

    # Apply SHAP 
    def shap(self, model, features):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_test)
        shap.summary_plot(shap_values, self.X_test, feature_names = features)

        return plt.show()

    def k_best(self):
        # define feature selection
        fs = SelectKBest(score_func=f_classif, k=4)
        # apply feature selection
        X_selected = fs.fit_transform(self.X, self.y)
        cols = fs.get_support(indices=True)
        features_df_new = self.X.iloc[:,cols]  
        return features_df_new
    
    # Correlation Matrix
    def corr_matrix(self, corrmat):
        # Visualize feature correlation
        fig, ax = plt.subplots()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corrmat, dtype=bool))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(250, 15, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corrmat, annot=True, annot_kws={"size": 10}, 
                    fmt="0.2f", mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=False, linewidths=.5, cbar_kws={"shrink": 1})

        ax.set_title('Feature Correlation', fontsize=14, color='black')
        return plt.show()
    
    # VIF for mutlicolinearity
    def vif(self,columns):
        # subsume into a dataframe
        vif = pd.DataFrame()
        vif["Features"] = columns
        vif["VIF Factor"] = [variance_inflation_factor(self.X_train, i) for i in range(self.X_train.shape[1])]
        return vif

if __name__ == "__main__":
    freeze_support()
    # Data
    df = pd.read_csv('data_with_features.csv')
    df.Date = pd.to_datetime(df.Date)
    df = df.set_index('Date', drop=True)
    columns = df.columns

    # Class Frequency
    c = df['Target'].value_counts()

    # Check Class Weights
    class_weight = cwts(df['Target'])


    # Feature Engineering
    FeatureSelection(df, 0.2, StandardScaler())