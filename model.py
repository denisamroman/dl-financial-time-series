from src.config import *
from feature_selection import FeatureSelection
from model_hp import *

data_frequency = 'hourly'
extended_features = True
is_feature_extraction_saved = True
is_feature_selection_completed = True
graphs_path = Path('img_output')

"""
Custom Functions
"""
def get_data_and_features():
    """
    Method to get the data from the local folder and extract new features 
    There are two datasets with different frequencies and timeframes (hourly 4 years; daily 7 years)
    There are two options of features for the hourly dataset (pandas-ta technical indicators; standard deviation and returns)
    """
    if (data_frequency == 'hourly'):
        if(extended_features):
            if(is_feature_extraction_saved ==  False):
                df = pd.read_csv('data_source/data/'+data_frequency+'/data_1h.csv')
                df.datetime = pd.to_datetime(df.datetime)
                df = df.set_index('datetime', drop=True)
                df['Return'] = np.log(df['close']).diff()
                print(df.describe().T)
                print(df.isna().sum())
                columns_base = ['open','high','low','close','Volume']
                df = get_features_ta(df, columns_base)
                df = get_target(df, data_frequency)
                print(df.describe().T)
                print(df.isna().sum())
                df.to_csv('data_source/data/'+data_frequency+'/data_1h_with_features_ext.csv')
            else:
                df = pd.read_csv('data_source/data/'+data_frequency+'/data_1h_with_features_ext.csv')
                df.datetime = pd.to_datetime(df.datetime)
                df = df.set_index('datetime', drop=True)
        else:
            if(is_feature_extraction_saved ==  False):
                df = pd.read_csv('data_source/data/'+data_frequency+'/data_1h.csv')
                df.datetime = pd.to_datetime(df.datetime)
                df = df.set_index('datetime', drop=True) 
                df = get_features(df)
                df = get_target(df, data_frequency)
                df = df.drop(['open','high','low','close','Volume MA','days','hours'], axis=1)
                df.to_csv('data_source/data/'+data_frequency+'/data_1h_with_features.csv')
            else:
                df = pd.read_csv('data_source/data/'+data_frequency+'/data_1h_with_features.csv')
                df.datetime = pd.to_datetime(df.datetime)
                df = df.set_index('datetime', drop=True)
        return df
    elif (data_frequency == 'daily'):
        if(is_feature_extraction_saved ==  False):
            df = pd.read_csv('data_source/data/'+data_frequency+'/data_1d.csv')
            df.Date = pd.to_datetime(df.Date)
            df = df.set_index('Date', drop=True)
            print(df.describe().T)
            print(df.isna().sum())
            columns_base = ['Open','High','Low','Close','Volume']
            df = get_features_ta(df, columns_base)
            df = get_features_fa(df)
            df = get_target(df, data_frequency)
            print(df.describe().T)
            print(df.isna().sum())
            df.to_csv('data_source/data/'+data_frequency+'/data_1d_with_features.csv')
        else:
            df = pd.read_csv('data_source/data/'+data_frequency+'/data_1d_with_features.csv')
            df.Date = pd.to_datetime(df.Date)
            df = df.set_index('Date', drop=True)
        return df
    else:
        print('Selected frequency is not available.')

def get_transformer(df):
    """
    Creates the custom transormed used in the feature selection process
    """
    if(data_frequency == 'hourly' and extended_features == True):
        num_data = df.copy()
        num_data.drop(['days', 'hours', 'Target'], axis=1, inplace=True)
        ct = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), num_data.columns),
                ('dtrans', DayTransformer(), ['days']),
                ('ttrans', TimeTransformer(), ['hours'])
            ])
        return ct
    else:
        return StandardScaler()


def plot_roc_pos_neg(y_test, y_pred, description):
    """
    Compute ROC curve and ROC area for each class
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc[1] = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc[1],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('ROC Curve (+) - '+description)
    plt.legend(loc="lower right")
    plt.savefig(graphs_path / str('ROC_curve_pos_'+description+'.png'))
    plt.close()

    fnr = 1-tpr
    tnr = 1-fpr
    roc_auc[0] = auc(fnr, tnr)
    plt.figure()
    plt.plot(
        fnr,
        tnr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc[0],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Negative Rate")
    plt.ylabel("True Negative Rate")
    plt.title('ROC Curve (-) - '+description)
    plt.legend(loc="lower right")
    plt.savefig(graphs_path / str('ROC_curve_neg_'+description+'.png'))
    plt.close()

"""
Trading Strategy and Backtesting
"""
def trading_strategy(data, Xtest, ypred, ypred_proba, seqlen):
    # Create a new dataframe to subsume outsample data
    df1 = data[-(len(Xtest)-seqlen):]

    # Predict the signal and store in predicted signal column
    
    df1['Signal'] = ypred
    df1['Proba'] = ypred_proba
    df1['Signal'] = df1['Signal'].shift(1).fillna(0)
    df1['Proba'] = df1['Proba'].shift(1).fillna(0)

    # spread
    s = 0 # (138-137.87)/137.87

    # Calculate the strategy returns; assumes 100% long
    df1['Strategy_100'] = (df1['Returns'] * df1['Signal'])-s
    df1['Strategy_Kelly'] = (df1['Returns'] * (2*df1['Proba']-1))-s
    df1.to_csv('trading_strategy.csv')

    # Localize index for pyfolio
    df1.index = df1.index.tz_localize('utc')
    #df1.to_csv('backtesting.csv')
    # Create Tear sheet using pyfolio for outsample - for X_test
    pf.create_simple_tear_sheet(df1['Strategy_100'])
    pf.create_returns_tear_sheet(df1['Strategy_100'])
    #f.savefig('pyfolio_strategy_100_sheet2.png')
    pf.create_simple_tear_sheet(df1['Strategy_Kelly'])
    pf.create_returns_tear_sheet(df1['Strategy_Kelly'])
    #f2.savefig('pyfolio_strategy_Kelly_sheet2.png')
    plt.plot()
    plt.savefig(graphs_path / str('pyfolio_strategy_100_sheet.png'))
    plt.close()

    df1['Cum_Returns']=(1+df1['Returns']).cumprod()
    df1['Cum_Returns_100Bets'] = (1+df1['Strategy_100']).cumprod()
    df1['Cum_Returns_Kelly'] = (1+df1['Strategy_Kelly']).cumprod()
    # Visualize raw price series
    fig = plt.figure()

    plt.plot(df1['Cum_Returns'], color='cornflowerblue', label='AAPL')
    plt.plot(df1['Cum_Returns_100Bets'], color='red', label='100% bets')
    plt.plot(df1['Cum_Returns_Kelly'], color='green', label='Kelly')
    plt.title('APPL vs Model Prediction')
    plt.legend()
    plt.savefig(graphs_path / str('strategy_returns.png'))
    plt.close()
"""
Apply Models
"""

def apply_model_generator(model, g, g_, data, Xtest, ytest, seqlen, cwts, model_name, results_path, include_trading=False):
    print(model.summary())
    plot_model(model, to_file='img/model_'+model_name+'.png', show_shapes=True, show_layer_names=True)

    model_path = (results_path / str('model_'+model_name+'.h5')).as_posix()
    logdir = os.path.join("logs"+model_name, dt.datetime.now().strftime("%Y%m%d-%H%M%S"))

    my_callbacks = [
        EarlyStopping(patience=10, monitor='loss', mode='min', verbose=1, restore_best_weights=True),
        ModelCheckpoint(filepath=model_path, verbose=1, monitor='loss', save_best_only=True),
        TensorBoard(log_dir=logdir, histogram_freq=1)
    ]
    # Model fitting
    model.fit(g, epochs=500, verbose=1, callbacks=my_callbacks, shuffle=False, class_weight=cwts)
        
    # predictions
    ypred_proba = model.predict(g_, verbose=False) 
    ypred = np.where(model.predict(g_, verbose=False) > 0.5, 1, 0)
        
    # load model - os.path.abspath(model_path)
    model = load_model(model_path)

    # summarize model
    model.summary()
    # evaluate the model
    score = model.evaluate(g_, verbose=0)
    print("Testing scores:")
    print(f'{model.metrics_names[0]}, {score[0]*100:.4}%')
    print(f'{model.metrics_names[1]}, {score[1]*100:.4}%')
    print(f'{model.metrics_names[2]}, {score[2]*100:.4}%')
    print(f'{model.metrics_names[3]}, {score[3]*100:.4}%')
    print(f'{model.metrics_names[4]}, {score[4]*100:.4}%')

    score = model.evaluate(g, verbose=0)
    print("Training scores:")
    print(f'{model.metrics_names[0]}, {score[0]*100:.4}%')
    print(f'{model.metrics_names[1]}, {score[1]*100:.4}%')
    print(f'{model.metrics_names[2]}, {score[2]*100:.4}%')
    print(f'{model.metrics_names[3]}, {score[3]*100:.4}%')
    print(f'{model.metrics_names[4]}, {score[4]*100:.4}%')

    # Plot Confusion Matrix
    cm = confusion_matrix(ytest[seqlen:], ypred)
    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.4g')

    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    plt.savefig(graphs_path / f'confusion_matrix_{model_name}.png', dpi=150)
    plt.close()

    # Classification Report
    print(classification_report(ytest[seqlen:], ypred))

    # Plot ROC
    plot_roc_pos_neg(ytest[seqlen:], ypred, model_name)

    if(include_trading == True):
        trading_strategy(data, Xtest, ypred, ypred_proba, seqlen)

    # plot predictions
    """df1 = data.Target[-(len(Xtest)-seqlen):]
    fig, axs = plt.subplots(2, sharex=True, figsize=(20,10))
    fig.suptitle('Test data with signals', size=18)
    axs[0].plot(df1.index, df1)
    axs[1].plot(df1.index, ypred)
    plt.show()"""

"""
Create Models
"""
def create_model_mlp(X_train, y_train, X_test, y_test, num=2):
    """
    Build a multi layer perceptron and fit the data
    Activation function: relu
    Optimizer function: SGD
    Learning Rate: Inverse Scaling
    """
    classifier = MLPClassifier(hidden_layer_sizes=num, max_iter=100, activation='logistic', solver='adam', verbose=10, random_state=762, early_stopping=True)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    score = accuracy_score(y_test, predictions)
    print('Mean accuracy of test predictions'+str(score))
    predictions_train = classifier.predict(X_train)
    score_train = accuracy_score(y_train, predictions_train)
    print('Mean accuracy of train predictions'+str(score_train))
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.4g')

    plt.title(f'Confusion Matrix - MLP')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    plt.savefig(graphs_path / f'confusion_matrix_MLP.png', dpi=150)
    plt.close()

    # Classification Report
    print(classification_report(y_test, predictions))

    # Plot ROC
    plot_roc_pos_neg(y_test, predictions, 'MLP')

def create_model_gru(hu=256, lookback=60, features=1):
    """
    Build a GRU model with multiple layers
    Droupout: 0.2
    Activation function: tanh
    Optimizer function: Adam
    """
    tf.keras.backend.clear_session()   

    model = Sequential()
    model.add(GRU(units=hu, input_shape=(lookback, features), return_sequences=True, activation='tanh', name='GRU1'))
    model.add(Dropout(0.2, name='Dropout1'))
    model.add(GRU(units=hu, activation='tanh', name='GRU2'))
    model.add(Dropout(0.2, name='Dropout2'))
    model.add(Dense(units=1, activation='sigmoid', name='Output'))

    # specify optimizer separately (preferred method))
    opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)       
    
    # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy', 
                            Precision(),
                            Recall(),
                            AUC()])
        
    return model

# Create a simple LSTM
def create_model_simple(hu=256, lookback=60, features=1):
    """
    Build a simple LSTM model with a layer of cells
    Activation function: elu
    Optimizer function: Adam
    """
    tf.keras.backend.clear_session()   
        
    # instantiate the model
    model = Sequential()
    model.add(LSTM(units=hu*2, input_shape=(lookback, features), activation = 'elu', return_sequences=False, bias_regularizer=L1L2(l1=0.0001, l2=0.0001), name='LSTM'))
    model.add(Dense(units=1, activation='sigmoid', name='Output'))             
    
    # specify optimizer separately (preferred method))
    opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)       
    
    # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy', 
                           Precision(),
                           Recall(),
                           AUC()])
        
    return model

# Create LSTM models with multiple layers
def create_model_multiple_1(hu=256, lookback=60, features=1):
    """
    Build a multi layer LSTM model 
    Layers: 2
    Dropout: 0.4
    Activation function: tanh
    Optimizer function: Adam
    """
    tf.keras.backend.clear_session()   

    model = Sequential()
    model.add(LSTM(units=hu, input_shape=(lookback, features), return_sequences=True, activation='tanh', name='LSTM1'))
    model.add(Dropout(0.4, name='Dropout1'))
    model.add(LSTM(units=hu, activation='tanh'))
    model.add(Dropout(0.4, name='Dropout2'))
    model.add(Dense(units=1, activation='sigmoid', name='Output'))

    # specify optimizer separately (preferred method))
    opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)       
    
    # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy', 
                            Precision(),
                            Recall(),
                            AUC()])
        
    return model

def create_model_multiple_2(hu=256, lookback=60, features=1):
    """
    Build a multi layer LSTM model 
    Layers: 3
    Dropout: 0.2
    Activation function: elu
    Optimizer function: Adam
    Bias L1, L2 Regularization
    """
    tf.keras.backend.clear_session()   
        
    # instantiate the model
    model = Sequential()
    
    model.add(LSTM(units=hu*2, input_shape=(lookback, features), activation = 'elu', return_sequences=True, bias_regularizer=L1L2(l1=0.01, l2=0.01), name='LSTM1'))
    model.add(Dropout(0.2, name='Drouput1'))
    model.add(LSTM(units=hu, activation = 'elu', return_sequences=True, bias_regularizer=L1L2(l1=0.01, l2=0.01), name='LSTM2'))
    model.add(Dropout(0.2, name='Drouput2'))
    model.add(LSTM(units=hu, activation = 'elu', return_sequences=False, bias_regularizer=L1L2(l1=0.01, l2=0.01), name='LSTM3'))
    model.add(Dense(units=1, activation='sigmoid', name='Output'))             
    
    # specify optimizer separately (preferred method))
    opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)       
    
    # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy',
                           Precision(),
                           Recall(),
                           AUC()])
        
    return model

def create_model_multiple_3(hu=256, lookback=60, features=1):
    """
    Build a multi layer LSTM model 
    Layers: 4
    Dropout: 0.2
    Activation function: elu
    Optimizer function: Adam
    Bias L1, L2 Regularization
    """
    tf.keras.backend.clear_session()   
        
    # instantiate the model
    model = Sequential()
    
    model.add(LSTM(units=hu*2, input_shape=(lookback, features), activation = 'elu', return_sequences=True, bias_regularizer=L1L2(l1=0.01, l2=0.01), name='LSTM1'))
    model.add(Dropout(0.2, name='Drouput1'))
    model.add(LSTM(units=hu, activation = 'elu', return_sequences=True, bias_regularizer=L1L2(l1=0.01, l2=0.01), name='LSTM2'))
    model.add(Dropout(0.2, name='Drouput2'))
    model.add(LSTM(units=hu, activation = 'elu', return_sequences=True, bias_regularizer=L1L2(l1=0.01, l2=0.01), name='LSTM3'))
    model.add(Dropout(0.2, name='Drouput3'))
    model.add(LSTM(units=hu, activation = 'elu', return_sequences=False, bias_regularizer=L1L2(l1=0.01, l2=0.01), name='LSTM4'))
    model.add(Dropout(0.2, name='Drouput4'))
    model.add(Dense(units=1, activation='sigmoid', name='Output'))             
    
    # specify optimizer separately (preferred method))
    opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)       
    
    # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy', 
                           Precision(),
                           Recall(),
                           AUC()])
        
    return model

def create_model_multiple_4(hu=256, lookback=60, features=1):
    """
    Build a multi layer LSTM model/ Model 2 improved
    Layers: 3
    Dropout: 0.2
    Activation function: elu
    Optimizer function: Adam; higher epsilon (and the denominator) => smaller weight updates
    Recurrent L1, L2 Regularization
    """
    tf.keras.backend.clear_session()   
        
    # instantiate the model
    model = Sequential()
    
    model.add(LSTM(units=hu*2, input_shape=(lookback, features), activation = 'elu', return_sequences=True, recurrent_regularizer=L1L2(l1=0.001, l2=0.001), name='LSTM1'))
    model.add(Dropout(0.2, name='Drouput1'))
    model.add(LSTM(units=hu, activation = 'elu', return_sequences=True, recurrent_regularizer=L1L2(l1=0.001, l2=0.001), name='LSTM2'))
    model.add(Dropout(0.2, name='Drouput2'))
    model.add(LSTM(units=hu, activation = 'elu', return_sequences=False, recurrent_regularizer=L1L2(l1=0.001, l2=0.001), name='LSTM3'))
    model.add(Dense(units=1, activation='sigmoid', name='Output'))             
    
    # specify optimizer separately (preferred method))
    opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)       
    
    # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy', 
                           Precision(),
                           Recall(),
                           AUC()])
        
    return model

def create_model_multiple_5(hu=256, lookback=60, features=1):
    """
    Build a multi layer LSTM model
    Layers: 3
    Dropout: 0.4
    Activation function: LeakyReLU
    Optimizer function: Adam
    """
    tf.keras.backend.clear_session()   
        
    # instantiate the model
    model = Sequential()
    
    model.add(LSTM(units=hu*2, input_shape=(lookback, features), activation = LeakyReLU(alpha=0.01), return_sequences=True, name='LSTM1'))
    model.add(Dropout(0.4, name='Drouput1'))
    model.add(LSTM(units=hu, activation = LeakyReLU(alpha=0.01), return_sequences=True, name='LSTM2'))
    model.add(Dropout(0.4, name='Drouput2'))
    model.add(LSTM(units=hu, activation = LeakyReLU(alpha=0.01), return_sequences=False, name='LSTM3'))
    model.add(Dense(units=1, activation='sigmoid', name='Output'))             
    
    # specify optimizer separately (preferred method))
    opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)       
    
    # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy', 
                           Precision(),
                           Recall(),
                           AUC()])
        
    return model

def create_model_multiple_hp(hu=256, lookback=60, features=1):
    """
    Build a multi layer LSTM model
    Layers: 3
    Activation function: tanh
    Optimizer function: Adam
    """
    tf.keras.backend.clear_session()   
        
    # instantiate the model
    model = Sequential()
    
    model.add(LSTM(units=4, input_shape=(lookback, features), activation = 'tanh', return_sequences=True, recurrent_regularizer=L1L2(l1=0.001, l2=0.001), bias_regularizer=L1L2(l1=0.001, l2=0.001), name='LSTM1'))
    model.add(LSTM(units=32, activation = 'tanh', return_sequences=True, recurrent_regularizer=L1L2(l1=0.001, l2=0.001), bias_regularizer=L1L2(l1=0.001, l2=0.001), name='LSTM2'))
    model.add(LSTM(units=32, activation = 'tanh', return_sequences=False, name='LSTM3'))
    model.add(Dense(units=1, activation='sigmoid', name='Output'))             
    
    # specify optimizer separately (preferred method))
    opt = Adam(lr=0.01, epsilon=1e-08, decay=0.0)       
    
    # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy', 
                           Precision(),
                           Recall(),
                           AUC()])
        
    return model

def create_model_cnn_lstm(hu=256, lookback=60, features=1):
    """
    Build a CNN-LSTM model
    Dropout: 0.4
    Activation function: relu
    Optimizer function: Adam
    """
    tf.keras.backend.clear_session() 

    # instantiate the model
    model = Sequential()
    
    model.add(Conv1D(filters=hu*2, kernel_size=1, padding='same', input_shape=(lookback, features), activation = 'relu', name='Conv1D'))
    model.add(MaxPooling1D(pool_size=1, padding='same', name='MaxPooling1D'))
    model.add(LSTM(units=hu*2, activation = 'relu', return_sequences=True, name='LSTM1'))
    model.add(Dropout(0.4, name='Drouput1'))
    model.add(LSTM(units=hu, activation = 'relu', return_sequences=True, name='LSTM2'))
    model.add(Dropout(0.4, name='Drouput2'))
    model.add(LSTM(units=hu, activation = 'relu', return_sequences=False, name='LSTM3'))
    model.add(Dense(units=1, activation='sigmoid', name='Output'))              
    
    # specify optimizer separately (preferred method))
    opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)       
    
    # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy', 
                           Precision(),
                           Recall(),
                           AUC()])
        
    return model


def create_model_stacked(X_train, y_train):
    ct = StandardScaler()
    # cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    # specify estimators
    dtc = Pipeline([('transformer', ct),('dtc', DecisionTreeClassifier())])
    rfc = Pipeline([('transformer', ct), ('rfc', RandomForestClassifier())])
    knn = Pipeline([('transformer', ct), ('knn', KNeighborsClassifier())])
    gbc = Pipeline([('transformer', ct), ('gbc', GradientBoostingClassifier())])
    # get cv score
    clf = [dtc,rfc,knn,gbc]
    for estimator in clf:
        score = cross_val_score(estimator, X_train, y_train, scoring = 'accuracy', cv=tscv, n_jobs=-1)
        print(f"The accuracy score of {estimator} is: {score.mean()}")
    # list of (str, estimator)
    clf = [('dtc',dtc),('rfc',rfc),('knn',knn),('gbc',gbc)] 
    # perform stacking with cv
    stack_model = StackingClassifier(estimators = clf, final_estimator = LogisticRegression())
    score = cross_val_score(stack_model, X_train, y_train, cv = tscv, scoring = 'accuracy')
    print(f"The accuracy score of is: {score.mean()}")


"""
Start of the project
"""

def main():

    df = get_data_and_features()

    # Visualization 
    # plt.figure(figsize=(14,6))
    # plt.title('AAPL Hourly Price')
    # plt.plot(df['close'])

    # Class Frequency
    c = df['Target'].value_counts()
    print(c)
    # Check Class Weights
    class_weight = cwts(df['Target'])
    print(class_weight)
    # With the calculated weights, both classes gain equal weight
    print(class_weight[0] * c[0]) 
    print(class_weight[1] * c[1])

    if(is_feature_selection_completed != True):
        data = df.copy()
        data.drop(['Returns'], axis=1, inplace=True)
        columns = data.columns
        # Scaling for feature selection
        ct = get_transformer(df)

        # Feature Engineering
        # Step 1 - Boruta 
        feature_selection = FeatureSelection(data, 0.2, data_frequency, extended_features, ct)
        forest = feature_selection.random_forest(class_weight)
        columns_filtered = feature_selection.boruta(forest)
        columns_filtered = np.append(columns_filtered, 'Target')
        columns_to_eliminate = columns[~columns.isin(columns_filtered)]
        data = df.copy()
        data.drop(columns_to_eliminate, axis=1, inplace=True)

        # Step 2 - SHAP/ VIF/ Corr Matrix   
        features = data.copy()
        features.drop(['Target'], axis=1, inplace=True) 
        feature_selection = FeatureSelection(data, 0.2, data_frequency, extended_features, StandardScaler(), False)
        # K-best
        print(feature_selection.k_best())
        # VIF
        vif = feature_selection.vif(features.columns).round(2)
        print(vif)
        # SHAP with XGBoost
        model = feature_selection.xgb_classifier()
        shap = feature_selection.shap(model, features.columns)
        # Corr Matrix
        corr = feature_selection.corr_matrix(features.corr())
    
    data = pd.read_csv('data_source/data/hourly/features_1H.csv')
    data.datetime = pd.to_datetime(data.datetime)
    data = data.set_index('datetime', drop=True) 
    data_with_returns = data.copy()
    data_with_returns['Returns'] = df['Return']
  
    # Selected Features Analysis
    # features = data.copy()
    # features.drop(['Target'], axis=1, inplace=True)
    # pd.plotting.scatter_matrix(features, alpha=0.2)
    # sns.pairplot(features)
    # plt.show()
    # data.describe().T
    # Create a density plot of the 'feature' column
    """sns.kdeplot(df['Volume'], shade=True)
    plt.show()
    sns.kdeplot(df['BBP_5_2.0'], shade=True)
    plt.show()
    sns.kdeplot(df['DPO_20'], shade=True)
    plt.show()
    sns.kdeplot(df['BULLP_13'], shade=True)
    plt.show()"""
    

    # Model Building
    
    results_path = Path('results', 'lstm_time_series')
    if not results_path.exists():
        results_path.mkdir(parents=True)

    # Option to use TimeseriesGenerator for reshaping the dataset for LSTM model
    # use_time_series_generator = True

    # Test size
    test_size=0.2

    # Number of features
    numfeat = data.shape[1] - 1
    if (numfeat != numfeatures):
        print('Please update the numfeatures variable in the model_hp file to:'+numfeat)

    # Scaling applied 
    ct = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), ['Volume','BBP_5_2.0']),
                ('normal', MinMaxScaler(), ['DPO_20', 'BULLP_13'])
            ])

    # Prepare the train and test set
    Xtrain, ytrain, Xtest, ytest = get_dataset_split(data, test_size, seqlen, True, ct, numfeat)
    
    # MLP Model
    create_model_mlp(Xtrain, ytrain, Xtest, ytest,3)

    # Stacked ensemble
    create_model_stacked(Xtrain, ytrain)

    # Apply TimeseriesGenerator
    g = TimeseriesGenerator(Xtrain, ytrain, length=seqlen)
    g_ = TimeseriesGenerator(Xtest, ytest, length=seqlen)
    # verify length
    print(len(g), len(g_))
    print(g[0][0])
    print(g[0][1])
    # verify batch size
    for i in range(len(g)):
        a, b = g[i]
        print(a.shape, b.shape)

    # GRU Model 
    model_gru = create_model_gru(hu=10, lookback=seqlen, features=numfeat)
    apply_model_generator(model_gru, g, g_, data, Xtest, ytest, seqlen, class_weight, 'GRU_Gen_1h', results_path)
        
        
    # Simple LSTM
    model_unstacked = create_model_simple(hu=10, lookback=seqlen, features=numfeat)
    apply_model_generator(model_unstacked, g, g_, data, Xtest, ytest, seqlen, class_weight, 'SimpleLSTM_Gen_1h', results_path)
        
    # Stacked LSTM
    model_stacked = create_model_multiple_3(hu=10, lookback=seqlen, features=numfeat)
    apply_model_generator(model_stacked, g, g_, data_with_returns, Xtest, ytest, seqlen, class_weight, 'StackedLSTM3_Gen_1h', results_path, True)

    # CNN-LSTM
    model_stacked = create_model_cnn_lstm(hu=10, lookback=seqlen, features=numfeat)
    apply_model_generator(model_stacked, g, g_, data, Xtest, ytest, seqlen, class_weight, 'CNN-LSTM_Gen_1h', results_path)


    # Hyperparameter Optimization 
    """
    For tuner select: HyperBand, RandomSearch, or BayesianOptimization
    """
    tuner = 'RandomSearch'
    result = create_model_hp(tuner, g, g_, class_weight, seqlen, numfeat)

  
        

        

if __name__ == '__main__':
    freeze_support()
    main()
