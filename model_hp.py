from src.config import *

numfeatures = 4
seqlen = 14

def build_model(hp):
    tf.keras.backend.clear_session()

    # instantiate the model
    model = Sequential()

    # Tune the number of units in the layers
    hp_units1 = hp.Int('units1', min_value=4, max_value=32, step=4)
    hp_units2 = hp.Int('units2', min_value=4, max_value=32, step=4)
    hp_units3 = hp.Int('units3', min_value=4, max_value=32, step=4)

    # Tune the dropout rate
    hp_dropout1 = hp.Float(
    'Dropout_rate', min_value=0, max_value=0.5, step=0.1)
    hp_dropout2 = hp.Float(
    'Dropout_rate', min_value=0, max_value=0.5, step=0.1)

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice(
    'learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Tune activation functions
    hp_activation1 = hp.Choice(name='activation', values=[
       'relu', 'elu', 'tanh'], ordered=False)
    hp_activation2 = hp.Choice(name='activation', values=[
       'relu', 'elu', 'tanh'], ordered=False)
    hp_activation3 = hp.Choice(name='activation', values=[
       'relu', 'elu', 'tanh'], ordered=False)
    
    hp_w_initializer1 = hp.Choice('kernel_initializer', values=['glorot_uniform', 'glorot_normal'])
    hp_w_initializer2 = hp.Choice('kernel_initializer', values=['glorot_uniform', 'glorot_normal'])
    hp_w_initializer3 = hp.Choice('kernel_initializer', values=['glorot_uniform', 'glorot_normal'])
    
    hp_bias_l1 = hp.Choice('lb1', values=[0.0, 0.01, 0.001])
    hp_bias_l2 = hp.Choice('lb2', values=[0.0, 0.01, 0.001])
    hp_bias_l3 = hp.Choice('lb3', values=[0.0, 0.01, 0.001])

    hp_rec_l1 = hp.Choice('lr1', values=[0.0, 0.01, 0.001])
    hp_rec_l2 = hp.Choice('lr2', values=[0.0, 0.01, 0.001])
    hp_rec_l3 = hp.Choice('lr3', values=[0.0, 0.01, 0.001])

    model.add(LSTM(hp_units1, input_shape=(seqlen, numfeatures),
      activation=hp_activation1, return_sequences=True, kernel_initializer=hp_w_initializer1, bias_regularizer=L1L2(l1=hp_bias_l1, l2=hp_bias_l1), recurrent_regularizer=L1L2(l1=hp_rec_l1, l2=hp_rec_l1), name='LSTM1'))
    model.add(Dropout(hp_dropout1, name='Drouput1'))

    model.add(LSTM(hp_units2, activation=hp_activation2,
      return_sequences=True, kernel_initializer=hp_w_initializer2, bias_regularizer=L1L2(l1=hp_bias_l2, l2=hp_bias_l2),  recurrent_regularizer=L1L2(l1=hp_rec_l2, l2=hp_rec_l2), name='LSTM2'))
    model.add(Dropout(hp_dropout2, name='Drouput2'))

    model.add(LSTM(hp_units3, activation=hp_activation3,
      return_sequences=False, kernel_initializer=hp_w_initializer3, bias_regularizer=L1L2(l1=hp_bias_l2, l2=hp_bias_l2),  recurrent_regularizer=L1L2(l1=hp_rec_l2, l2=hp_rec_l2), name='LSTM3'))

    model.add(Dense(units=1, activation='sigmoid', name='Output'))

    # specify optimizer separately (preferred method))
    opt = Adam(lr=hp_learning_rate, epsilon=1e-08, decay=0.0)

    # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile(optimizer=opt,
            loss=BinaryCrossentropy(),
            metrics=['accuracy',
            Precision(),
            Recall(),
            AUC()])

    return model


def create_model_hp(tune_method, g, g_, class_weight, seqlen, numfeat):
    # initialize an early stopping call back to prevent the model from
    # overfitting/spending too much time training with minimal gains
    callback = [EarlyStopping(patience=5, monitor='loss', mode='min', verbose=1, restore_best_weights=True),
        TensorBoard(log_dir="./tensorboard/"+tune_method+"logs")]

    if(tune_method == 'RandomSearch'):
        # RandomSearch algorithm from keras tuner
        tuner = RandomSearch(
            build_model,
            objective="val_accuracy",
            max_trials=5,
            directory="./keras",
            project_name="rstrail",
            overwrite=True
            )
    if(tune_method == 'HyperBand'):
        tuner = kt.Hyperband(
            build_model,
            objective="val_accuracy",
            max_epochs=5,
            hyperband_iterations=15,
            directory="./keras",
            project_name="hbtrail",
            overwrite=True
            )
    if(tune_method == 'BayesianOptimization'):
        tuner = BayesianOptimization(
            build_model,
            objective="val_accuracy",
            max_trials=15,
            num_initial_points=2,
            hyperparameters=None,
            tune_new_entries=True,
            allow_new_entries=True,
            overwrite=True,
            directory="./keras",
            project_name="botrial"
            )
    # launch tuning process
    tuner.search(g, epochs=50, validation_data=g_, callbacks=callback, class_weight = class_weight, shuffle=False)

    # display the best hyperparameter values for the model based on the defined objective function
    best_rshp = tuner.get_best_hyperparameters()[0]
    print(best_rshp.values)

    # display tuning results summary
    tuner.results_summary()

    return tuner
