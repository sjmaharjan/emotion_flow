from contextlib import redirect_stdout
from keras import callbacks, Input
from keras.layers.recurrent import GRU
from keras.layers import Dense, Dropout, Bidirectional
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np
import os
from deep_learning.layers import MLPAttention
from deep_learning.utils import get_optimizers
from keras.models import Model


class Experiemnt:
    def __init__(self, train_Xs, train_Ys, val_Xs, val_Ys, test_Xs, test_Ys, le, params, name):
        self.train_Xs = train_Xs
        self.train_Ys = train_Ys
        self.test_Xs = test_Xs
        self.test_Ys = test_Ys
        self.val_Xs = val_Xs
        self.val_Ys = val_Ys
        self.le = le
        self.params = params
        self.name = name

    def create_model(self, settings):
        pass

    def predict(self, checked_model_path, settings):
        model = self.create_model(settings)
        model.load_weights(checked_model_path)
        Y_test_pred = model.predict(self.test_Xs)
        if len(Y_test_pred) == 2:
            print("MT prediction task")
            Y_test_pred = Y_test_pred[0].flatten()
        Y_test_pred = np.array([1 if y > 0.5 else 0 for y in Y_test_pred])
        print("F1 Weighted Score: {}".format(f1_score(self.test_Ys['success_output'].flatten(),
                                                      Y_test_pred.flatten(), average='weighted', pos_label=None)))

    def run(self, output_folder, checkpoint_path):
        final_results = {}
        out_file = '{}.txt'.format(self.name)
        opfpath = os.path.join(output_folder, out_file)
        print("output path: ", opfpath)
        with open(opfpath, 'w') as f:
            with redirect_stdout(f):
                best_settings = self.params

                # train with best settings

                best_model = self.create_model(best_settings)
                best_model.summary()

                earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=best_settings['patience'],
                                                       verbose=0,
                                                       mode='auto')
                check_cb = callbacks.ModelCheckpoint(
                    checkpoint_path + self.name + '_succes_best_1.hdf5', monitor='val_loss',
                    verbose=1, save_best_only=True, mode='min')

                history = best_model.fit(self.train_Xs, self.train_Ys,
                                         epochs=best_settings['epochs'],
                                         batch_size=best_settings['batch_size'],
                                         validation_data=(self.val_Xs, self.val_Ys),
                                         shuffle=True,
                                         callbacks=[earlystop_cb, check_cb]
                                         )

                best_model.save_weights(checkpoint_path + self.name + '.h5')

                Y_test_pred = best_model.predict(self.test_Xs)
                if len(Y_test_pred) == 2:
                    # print("MT prediction task")
                    Y_test_pred = Y_test_pred[0].flatten()

                Y_test_pred = np.array([1 if y > 0.5 else 0 for y in Y_test_pred])

                final_results['train_val_dev'] = f1_score(self.test_Ys['success_output'].flatten(),
                                                          Y_test_pred.flatten(), average='weighted', pos_label=None)

                self.report(self.test_Ys['success_output'].flatten(), Y_test_pred.flatten())

                print("--" * 25, flush=True)

                print(flush=True)

    def print_predictions(self, Y_test, y_pred):
        print('*' * 25)
        print('Actual, Predicted')
        for x, y in zip(Y_test.flatten(), np.array(y_pred).flatten()):
            print("{}, {}".format(x, y))

        print('*' * 25)

    def report(self, Y_test, y_pred):

        print(classification_report(Y_test, y_pred))

        print("")
        print("Confusion matrix")
        print("============================================================")

        print()
        print(confusion_matrix(Y_test, y_pred))

        self.print_predictions(Y_test, y_pred)

        print("")
        print("F1 weighted {}".format(f1_score(Y_test, y_pred, average='weighted', pos_label=None)))


# Sentiment Flow Experiments
# MultitasK Models

class ExperimentSentimentAttention(Experiemnt):
    def create_model(self, settings):
        recurrent_units = settings['recurrent_units']
        recurrent_dropout = settings['recurrent_dropout']
        attention_units = settings['attention_units']

        model_inputs = []

        for input_name, data in self.train_Xs.items():

            if input_name == 'genre':
                model_inputs.append(Input(shape=(data.shape[-1],), dtype='float32', name=input_name))

            else:

                model_inputs.append(Input(shape=(data.shape[-2], data.shape[-1]), dtype='float32', name=input_name))
                blstm_layer = Bidirectional(
                    GRU(recurrent_units, activation=settings['recurrent_activation'], return_sequences=True,
                        recurrent_dropout=recurrent_dropout,
                        dropout=recurrent_dropout,
                        bias_initializer='ones', name='bi_gru'))(model_inputs[-1])

        attention = MLPAttention(units=attention_units, activation='selu')
        attention_out = attention(blstm_layer)

        # droput
        dropout_2_out = Dropout(settings['dropout_1'], name='dropout_final')(attention_out)

        # classification
        success_output = Dense(1, activation='sigmoid', name='success_output')(dropout_2_out)

        optimizer = get_optimizers('adam', settings['lr'])

        success_model = Model(inputs=model_inputs, outputs=success_output)
        success_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return success_model


class ExperimentSentimentAttentionMT(Experiemnt):
    def create_model(self, settings):
        number_of_genres = 8
        recurrent_units = settings['recurrent_units']
        recurrent_dropout = settings['recurrent_dropout']
        attention_units = settings['attention_units']

        for input_name, data in self.train_Xs.items():
            if input_name == 'genre':
                pass
            else:  # only one element for input
                x = Input(shape=(data.shape[-2], data.shape[-1]), dtype='float32', name=input_name)
                blstm_layer = Bidirectional(
                    GRU(recurrent_units, activation=settings['recurrent_activation'], return_sequences=True,
                        recurrent_dropout=recurrent_dropout,
                        dropout=recurrent_dropout,
                        bias_initializer='ones', name='bi_gru'))(x)

        attention = MLPAttention(units=attention_units, activation='selu')
        attention_out = attention(blstm_layer)

        # droput
        dropout_2_out = Dropout(settings['dropout_1'], name='dropout_final')(attention_out)

        # classification
        success_task = Dense(1, activation='sigmoid', name='success_output')(dropout_2_out)
        genre_task = Dense(number_of_genres, activation='softmax', name='genre_output',
                           kernel_initializer=settings['initializer'])(dropout_2_out)

        optimizer = get_optimizers('adam', settings['lr'])

        book_model = Model(inputs=x, outputs=[success_task, genre_task])
        book_model.compile(optimizer=optimizer,
                           loss={'success_output': 'binary_crossentropy', 'genre_output': 'categorical_crossentropy'},
                           loss_weights={'success_output': 1., 'genre_output': 1.})
        return book_model


### Simple Multitask Models


class ExperimentSentimentBiGRU(Experiemnt):
    def create_model(self, settings):
        recurrent_units = settings['recurrent_units']
        recurrent_dropout = settings['recurrent_dropout']

        model_inputs = []

        for input_name, data in self.train_Xs.items():
            if input_name == 'genre':
                model_inputs.append(Input(shape=(data.shape[-1],), dtype='float32', name=input_name))
            else:
                model_inputs.append(Input(shape=(data.shape[-2], data.shape[-1]), dtype='float32', name=input_name))
                blstm_layer = Bidirectional(
                    GRU(recurrent_units, activation=settings['recurrent_activation'], return_sequences=False,
                        recurrent_dropout=recurrent_dropout,
                        dropout=recurrent_dropout,
                        bias_initializer='ones', name='bi_gru'))(model_inputs[-1])
        # droput
        dropout_2_out = Dropout(settings['dropout_1'], name='dropout_final')(blstm_layer)

        # classification
        success_output = Dense(1, activation='sigmoid', name='success_output')(dropout_2_out)
        optimizer = get_optimizers('adam', settings['lr'])

        success_model = Model(inputs=model_inputs, outputs=success_output)
        success_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return success_model


class ExperimentSentimentBiGRUMT(Experiemnt):
    def create_model(self, settings):
        number_of_genres = 8
        recurrent_units = settings['recurrent_units']
        recurrent_dropout = settings['recurrent_dropout']

        for input_name, data in self.train_Xs.items():
            if input_name == 'genre':
                pass
            else:  # only one element for input
                x = Input(shape=(data.shape[-2], data.shape[-1]), dtype='float32', name=input_name)
                blstm_layer = Bidirectional(
                    GRU(recurrent_units, activation=settings['recurrent_activation'], return_sequences=False,
                        recurrent_dropout=recurrent_dropout,
                        dropout=recurrent_dropout,
                        bias_initializer='ones', name='bi_gru'))(x)

        # dropout
        dropout_2_out = Dropout(settings['dropout_1'], name='dropout_final')(blstm_layer)

        # classification
        success_task = Dense(1, activation='sigmoid', name='success_output')(dropout_2_out)

        genre_task = Dense(number_of_genres, activation='softmax', name='genre_output',
                           kernel_initializer=settings['initializer'])(dropout_2_out)

        optimizer = get_optimizers('adam', settings['lr'])

        book_model = Model(inputs=x, outputs=[success_task, genre_task])
        book_model.compile(optimizer=optimizer,
                           loss={'success_output': 'binary_crossentropy', 'genre_output': 'categorical_crossentropy'},
                           loss_weights={'success_output': 1., 'genre_output': 1.})
        return book_model
