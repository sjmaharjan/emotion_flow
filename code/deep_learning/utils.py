from keras.optimizers import Adam, Nadam, SGD,RMSprop


def get_optimizers(name, lr=0.00001):
    opt = {'adam': Adam(lr=lr, decay=1e-6),
           'nadam': Nadam(lr=lr, clipnorm=0.5),
            'sgd':SGD(lr=lr, momentum=0.9),
           'rmsprop':RMSprop(lr=lr, rho=0.7)
           }

    return opt.get(name, None)