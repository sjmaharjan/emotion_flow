import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True

    FEATURES = ['nrc_mean_sd_50','nrc_mean_sd_50_all']

    VECTORS = os.path.join(basedir, '../vectors')  # 70:30 vectors genre_vectors ->genre specific stratified vectors
    SUCCESS_OUTPUT = os.path.join(basedir, '../results/success')#'../results/success/')
    SUCCESS_OUTPUT_MT = os.path.join(basedir, '../results/cover_success_final/mt')#'../results/success/')
    BOOK_META_INFO = 'gutenberg_goodread_2016_match.xlsx'



class TestingConfig(Config):
    TESTING = True


class ProductionConfig(Config):
    pass


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
