from sklearn.pipeline import FeatureUnion
from . import sentiments

__all__ = ['sentiments', 'get_feature', 'create_feature']


def get_feature(f_name):
    """Factory to create features objects

    Parameters
    ----------
    f_name : features name

    Returns
    ----------
    features: BaseEstimator
        feture object

    """
    features_dic = dict(

        nrc_mean_sd_10=sentiments.NRCSentimentFlowMeanSD(chunk_size=10),
        nrc_sum_10=sentiments.NRCSentimentFlowSum(chunk_size=10),

        nrc_mean_sd_20=sentiments.NRCSentimentFlowMeanSD(chunk_size=20),
        nrc_sum_20=sentiments.NRCSentimentFlowSum(chunk_size=20),

        nrc_mean_sd_30=sentiments.NRCSentimentFlowMeanSD(chunk_size=30),
        nrc_sum_30=sentiments.NRCSentimentFlowSum(chunk_size=30)

    )

    return features_dic[f_name]


def create_feature(feature_names):
    """Utility function to create features object

    Parameters
    -----------
    feature_names : features name or list of features names


    Returns
    --------
    a tuple of (feature_name, features object)
       lst features names are joined by -
       features object is the union of all features in the lst

    """
    try:
        # print (feature_names)
        if isinstance(feature_names, list):
            return ("-".join(feature_names), FeatureUnion([(f, get_feature(f)) for f in feature_names]))
        else:

            return (feature_names, get_feature(feature_names))
    except Exception as e:
        print(e)
        raise ValueError('Error in function ')
