import tensorflow as tf
from tensorflow.python.estimator.canned import metric_keys

def _accuracy_higher(best_eval_result, current_eval_result):
    default_key = metric_keys.MetricKeys.ACCURACY
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
                'best_eval_result cannot be empty or no accuracy found in it.')
    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
                'current_eval_result cannot be empty or no loss is found in it.')

        return best_eval_result[default_key] > current_eval_result[default_key]
