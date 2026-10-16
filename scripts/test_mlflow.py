import os
os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///C:/Users/isha/OneDrive/Documents/TimeSeries/m5-forecasting-platform/mlflow.db'
os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'

import mlflow
mlflow.set_experiment('m5-lgbm')
print('Tracking URI :', mlflow.get_tracking_uri())
print('Experiment   :', mlflow.get_experiment_by_name('m5-lgbm').name)
print('ALL OK')
