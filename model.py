import yaml
from ludwig import LudwigModel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

model_definition = yaml.load(open('model_definition.yaml', 'r'))

ludwig_model = LudwigModel(model_definition)

train_stats = ludwig_model.train(data_csv='./chitchat.csv')

model = ludwig_model.load('results/_run_0/model')

results = model.predict(data_csv='./chitchat.csv')

print(results)

model.close()