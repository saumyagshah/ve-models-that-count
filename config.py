# paths
qa_path = 'data'  # directory containing the question and annotation jsons
# directory containing the .tsv file(s) with bottom up features
bottom_up_trainval_path = 'data/trainval'
# directory containing the .tsv file(s) with bottom up features
bottom_up_test_path = 'data/test2015'
# path where preprocessed features from the trainval split are saved to and loaded from
preprocessed_trainval_path = 'genome-trainval.h5'
# path where preprocessed features from the test split are saved to and loaded from
preprocessed_test_path = 'genome-test.h5'
# path where the used vocabularies for question and answers are saved to
vocabulary_path = 'vocab.json'

task = 'OpenEnded'
dataset = 'mscoco'

test_split = 'test2015'  # either 'test-dev2015' or 'test2015'

# preprocess config
output_size = 100  # max number of object proposals per image
output_features = 1024  # number of features in each object proposal # CHANGED

# training config
epochs = 5
batch_size = 256
initial_lr = 1.5e-3
lr_halflife = 50000  # in iterations
data_workers = 4
max_answers = 3  # CHANGED
