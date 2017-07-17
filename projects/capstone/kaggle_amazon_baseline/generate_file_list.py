import pandas as pd
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join

def generate_list(fname, new_fname):
    with open(fname, 'r') as from_file, open(new_fname, 'w') as to_file:
        lines = from_file.readlines()[1:]
        for line in lines:
            train_file_name, feature_string = line.split(',')
            train_file_full_dir = './data/train-jpg/' + train_file_name + '.jpg'
            features = feature_string.split(' ')
            features.insert(0, train_file_full_dir)
            new_line = ",".join(features)

            to_file.write(new_line)
        from_file.close()
        to_file.close()

def generate_list2(fname, new_fname):
    with open(fname, 'r') as from_file, open(new_fname, 'w') as to_file:
        lines = from_file.readlines()[1:]
        for line in lines:
            train_file_name, feature_string = line.split(',')
            train_file_full_dir = './data/train-jpg/' + train_file_name + '.jpg'
            new_line = train_file_full_dir + ',' + feature_string

            to_file.write(new_line)
        from_file.close()
        to_file.close()

def generate_test_list(new_fname):
    onlyfiles = [f for f in listdir('./data/test-jpg') if isfile(join('./data/test-jpg', f))]
    #print(onlyfiles)
    
    with open(new_fname, 'w') as to_file:
        for file_name in onlyfiles:
            new_line = './data/test-jpg/' + file_name + '\n'
            to_file.write(new_line)
   
    to_file.close()

def generate_train_test_list(fname, train_list_name, test_list_name):
    train = pd.read_csv(fname)
    X = train['image_name']
    Y = train['tags']

    # TODO: Shuffle and split the data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print (len(X_train))
    print (len(X_test))
    with open(train_list_name, 'w') as train_file:
        for idx, val in X_train.iteritems():
            new_line = X_train[idx] + ',' + y_train[idx] + '\n'
            train_file.write(new_line)
    #        print (idx)
    train_file.close()

    with open(test_list_name, 'w') as test_file:
        for idx, val in X_test.iteritems():
            new_line = X_test[idx] + ',' + y_test[idx] + '\n'
            test_file.write(new_line)
    test_file.close()
    

def main():
    #generate_list2('train_v2.csv', 'train_v2_new.csv')
    #generate_train_test_list('train_v2_new_2.csv', 'train_data.csv', 'test_data.csv')
    generate_test_list('sample_submission_v2_new.csv')
    
if __name__ == "__main__":
    main()

