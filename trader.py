# import from library
import argparse
import csv

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class Trader():    
    condition = 0
    predict_result = []
    def checkCondition( self, pre, last):
        ratio = 0.0041
        value = (last - pre) / pre

        if(-0.5 * ratio <= value and value < ratio * 0.5):
            return 0
        elif(0.5 * ratio <= value and value < ratio * 1.5):
            return 1
        elif(1.5 * ratio <= value and value < ratio * 2.5):
            return 2
        elif(2.5 * ratio <= value and value < ratio * 3.5):
            return 3
        elif(3.5 * ratio <= value):
            return 4
        elif(-1.5 * ratio <= value and value < ratio * -0.5):
            return -1
        elif(-2.5 * ratio <= value and value < ratio * -1.5):
            return -2
        elif(-3.5 * ratio <= value and value < ratio * -2.5):
            return -3
        elif(value < ratio * -3.5):
            return -4
    
    def train( self, data ):
        # produce training data
        train_X = []
        train_Y = []
        
        for i in range(1, len(data)):
            train_X.append(data[i-1])
            train_Y.append(self.checkCondition(data[i-1][0], data[i][0]))
        
        validating_X = train_X[-150:]
        validating_Y = train_Y[-150:]
        train_X = train_X[:-150]
        train_Y = train_Y[:-150]
        
        # cross validation process
        #parameters = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}
        #self.CV = GridSearchCV(estimator = self.clf, param_grid = parameters, cv=5)
        #self.CV.fit(train_X, train_Y)
        #print (self.CV.best_params_)
        #print (self.CV.best_score_)
        
        self.clf = KNeighborsClassifier(n_neighbors = 3)
        self.clf.fit(train_X, train_Y)
        return
    
    def predict_action( self, data ):
        result = self.clf.predict(data)
        self.predict_result.append(result[0])
        if(result[0] >= 1):
            #rise
            if(self.condition == 1):
                self.condition = 0
                return -1
            elif(self.condition == 0):
                self.condition = -1
                return -1
            elif(self.condition == -1):
                return 0
        
        elif(result[0] <= -1):
            #fall
            if(self.condition == 1):
                return 0
            elif(self.condition == 0):
                self.condition = 1
                return 1
            elif(self.condition == -1):
                self.condition = 0
                return 1
        
        elif(result[0] == 0):
            #nothing happen
            return  0

def load_data( str ):
    with open(str, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        tmp = []
        for row in data:
            tmp.append(row)
        return tmp


if __name__ == '__main__':
    # program parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # training_data form: [open, high, low, close]
    # load in training & testing data
    training_data = load_data(args.training)
    testing_data = load_data(args.testing)
    
    trader = Trader()
    trader.train(training_data)
    
    money = 0
    start, end = 0, len(testing_data)-1
    
    with open(args.output, 'w') as output_file:
        condition = trader.condition
        for i in range(start , end):
            # We will perform your action as the open price in the next day.
            condition = trader.condition
            action = trader.predict_action([testing_data[i]])
            
            if(condition == action and action != 0):
                print(action, trader.condition)
            
            output_file.write(str(action)+'\n')
    
