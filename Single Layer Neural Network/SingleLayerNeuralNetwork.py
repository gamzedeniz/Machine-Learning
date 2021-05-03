import numpy as np # 1.18.0
import pandas as pd # 0.25.3
import matplotlib.pyplot as plt # 3.1.2
#Python 3.6.9
def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))

def sigmoid_derivative(x):
    return 0.005 * x * (1 - x )

def read_and_divide_into_train_and_test(csv_file):
    data=pd.read_csv(csv_file, delimiter = ',')
    new=data.replace(to_replace = "?", value = np.nan) #to drop missing values I am replacing question marks with null
    new_data = new.dropna(axis = 0, how ='any') #dropping missing values
    train=new_data[:546] #taking 80½ of the data
    test=new_data[546:] #taking 20% of the data

    p=train.iloc[:, 1:10].astype(int)
    p=p.corr()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.matshow(p)
    plt.yticks(np.arange(len(p.index)), p.index,fontsize=10)
    plt.xticks(np.arange(len(p.columns)), p.columns,rotation =330,fontsize=7)
    plt.imshow(p, cmap="magma_r")
    plt.colorbar().set_label('Correlation', rotation=90)
    for i in range(len(p.index)):
        for j in range(len(p.columns)):
            plt.text(j, i, str(round(p.iloc[i][j],2)),ha="center", va="center", color="w")
    plt.show()

    train_input=train.values.astype(int)
    training_inputs=np.delete(train_input, 0, axis=1) #not taking code names
    training_inputs=np.delete(training_inputs, -1, axis=1) #not taking class attribitues
    training_labels=np.delete(train_input, [0,1,2,3,4,5,6,7,8,9], axis=1) #taking only class attributes
    test_input=test.values.astype(int)
    test_inputs=np.delete(test_input, 0, axis=1)
    test_inputs=np.delete(test_inputs, -1, axis=1)
    test_labels=np.delete(test_input, [0,1,2,3,4,5,6,7,8,9], axis=1)
    return training_inputs, training_labels, test_inputs, test_labels

def run_on_test_set(test_inputs, test_outputs, weights):
    tp = 0
    test_predictions= sigmoid (np.dot(test_inputs,weights))#calculate test_predictions
    for i in test_predictions:
        if i[0] > 0.5:
            i[0]=1
        else:
            i[0]=0
    #each prediction is either 0 or 1
    for predicted_val, label in zip(test_predictions, test_outputs):
        if predicted_val == label:
            tp += 1
    accuracy = tp / len(test_predictions)
    # accuracy = tp_count / total number of samples
    return accuracy


def plot_loss_accuracy(accuracy_array, loss_array):
    plt.subplot(2, 1, 1)
    plt.plot(loss_array)
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(accuracy_array)
    plt.ylabel('Accuracy')
    plt.xlabel("Epochs/Iterations")
    plt.show()
    #ploting loss and accuracy change for each iteration

def main():
    csv_file = 'breast-cancer-wisconsin.csv'
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)


    for iteration in range(iteration_count):
        outputs= sigmoid(np.dot(training_inputs,weights)) #calculate outputs
        loss = training_labels - outputs #calculate loss
        tuning = loss * sigmoid_derivative(outputs)#calculate tuning
        weights += np.dot(np.transpose(training_inputs), tuning) #update weights
        accuracy_array.append(run_on_test_set(test_inputs, test_labels, weights))#run_on_test_set
        loss_array.append(np.mean(loss))

    accuracy_array=np.array(accuracy_array)
    loss_array=np.array(loss_array)
    plot_loss_accuracy(accuracy_array, loss_array)


main()
