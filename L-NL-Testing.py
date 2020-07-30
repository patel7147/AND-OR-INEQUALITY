import numpy as np
from perceptron_L_NL import Perceptron

training_inputs = []
training_inputs.append(np.array([1,1,1]))
training_inputs.append(np.array([1,0,1]))
training_inputs.append(np.array([0,1,0]))
training_inputs.append(np.array([0,0,0]))

labels = np.array([1,0,0,0])

perceptron =Perceptron(3)

perceptron.train(training_inputs,labels)

inputs = np.array([1,1,1])
print(f"Predicted Output: {perceptron.predict(inputs)}")
print(f"Actual Output : {labels[0]}")

print("----------------")

inputs = np.array([1,0,1])
print(f"Predicted Output: {perceptron.predict(inputs)}")
print(f"Actual Output : {labels[1]}")

print("----------------")

inputs = np.array([0,1,0])
print(f"Predicted Output: {perceptron.predict(inputs)}")
print(f"Actual Output : {labels[2]}")

print("----------------")

inputs = np.array([0,0,0])
print(f"Predicted Output: {perceptron.predict(inputs)}")
print(f"Actual Output : {labels[3]}")



