from sklearn import datasets
import matplotlib.pyplot as plt

print("Extraction of MNIST dataset ...")
mnist = datasets.fetch_openml('mnist_784', as_frame=False) 
images = mnist.data.reshape((-1, 28, 28)) 
plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest") 
plt.show()
print(f'class = {mnist.target[0]}')


import time as t
t.sleep(5)

# Others datasets
# House prices
print("Extraction of boston house prices dataset ...")
boston_house_prices = datasets.load_boston()
print(boston_house_prices.DESCR)
t.sleep(5)
# Diabetes
print("Extraction of diabetes dataset ...")
diabetes = datasets.load_diabetes()
print(diabetes.DESCR)
