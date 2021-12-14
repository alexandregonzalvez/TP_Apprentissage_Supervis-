from sklearn import datasets
import matplotlib.pyplot as plt
mnist = datasets.fetch_openml('mnist_784', as_frame=False) 
images = mnist.data.reshape((-1, 28, 28)) 
plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest") 
plt.show()
print(f'class = {mnist.target[0]}')