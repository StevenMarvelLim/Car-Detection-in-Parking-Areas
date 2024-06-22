import tensorflow as tf 
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

#Import Gambar
train_dir = "./Dataset"
data = tf.keras.utils.image_dataset_from_directory(train_dir, 
                                                   image_size=(30,30),
                                                   label_mode='categorical',
                                                   batch_size=16)
data_name = data.class_names
print(data_name)
data1 = data.map(lambda x, y : (x/255 , y))
data_new = data1.as_numpy_iterator().next()
print(data_new)

# Models
class VGG3(tf.keras.Model): 
    def __init__(self,n_class , *args, **kwargs):
        super(VGG3 , self).__init__(*args, **kwargs)
        self.La1 = tf.keras.Sequential(layers=[
            layers.Conv2D(32 , (3,3) , padding='same'), 
            layers.ReLU(), 
            layers.BatchNormalization(axis = -1),
            layers.MaxPooling2D((2,2)), 
            layers.Dropout(0.25)
        ])
        self.flat = layers.Flatten()
        self.linear1 = layers.Dense(128 , 
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.linear2 = layers.Dropout(0.25)
        self.Rel = layers.ReLU()
        self.soft = layers.Softmax()
        self.out = layers.Dense(n_class)
        self.bcn = layers.LayerNormalization()
    def call(self, inputs):
        x = self.La1(inputs)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.Rel(x)
        x = self.bcn(x)
        x = self.linear2(x)
        x = self.out(x)
        x = self.soft(x)
        return x 
        
ModelVGG = VGG3(2)

print(ModelVGG(tf.random.normal((1,30,30,3))))

ModelVGG.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.001 , momentum=0.9 , nesterov=True),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

ModelVGG.fit(data1 ,epochs=3 , use_multiprocessing=True , batch_size=16)
ModelVGG.save_weights('VGGw3.weight.h5')
ModelVGG.load_weights('VGGw3.weight.h5')
imgs1 = tf.keras.utils.load_img('not_empty\\00000846_00000199.jpg' , target_size=(30,30,3))
imgs = tf.keras.utils.img_to_array(imgs1)/255
imgs = tf.expand_dims(imgs , 0)
prediks = ModelVGG(imgs)
plt.title(f'{data_name[np.argmax(prediks.numpy())]}')
plt.imshow(imgs1)
print(prediks)
print(data_name[np.argmax(prediks.numpy())])
plt.show()

imgs2 = tf.keras.utils.load_img('empty\\00000000_00000280.jpg' , target_size=(30,30,3))
imgs = tf.keras.utils.img_to_array(imgs2)/255
imgs = tf.expand_dims(imgs , 0)
prediks = ModelVGG(imgs)
plt.title(f'{data_name[np.argmax(prediks.numpy())]}')
plt.imshow(imgs2)
print(prediks)
print(data_name[np.argmax(prediks.numpy())])
plt.show()
