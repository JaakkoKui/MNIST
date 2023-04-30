from PIL import Image, ImageOps
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#Ladataan MNIST datasetti
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Kuvat laitetaan grayscaleen
x_train = x_train / 255.0
x_test = x_test / 255.0

#Malli rakennetaan, ensin se muutetaan 28x28 kuvasta 1x784 vektoriksi, sitten hidden layereihin laitetaan 250 neuronia kumpaankin ja lopuksi ulos tulee 10 neuronin layeri, jokaiselle numerolle 0-9 oma.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

#Malli opetetaan ja sen tarkkuus testataan
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

#Kuopattu numeronpiirrosfunktio
'''def draw_number():
    image = np.zeros((28, 28))
    fig = plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show(block=False)
    while True:
        plt.waitforbuttonpress()
        if plt.get_current_fig_manager().toolbar.mode != '':
            break
        points = plt.ginput(1)
        if points:
            x, y = points[0]
            x = int(x)
            y = int(y)
            image[y, x] = 1.0
            plt.imshow(image, cmap='gray')
            plt.draw()
    plt.close(fig)
    return image
'''

#Kuvan preprocessaus
def preprocess_image(img):
    #Muuta kuva grayscaleksi
    img = img.convert('L')
    
    #Muuta kuva 28x28 pikselin kokoiseksi
    img = img.resize((28, 28))
    
    #Inverttaa värit
    img = ImageOps.invert(img)
    
    #Muuta kuva numpy arrayksi
    img_array = np.array(img) / 255.0
    
    #Muuta kuva 1x28x28 vektoriksi
    img_array = img_array.reshape(1, 28, 28)
    
    return img_array

#Käyttäjä syöttää kuvan polun, se preprocessataan ja ennustetaan numero, näytetään alkuperäinen kuva, prosessoitu kuva ja tulostetaan ennustettu numero
while input!='q':
    image_path = input('Enter the path to the image file: ')
    
    #Lataa annettu kuva
    img = Image.open(image_path)
    
    #Preprosessoi se
    img_array = preprocess_image(img)
    
    #Arvaa numero
    prediction = model.predict(img_array)
    print('Kurssiarvosana:', np.argmax(prediction))
    
    #Näytä kuva
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.show()
    
#Failure4
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

# Load new images
image6 = Image.open('image6.png').convert('L')
images = [image6]

# Preprocess the new images
preprocessed_images = []
for image in images:
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = 1 - (image_array / 255.0)
    preprocessed_images.append(image_array)

#write that shows the image
plt.imshow(image6, cmap='gray')

# Make predictions on the new images
for image in preprocessed_images:
    prediction = model.predict(np.array([image]))
    print('Prediction:', np.argmax(prediction))
    '''
#Failure3
'''
import tensorflow as tf
import keras as K
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from PIL import Image
from keras.datasets import mnist

# Load Data to train, validation and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the input data
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define the test images
test_images = ['image3.png']

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=6)

# Evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# Save the model
model.save('epic_num_reader.model')

# Load the model
new_model = tf.keras.models.load_model('epic_num_reader.model')


def preprocess_image(image_path):
    # Load the image and convert to grayscale
    image = Image.open(image_path).convert('L')

    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the pixel values to be between 0 and 1
    image_array = image_array / 255.0

    # Display the preprocessed image
    plt.imshow(image, cmap='gray')
    plt.show()

    # Reshape the image to be compatible with the model's input shape
    image_array = image_array.reshape(1, 28, 28, 1)

    # Normalize the image data
    image_array = tf.keras.utils.normalize(image_array, axis=1)

    # Display the normalized image
    plt.imshow(image_array.reshape((28, 28)), cmap='gray')
    plt.show()

    return image_array

# Loop through the test images and make predictions
for image_path in test_images:
    preprocessed = preprocess_image(image_path)
    prediction = new_model.predict(preprocessed)
    digit = np.argmax(prediction)
    print(f"Predicted digit: {digit}")
    
'''
#Failure2
'''
#Third try, didnt work because images  wrong somehow

import tensorflow as tf
import keras as K
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.datasets import mnist

# Load Data to train, validation and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()


image1 = Image.open('image1.png')
image2 = Image.open('image2.png')
image3 = Image.open('image3.png')

test_images = [image1, image2, image3] 

def preprocess_image(image):
    image = Image.open(image)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.flatten() / 255.0 # normalize the pixel values
    image = image.reshape(1, 28, 28)
    return image


# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)]
)

# Compile the model
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=3)

# Evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# Save the model
model.save('epic_num_reader.model')

# Load the model
new_model = tf.keras.models.load_model('epic_num_reader.model')

for image in test_images:
    preprocessed = preprocess_image(image)
    print(preprocessed)

    prediction = new_model.predict(preprocessed)
    print(prediction)


'''
#Failure1
'''
#Second try, found better and clearer model

train = mnist.train.num_examples
validation = mnist.validation.num_examples
test = mnist.test.num_examples

input_size = 784
hidden1_size = 50
hidden2_size = 50
hidden3_size = 50
output_size = 10

learning_rate = 0.001
training_epochs = 100
batch_size = 100
dropout_rate = 0.5

X = tf.placeholder(tf.float32, shape=[None, input_size])
Y = tf.placeholder(tf.float32, shape=[None, output_size])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'W1': tf.Variable(tf.truncated_normal(shape=[input_size, hidden1_size], stddev=0.1)),
    'W2': tf.Variable(tf.truncated_normal(shape=[hidden1_size, hidden2_size], stddev=0.1)),
    'W3': tf.Variable(tf.truncated_normal(shape=[hidden2_size, hidden3_size], stddev=0.1)),
    'W4': tf.Variable(tf.truncated_normal(shape=[hidden3_size, output_size], stddev=0.1))
}

biases = {
    'B1': tf.Variable(tf.zeros(shape=[hidden1_size])),
    'B2': tf.Variable(tf.zeros(shape=[hidden2_size])),
    'B3': tf.Variable(tf.zeros(shape=[hidden3_size])),
    'B4': tf.Variable(tf.zeros(shape=[output_size]))
}

layer_1 = tf.add(tf.matmul(X, weights['W1']), biases['B1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['W2']), biases['B2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['W3']), biases['B3'])
layer_3 = tf.nn.dropout(layer_3, keep_prob=keep_prob)
output_layer = tf.add(tf.matmul(layer_3, weights['W4']), biases['B4'])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=Y))

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_loss = 0
    total_batch = int(train / batch_size)

    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, loss_val = sess.run([train_step, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout_rate})
        avg_loss += loss_val / total_batch

    print("Epoch: {0}, Loss: {1}".format(epoch, avg_loss))

print("Accuracy: ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))

'''