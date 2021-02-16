# Plant-disease-Classifier
My approach for building such a model was first of all we do image acquisition, the point to note here is that image is taken as a matrix of pixels then it gets convolves with a feature detector( a random 3x3 matrix) the convolved result is called a feature map. 
This feature map then needs be converted to a pooled feature map because we need a smaller version of the convoluted feature map, so for this we use Maxpooling after every convolutional layer (cnn layer). 
After some number of such cnn layers depending upon complexity of training image type for example we have used 4 cnn layers starting from 32 to 64 then 128 and again 128 neurons note that always increase the next layer size in 2’s multiple a pyramidal kind of structure which gets effectively trained than any other type. After cnn layers we add a flattening layer which converts a multidimensional array as a 1D array. We can also add dropout to prevent overfitting issues.



After all the cnn layers we add fully connected layer(s) which is also called as hidden layer. I have tried training the model with different number of FC layers but the optimal result was obtained by adding a single FC layer with 256 neurons.

At the end we add a output layer with number of neurons more than or equal to number of classes among which we want our model to classify input images.
At last after choosing appropriate optimizer and lossfunction depending on binary or multiclass classification.
After creating and saving an appropriate model I have developed a predictor file which loads the previously saved hdf5 format model and predicts the appropriate class as output at the console.
Concluding my thoughts when I see developing an optimal method I could see that making a sequential model is not that tough but continuously checking for improvements and patiently training the model with n number of specifications every time is the key to get a efficient model
