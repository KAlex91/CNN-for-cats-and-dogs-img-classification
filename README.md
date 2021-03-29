# CNN-for-cats-and-dogs-img-classification

This project was inspired by sendex. I tried many different configurations for my CNN model to classify cat and dog images. 
I tried making this CNN with all the following configurations:\
Number of Conv Layers : 2,3 \
Number of Dense Layers: 1,2 \
Conv Layer Nodes: 64,128 \
Dense Layer Nodes: 128,256\
Conv Dim : 3x3,4x4 \
Which is 2^5=32 total combinations.
If u want to train the same model, u need to run the "data preprocessing.py" file to create the X.pickle and y.pickle files and train the model using "main.py".
Alternatively, u can use the saved model located in the "128x3_CNN.model" file.
For this project i used my gpu, which at present is a GTX 1050ti. All the models together took me about one and a half hours. If you want you can try google colab to run it.
