## Training
To train the network, one needs to to specify the path in the DATA_PATH_TRAIN variable in trainLSTM. The data needs to be 
stored in separated folders where each folder represents one identity. The folders name must correspond to the 
identity id. For each identity, there must be the sequences of 20 images in the folder and each folder must contain 
4 - 10 sequences.

The training starts by running the command python3.6 trainLSTM.py in /src.
Once the training is finished, the resulting model together with the identity labels map is stored in the same 
directory.

## Prediction
To predict a label of a sequence, run the python file predict.py in the /src directory.
The 'DATA_PATH' needs to be specified.

# personReID
 ## Taylor modules loading
 For training of the neural network, we use the Taylor server which is available to all students at CTU. 
 To load all necessary libraries at the server, run the following commands:
 
module load Keras/2.2.2-goolfc-2017b-Python-3.6.4

module load OpenCV/3.4.2-fosscuda-2018b-Python-3.6.4 

module load CUDA/9.0.176-GCC-6.4.0-2.28

module load cuDNN/7.2.1.38-goolfc-2017b

module unload OpenCV/3.4.2-fosscuda-2018b-Python-3.6.4

module load OpenCV/3.4.2-foss-2018b-Python-3.6.4 

module unload gcccuda/2018b

module load gcccuda/2017b

module load GCC/7.3.0-2.30

module load cuDNN/7.2.1.38-goolfc-2017b

module load scikit-learn/0.19.2-foss-2018b-Python-3.6.4 

nvidia-smi

export CUDA_VISIBLE_DEVICES=7



