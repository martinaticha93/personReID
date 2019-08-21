# personReID
 ## Taylor modules loading
 
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



## Training
To train the network, one needs to have the training data in /data/simple_data_set_train folder. The data needs to be 
stored in separated folders where each folder represents one identity. The folders name must correspond to the 
identity id.

The training starts by running the command python3.6 trainLSTM.py in /videoclassification.
Once the training is finnished, the resulting model together with the identity labels map is stored in the same 
directory.

## Prediction
To predict a label of a sequence, run the python file predict.py in the /videoclassification directory.
The 'DATA_PATH' needs to be specified.