# Age_estimation_Application
This repo is for estimate the human age relying on their face

## Data preparation
- Using the UTK dataset.
- Link [download](https://susanqq.github.io/UTKFace/) dataset
- Note:<br> 
You should choose the processed data (after align and crop) name: ```Aligned and Cropped Faces, ZIP file 107MB```<br>
Age label is the first digit number of image's name (ex: image name: ```61_0_0_201701043423453.jpg.chip.jpg``` will have age label: ```61```)
- Extract and copy to the folder ```./dataset```
- run ```python create_file_csv.py``` to create 2 files ```data_train.csv``` and ```data_test.csv```. These files contain all paths to training images and testing images.
## Model
- We use the ResNet18 to train this task. The model is defined in ```Resnet.py```
## Training
- To train the model, run ```python train.py```
- The checkpoints will saved in folder ```./saved_models``` after every epoch
## Testing
- Modify the path of testing image you want in ```test.py```. Then, run ```python test.py``` and see the results
