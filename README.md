# Sereact-Coding-Challenge

## Files
1. Documentation.pdf contains information about the code regarding data flow, architecture etc.
2. Results has some visualizations of the predicted masks and bounding boxes.




## Installation 
1. Clone the repository
2. Install Dependencies
```
pip3 install -r requirements.txt
```
3. Download the pre-trained weights [here](https://drive.google.com/file/d/1_W6hjik8gdRwgeBmPDxYcKWDnfHbxQnt/view?usp=sharing)

## Usage
1. To train the model
```
python train.py --root_dir /path/to/dataset --checkpoint_path /path/to/save/weights.pth

```

2. To run inference on the pre-trained model

```
python inference.py --model_path /path/to/weights.pth


```
