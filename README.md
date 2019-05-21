# CMSC 498L Final Project

# Link to Presentation
[https://docs.google.com/presentation/d/1SqJgX45OKA-G44TQ-mGuCZ5S14A1APbDf60kIxYxTco/edit?usp=sharing]

## Dataset

### Link
[https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8]

### Instructions
- Download the dataset from the google drive link as a `zip` file to the `data/` directory
- Unzip `celeba-dataset.zip`
- Unzip `img_align_celeba.zip`
- Remove the `.zip` files (optional)

## Training
- A sample script for training has been provided in `train.py`
- If more configurations want to be made, refer to the `train()` function in `model.py`
- Simply call `python3 train.py`

## Testing
- A sample script for testing has been provided in `test.py`
- If more configurations want to be made, refer to the `test()` function in `model.py`
- Simply call `python3 test.py`

## Current Results
- The network is saved within the `model_weights` directory
- `model.py` has a function, `load_weights()`, that loads the saved weights from the directory
- When testing, the default behavior is to load the weights from the specified directory
