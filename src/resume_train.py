from ultralytics import YOLO

# Load a model
model = YOLO('C:\\Users\\ravis\\OneDrive\\Desktop\\yolo\\runs\\detect\\train3\\weights\\last.pt')  # load a partially trained model

""" Force model to use cuda cores to train the model as it is faster.
    If in case cuda cores are not available then the line can be commented,
    the training will run on CPU.
"""
model = model.cuda()

# Resume training
results = model.train(resume=True)