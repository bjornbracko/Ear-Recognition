import fastai
from fastai.vision.all import *
from fastai.metrics import error_rate # 1 - accuracy
from torchvision import transforms, datasets, models
import matplotlib as plt

item_tfms = [ Resize((224, 224), method='squish')]
batch_tfms = [Brightness(max_lighting = 0.3, p = 0.4), Contrast(max_lighting = 0.6, p = 0.4), Rotate(max_deg=10, p=0.5), RandomCrop(200), Flip(), Saturation()]
data = ImageDataLoaders.from_csv(path='data/perfectly_detected_ears/', csv_fname='ids_train.csv', item_tfms=item_tfms, batch_tfms=batch_tfms)



learn = cnn_learner(data, models.resnet34, metrics=error_rate)

early_stop = EarlyStoppingCallback(monitor='valid_loss', patience=50)
save_best_model = SaveModelCallback(monitor='valid_loss', fname='best_resnet34')

#frozen training step
defaults.device = torch.device('cuda') # makes sure the gpu is used
learn.fit_one_cycle(400, cbs=[early_stop, save_best_model])


learn.load('best_resnet34')
learn.unfreeze()

def find_appropriate_lr(model: Learner, lr_diff: int = 15, loss_threshold: float = .05, adjust_value: float = 1,
                        plot: bool = False) -> float:
    # Run the Learning Rate Finder
    model.lr_find()

    # Get loss values and their corresponding gradients, and get lr values
    losses = np.array(model.recorder.losses)
    min_loss_index = np.argmin(losses)

    # loss_grad = np.gradient(losses)
    lrs = model.recorder.lrs

    # return the learning rate that produces the minimum loss divide by 10
    return lrs[min_loss_index] / 10


optimal_lr = find_appropriate_lr(learn)

learn.fit_one_cycle(400, lr_max=slice(optimal_lr/10, optimal_lr), cbs=[early_stop, save_best_model])


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

interp.plot_top_losses(9, figsize=(15,15))
learn.save('modelx')

'''
#run inference on test images
import glob
from IPython.display import Image, display
import PIL

model = learn.model
model = model.cuda()
for imageName in glob.glob('data/perfectly_detected_ears/test/*.png'):
    print(imageName)
    #img = .Image.open(imageName)
    prediction = learn.predict(imageName)
    #print(prediction)
    print(prediction[0])
    display(Image(filename=imageName))
    print("\n")
'''