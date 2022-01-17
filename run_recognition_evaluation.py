import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist
from metrics.evaluation_recognition import Evaluation
import fastai
from fastai.vision.all import *
from fastai.metrics import error_rate # 1 - accuracy
from torchvision import transforms, datasets, models
from fastai.data.external import *

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config_recognition.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []

        eval = Evaluation()

        cla_d = self.get_annotations(self.annotations_path)
        
        # Change the following extractors, modify and add your own
        item_tfms = [Resize((224, 224), method='squish')]
        data = ImageDataLoaders.from_csv(path='data/perfectly_detected_ears/', csv_fname='ids_train.csv',
                                         item_tfms=item_tfms)
        learn = cnn_learner(data, models.resnet34, metrics=error_rate)
        learn.load('aug_best')
        model = learn.model
        model = model.cuda()

        y = []
        preds = []
        preds_prob = []
        res = []
        c = 0

        for im_name in im_list:
            img = PILImage(PILImage.create(im_name))#.resize((224, 224)))
            prediction = learn.predict(img)

            true_label = cla_d['/'.join(im_name.split('/')[-2:])]
            preds_prob.append(np.array(prediction[2]))
            y.append(true_label)

            #print(true_label, ' ', prediction[0])
            #print(prediction)
            preds.append(prediction[0])
            if int(prediction[0]) == int(true_label):
                res.append(1)
            else:
                res.append(0)
            c += 1
        #print(sum(res))
        print("Accuracy: ", sum(res)/c)
        preds_prob = np.stack(preds_prob)

        Y_plain = cdist(preds_prob, preds_prob, 'jensenshannon')

        r1 = eval.compute_rank1(Y_plain, y)
        print('Rank-1[%]', r1)



if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()