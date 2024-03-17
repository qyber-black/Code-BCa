import numpy
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from scipy.ndimage.measurements import label, find_objects
from scipy.stats import pearsonr
import difflib
import scipy.spatial
import numpy as np
import os
import glob
import SimpleITK as sitk
import medpy
import scipy.spatial
import medpy.metric.binary as medpyMetrics

# Define the evalation matrics 
def getHausdorff(testImage, resultImage):
    """Compute the Hausdorff distance."""

    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float('nan')

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage   = sitk.BinaryErode(testImage, (1,1,0) )
    eResultImage = sitk.BinaryErode(resultImage, (1,1,0) )

    hTestImage   = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)

    hTestArray   = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)

    # Convert voxel location to world coordinates. Use the coordinate system of the test image

    testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
    resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]


    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]

    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)

    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))
def hd95(result, reference, voxelspacing=None, connectivity=1):
    try:
        hd1 = medpyMetrics.__surface_distances(result, reference, voxelspacing, connectivity)
        hd2 = medpyMetrics.__surface_distances(reference, result, voxelspacing, connectivity)
        hd95 = numpy.percentile(numpy.hstack((hd1, hd2)), 95)
    except:
        hd95 = 95
    return hd95
def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray   = sitk.GetArrayFromImage(testImage).flatten()
    resultArray = sitk.GetArrayFromImage(resultImage).flatten()

    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)

def recall(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def sensitivity(result, reference):

    return recall(result, reference)
def specificity(result, reference):

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0

    return specificity
Dice_all=[]
H95_all=[]


# get the evaluation matrics results per_sample (per_patient)
import math
def get_score_per_sampe(my_model,image, maks):
           Disces=[list() for i in range(0,3)]
           Haus95s=[list() for i in range(0,3)]
           Specs=[list() for i in range(0,3)]
           Senss=[list() for i in range(0,3)]
           domains=['whole','Core','Enhance'] #[0,1,2]
           predcition=my_model.predict(image)
           prediction=np.argmax(predcition, axis=4)[0,:,:,:]

           for k, dom in enumerate(domains):
                if k==0:
                    test_prediction = (prediction>0.4).astype(int)
                    msk=(mask >0.4).astype(int)
                elif k==1:
                    test_prediction1=(prediction ==1).astype(int)
                    test_prediction2=(prediction ==3).astype(int)
                    test_prediction1=(test_prediction1 >0.4).astype(float)
                    test_prediction2=(test_prediction2 >0.4).astype(float)
                    test_prediction=test_prediction1 + test_prediction2
                    test_prediction=(test_prediction >0.4).astype(int)

                    msk1=(mask ==1).astype(int)
                    msk2=(mask ==3).astype(int)
                    msk1=(msk1 >0.4).astype(float)
                    msk2=(msk2 >0.4).astype(float)
                    msk=msk1+msk2
                    msk=(msk > 0.3).astype(int)
                   # print(np.unique(msk),np.unique(test_prediction))
                else:
                    test_prediction =(prediction ==3).astype(int)
                    test_prediction =(test_prediction >0.4).astype(int)
                    msk=(mask ==3).astype(int)

                Spec=specificity(test_prediction , msk)
                Sens=sensitivity(test_prediction , msk)
                image_sitk = sitk.GetImageFromArray(msk)
                image_sitk1 = sitk.GetImageFromArray(test_prediction)
                Dice=getDSC(image_sitk, image_sitk1)
                if math.isnan(Dice):
                    Dice=1
                print('dice ---- ',Dice)

                Haus=hd95(test_prediction , msk, voxelspacing=None, connectivity=1)
                Senss[k].append(100*Sens)
                Specs[k].append(100*Spec)
                Disces[k].append(100*Dice)
                Haus95s[k].append(Haus)
           return  Senss,Specs,Disces,Haus95s

# To Evaluate 
from keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization
import pickle
import numpy as np
import glob
import os

# Set the base directory dynamically
base_directory = os.getcwd()

# Define the dataset path dynamically
images_path = os.path.join(base_directory, 'Results','data2020')  # Adjust as needed
models_dir = os.path.join(base_directory, 'Results','2020')  # Adjust as needed

# Load train and validation list
folds_file_path = os.path.join(base_directory, 'Results','folds_dic.pkl')
with open(folds_file_path, "rb") as open_file:
    trainvallist = pickle.load(open_file)

# Initialize lists for metrics
Dices_all = []
H95s_all = []
Senss_all = []
Specs_all = []

for k in range(0, 5):
    model_for_this_fold = [m for m in os.listdir(models_dir) if '2020fold_' + str(k) in m]
    model_path = os.path.join(models_dir, model_for_this_fold[0])
    model = load_model(model_path, compile=False)

    val_list = trainvallist['validation'][k][:69]
    print('******', len(val_list))
    Dice_fold, H95_fold, Sens_fold, Spec_fold = [], [], [], []

    for i, dir_name in enumerate(val_list):
        print('*********', i, dir_name)
        image_path = glob.glob(os.path.join(images_path, dir_name, 'image_*.npy'))[0]
        image = np.load(image_path)
        image = np.expand_dims(image, 0)

        mask_path = glob.glob(os.path.join(images_path, dir_name, 'mask_*.npy'))[0]
        mask = np.load(mask_path)

        Senss, Specs, Disces, Haus95s = get_score_per_sample(model, image, mask)  
        Dice_fold.append(Disces)
        H95_fold.append(Haus95s)
        Spec_fold.append(Specs)
        Sens_fold.append(Senss)

    Dices_all.append(Dice_fold)
    H95s_all.append(H95_fold)
    Specs_all.append(Spec_fold)
    Senss_all.append(Sens_fold)

# Save the metrics
for metric_name, metric_data in zip(['dscs2020_paper1', 'hds2020_paper1', 'specs2020_paper1', 'sensis2020_paper1'], 
                                    [Dices_all, H95s_all, Specs_all, Senss_all]):
    with open(os.path.join(base_directory, metric_name + '.pkl'), "wb") as open_file:
        pickle.dump(metric_data, open_file)

# How to use it and plot the results:
# plot_scores([dsc_scores, hd95_scores], ['DSC', 'HD95'])
# plot_scores_per_region(1,metric='HD95')
