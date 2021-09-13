import torch
import numpy as np

from saliency import dataset

dataset_list = ("CAT2000", )
stim_location_list = ("Datasets/OSIE/data/predicting-human-gaze-beyond-pixels-master/data/stimuli", 
                        "Datasets", "Datasets")
function_list = ("FacialDetection", )

DATASET_CONFIG = {
	'data_path' : "Datasets",
	'dataset_json': 'saliency/data/dataset.json',
	'auto_download' : True
}

def main():
    for curr_dataset, curr_stim_location in zip(dataset_list, stim_location_list):
        ds = dataset.SaliencyDataset(config = DATASET_CONFIG)
        ds.load(curr_dataset)

        stim = ds.get("stimuli")

        if curr_dataset != "MIT1003":
            stim = torch.from_numpy(stim)

        for function in function_list:
            eval(function + "(stim, curr_stim_location, curr_dataset)")

def PoseEstimation(stim, stim_location, curr_dataset):
    pass

def FacialDetection(stim, stim_location, curr_dataset):
    import cv2
    import os
    import scipy.misc
    from PIL import Image

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for i, image in enumerate(stim):
        if not torch.is_tensor(image):
            image = torch.tensor(image)
        
        fd_map = torch.zeros(image.shape, dtype = torch.int)

        temp = image.cpu().numpy()
        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        '''
        im = Image.fromarray(temp)
        im.save(f"image{i}.jpeg")
        '''

        for face_no, face in enumerate(faces, start = 1):
            fd_map[face[1]:face[1] + face[3], face[0]:face[0] + face[2]] = face_no

        '''
        fd_map_temp = fd_map.cpu().numpy().astype("uint8") * 255 / len(faces)
        fd_map_temp = fd_map_temp.astype("uint8")
        fd = Image.fromarray(fd_map_temp)
        fd.save(f"imagefd{i}.jpeg")
        '''

        directory_path = os.path.join("Datasets", curr_dataset, "FacialDetectionPriors")
        file_name = "Data" + str(i) + ".npy"

        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        full = os.path.join(directory_path, file_name)
        np.save(full, fd_map)

        print(f"Saved: {full}")


def DeepGaze(stim, stim_location, curr_dataset):
    import DeepGaze.deepgaze_pytorch
    import torchvision
    from scipy.special import logsumexp
    from scipy.ndimage import zoom
    import os
    from scipy.misc import face

    image = face()

    DEVICE = "cuda"

    stim = stim.transpose(0, 3, 1, 2)
    stim = torch.from_numpy(stim)
    stim = torchvision.transforms.functional.resize(stim, [768, 1024])
    stim = stim.detach().numpy().transpose(0, 2, 3, 1)

    print("Stimuli made...")

    # you can use DeepGazeI or DeepGazeIIE
    #model = DeepGaze.deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

    # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
    # you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
    # alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.
    centerbias_template = np.load('centerbias_mit1003.npy')

    print("Center bias made...")

    for i, image in enumerate(stim):
        # rescale to match image size
        centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
        # renormalize log density
        centerbias -= logsumexp(centerbias)

        image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
        centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

        log_density_prediction = model(image_tensor, centerbias_tensor)

        directory_path = os.path.join("Datasets", curr_dataset, "DeepgazePriors")
        file_name = "Data" + str(i) + ".npy"

        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        full = os.path.join(directory_path, file_name)
        np.save(full, log_density_prediction.cpu().detach().numpy())

        print(f"Saved: {full}")

if __name__ == '__main__':
    main()