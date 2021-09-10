import torch
import numpy as np

from saliency import dataset

dataset_list = ("OSIE", "CAT2000", "MIT1003")
stim_location_list = ("Datasets/OSIE/data/predicting-human-gaze-beyond-pixels-master/data/stimuli", 
                        "Datasets", "Datasets")
function_list = ("DeepGaze", )

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

        for function in function_list:
            eval(function + "(stim, curr_stim_location, curr_dataset)")


def DeepGaze(stim, stim_location, curr_dataset):
    import DeepGaze.deepgaze_pytorch
    import torchvision
    from scipy.special import logsumexp
    from scipy.ndimage import zoom
    import os

    DEVICE = "cuda"

    stim = stim.transpose(0, 3, 1, 2)
    stim = torch.from_numpy(stim)
    stim = stim[0:5]
    stim = torchvision.transforms.functional.resize(stim, [768, 1024])
    stim = stim.detach().numpy().transpose(0, 2, 3, 1)

    print("Stimuli made...")

    # you can use DeepGazeI or DeepGazeIIE
    model = DeepGaze.deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

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

        full = os.path.join(direcory_path, file_name)
        np.save(full, log_density_prediction.numpy())

if __name__ == '__main__':
    main()