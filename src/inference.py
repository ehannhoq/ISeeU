from matplotlib import patches, pyplot as plt
import numpy as np
import matplotlib.image as mpimg

import model
import algorithms

if __name__ == "__main__":
    # image_path = input("Enter the path to the image: ")
    path = "training_data/8--Election_Campain/8_Election_Campain_Election_Campaign_8_36.jpg"
    image = mpimg.imread(path)

    neural_network = model.ISeeU(confidence_threshold=0.5, show_debug=True)
    predictions = neural_network.inference(image)