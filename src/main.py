import algorithms
import model


if __name__ == '__main__':

    # TODO: Vectorizations for convolution layers' forward pass and backpropagation.

    batch_size = 32
    neural_network = model.ISeeU(learning_rate=0.005, confidence_threshold=0.8, show_debug=True)
    
    i = 0
    while i != -1:
        dataset, i = algorithms.load_wider_data_set(imageset_master_path='training_data', annotation_file_path='training_data/image_info.txt', batch_size=batch_size, start_index=i)
        neural_network.train(dataset=dataset)
        



    
    
    






        