import algorithms
import model


if __name__ == '__main__':

    batch_size = 32
    neural_network = model.ISeeU(confidence_threshold=0.8, learning_rate=0.005, show_debug=True)

    input, expected, i = algorithms.load_wider_data_set(imageset_master_path='training_data', annotation_file_path='training_data/image_info.txt', target_size=neural_network.input_size, batch_size=batch_size, max_faces=10)
    neural_network.train(input=input, expected=expected)
        



    
    
    






        