import algorithms
import model


if __name__ == '__main__':

    neural_network = model.ISeeU(confidence_threshold=0.8, learning_rate=0.005, show_debug=True)

    # TODO: Print out each predicted box's IOU and ground truth; ideally show if each box is a true/false positive/negative.

    dataset_index = 0
    batch_size = 32
    epoch = 1000
    for i in range(epoch):
        input, expected, dataset_index = algorithms.load_wider_data_set(imageset_master_path='training_data', annotation_file_path='training_data/image_info.txt', target_size=neural_network.input_size, batch_size=batch_size, max_faces=10, start_index=dataset_index)
        print("Index:", dataset_index)
        print("Epoch:", i + 1)
        neural_network.train(input=input, expected=expected)

        