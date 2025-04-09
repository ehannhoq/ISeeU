import iseeu
import time
from algorithms import load_wider_data_set


if __name__ == '__main__':
    model = iseeu.Model(
        learning_rate=0.001,
        confidence_threshold=0.8,
        stride=1,
        show_debug=True
        )
    
    epoch = 0
    i = 0
    while i != -1 and epoch < 1000:
        print(f"Epoch: {epoch}")
        start_time = time.time()

        images, expected, i = load_wider_data_set(
            imageset_master_path='training_data',
            annotation_file_path='training_data/image_info.txt',
            target_size=(500, 500),
            batch_size=1,
            max_faces=10,
            start_index=i
        )

        model.train(
            input=images,
            expected=expected
        )

        end_time = time.time()
        print(f"Time elapsed for epoch {epoch + 1}: {(end_time - start_time):.2f} seconds")
        epoch += 1