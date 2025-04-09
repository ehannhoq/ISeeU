import argparse

import iseeu
import time
from algorithms import load_wider_data_set

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()
if __name__ == '__main__':
    model = iseeu.Model(
        learning_rate=0.001,
        confidence_threshold=0.8,
        stride=1,
        show_debug=True
        )
    
    max_epochs = args.epochs
    epoch = args.epochs
    i = 0
    while i != -1 and epoch > 0:
        print(f"Epoch: {max_epochs - epoch + 1}/{max_epochs}")
        start_time = time.time()

        images, expected, i = load_wider_data_set(
            imageset_master_path='training_data',
            annotation_file_path='training_data/image_info.txt',
            target_size=(500, 500),
            batch_size=args.batch_size,
            max_faces=10,
            start_index=i
        )

        model.train(
            input=images,
            expected=expected
        )

        end_time = time.time()
        print(f"Time elapsed for epoch {max_epochs - epoch + 1}: {(end_time - start_time):.2f} seconds")
        epoch -= 1