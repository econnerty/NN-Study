from tensorflow.python.summary.summary_iterator import summary_iterator
import csv
import os

# Path to the TensorBoard log directory
logdir = '../logs/fruitmodel'

# Create a CSV file
with open('tensorboard_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['step', 'value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over all event files
    for event_file in os.listdir(logdir):
        for e in summary_iterator(os.path.join(logdir, event_file)):
            # Each event contains a `Summary` object
            for v in e.summary.value:
                # The `tag` property of the `Summary` object is the name we provided to `add_scalar`
                if v.tag == 'Training Loss':
                    writer.writerow({'step': e.step, 'value': v.simple_value})
