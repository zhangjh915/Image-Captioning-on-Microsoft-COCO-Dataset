import numpy as np
import matplotlib.pyplot as plt
from CaptionRNN import RNNImageCaption
from train import CaptionTrain
from utils import *

np.random.seed(231)

data = load_coco_dataset(PCA_features=True)
small_data = load_coco_dataset(max_train=10000)

small_rnn_model = RNNImageCaption(cell_type='rnn', word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1], hidden_dim=512, wordvec_dim=256,)

small_rnn_solver = CaptionTrain(small_data, small_rnn_model, update='adam', num_epochs=50,
                                batch_size=100, update_params={'lr': 5e-3,}, lr_decay=0.95, print_freq=10)

small_rnn_solver.train()

# Plot the training losses
plt.plot(small_rnn_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()
