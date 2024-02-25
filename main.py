from src.core.helpers import N_LETTERS, category_from_output, get_batchs, load_data, show
from src.train import training
from src.models.rnn import RNN
import torch.nn as nn
import torch

# Cleaning CUDA
torch.cuda.empty_cache()

# Dataset
category_lines, all_categories = load_data()

# Hyperparameters
n_categories = len(all_categories)
n_hiddem = 128

# Model
model = RNN(N_LETTERS, n_hiddem, n_categories)
criterion = nn.NLLLoss()
l_r = 0.005
optim = torch.optim.SGD(model.parameters(), lr=l_r)


# Training parameters
current_loss = 0
losses = []
plot_steps, print_steps = 1000, 500
n_epochs = 200000

# Training loop
for epoch in range(1, n_epochs):
    
    print(' \n *********** Epoch {} *********** \n'.format(epoch))
    
    category, line, category_tensor, line_tensor = get_batchs(category_lines, all_categories)

    output, loss = training(model, criterion, optim, line_tensor, category_tensor)  

    current_loss += loss 

    if (epoch) % plot_steps == 0:
        losses.append(current_loss / plot_steps)        
        current_loss = 0 # restart
        
    if (epoch) % print_steps == 0:
        guess = category_from_output(output, all_categories)
        correct = "Certo" if guess == category else f"Errado ({category})"
        print(f"{epoch} {(epoch)/n_epochs*100} {loss:.4f} {line} / {guess} {correct}")
        
# Cleaning CUDA
torch.cuda.empty_cache()

show(losses)