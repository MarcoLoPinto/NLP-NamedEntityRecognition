import torch
from torch.utils.data import DataLoader

from seqeval.metrics import accuracy_score, f1_score

def train_and_evaluate(
    final_model,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader = None,
    valid_data_input = None,
    valid_data_output = None,
    epochs: int = 5,
    verbose: bool = True,
    device = 'cpu'
):

    """
    Training and evaluation function in order to save the best model \n
    Args:
        - final_model: the wrapped model created with FinalModel class, in order to simulate implementation.py
        - optimizer: the torch.optim.Optimizer used 
        - train_dataloader: the train data created with torch.utils.data.Dataloader
        - valid_dataloader: the dev data created with torch.utils.data.Dataloader 
        - valid_data_input: the dev data composed of list of senteces, in order to simulate the evaluation of implementation.py
        - valid_data_output: the dev data composed of list of labels, in order to simulate the evaluation of implementation.py 
        - epochs: number of maximum epochs 
        - verbose: if True, then each epoch will print the training loss, the validation loss and the f1-score
        - device: if we are using cpu or gpu \n
    Returns:
        a dictionary of histories:
            train_history: list of training loss values 
            valid_loss_history: list of validation loss values 
            valid_f1_history: list of f1 values
    """

    use_stop = False
    strong_stop = False
    warning_stop = strong_stop
    threshold = 0.01

    train_history = []
    valid_loss_history = []
    valid_f1_history = []

    final_model.model.to(device)

    for epoch in range(epochs):
        losses = []
        
        final_model.model.train()

        # batches of the training set
        for step, sample in enumerate(train_dataloader):
            inputs = sample['inputs'].to(device)
            chars = sample['chars'].to(device)
            labels = sample['outputs']
            
            optimizer.zero_grad()
            
            predictions = final_model.model.compute_outputs(inputs, chars)

            if final_model.model.crf is None:
                predictions = predictions.reshape(-1, predictions.shape[-1]) # (batch , sentence , n_labels) -> (batch*sentence , n_labels)
                labels = labels.view(-1) # (batch , sentence) -> (batch*sentence)

            predictions = predictions.to(device)
            labels = labels.to(device)
            sample_loss = final_model.model.compute_loss( predictions, labels, ~labels.eq(-1) )

            sample_loss.backward()
            optimizer.step()

            losses.append(sample_loss.item())
            

        mean_loss = sum(losses) / len(losses)
        train_history.append(mean_loss)
        
        if verbose or epoch == epochs - 1:
            print(f'  Epoch {epoch:3d} => avg_loss: {mean_loss:0.6f}')
        
        if valid_data_output is not None and valid_dataloader is not None:
            f1_s = evaluate_f1(final_model, valid_data_input, valid_data_output, device)
            valid_loss = evaluate_loss(final_model, valid_dataloader, device)
            valid_loss_history.append(valid_loss)
            valid_f1_history.append(f1_s)
            if verbose:
                print(f'    Validation loss => {valid_loss:0.6f} f1-score => {f1_s:0.6f}')

            # control stop:
            if use_stop and (len(valid_loss_history) > 1):
                if valid_loss_history[-1] - min(valid_loss_history) > threshold:
                    if warning_stop == True:
                        print(f'----- Forcing break -----')
                        break
                    else:
                        print(f'----- Warning stop activated! -----')
                        warning_stop = True
                elif warning_stop == True:
                    if not strong_stop:
                        print(f'----- Warning stop deactivated... -----')
                        warning_stop = False
                
        
    return {'train_history':train_history, 'valid_loss_history':valid_loss_history, 'valid_f1_history':valid_f1_history}

def evaluate_f1(final_model, valid_data_input, valid_data_label, device):
    final_model.model.eval()
    final_model.model.to(device)
    with torch.no_grad():
        data_eval_predict = final_model.predict(valid_data_input)
    f1_s = f1_score(valid_data_label, data_eval_predict, average="macro")
    return f1_s

def evaluate_loss(final_model, valid_dataset, device):
    valid_loss = 0.0
    final_model.model.eval()
    final_model.model.to(device)
    with torch.no_grad():
        for sample in valid_dataset:
            inputs = sample['inputs'].to(device)
            chars = sample['chars'].to(device)
            labels = sample['outputs']

            predictions = final_model.model.compute_outputs(inputs, chars)
            
            if final_model.model.crf is not None:
                predictions = torch.tensor(predictions)
            else:
                predictions = predictions.reshape(-1, predictions.shape[-1]).to(device) # (batch , sentence , n_labels) -> (batch*sentence , n_labels)
                labels = labels.view(-1) # (batch , sentence) -> (batch*sentence)

            predictions = predictions.to(device)
            labels = labels.to(device)

            sample_loss = final_model.model.compute_loss( predictions, labels, ~labels.eq(-1) )
            valid_loss += sample_loss.tolist()

    return valid_loss / len(valid_dataset)