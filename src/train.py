
def training(model, criterion, optim, line_tensor, category_tensor):
    hidden = model.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
        
    loss = criterion(output, category_tensor)
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    return output, loss.item()   
