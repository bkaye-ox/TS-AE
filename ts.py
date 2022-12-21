import torch
import tsnets
import torch.utils.data as tdata

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

dset = tsnets.TSDataSet(src='ts.feather', N_pred=1, nx=4, N_y=3, N_u=3)

loader = tdata.DataLoader(dataset=dset, batch_size=128, shuffle=True)

params = dset.get_params()
model = tsnets.TSNet(**params)

phase1_loss = tsnets.TSLoss(10, 0.3, 0)
phase2_loss = tsnets.TSLoss(0, 10, 1)

optimizer = torch.optim.Adam(model.parameters())

def train(dataloader, model, loss_fn, optimizer, N_epochs):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(N_epochs):
        for batch, X in enumerate(dataloader):
            X = [x.to(device) for x in X]

            # Compute prediction error
            pred = model(*X)
            out = model.get_x_kp1(*X[0:2])
            loss = loss_fn(pred, out)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


train(dataloader=loader,model=model,loss_fn=phase1_loss,optimizer=optimizer, N_epochs=50)

pass