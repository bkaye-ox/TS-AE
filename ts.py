import torch
import tsnets
import torch.utils.data as tdata

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_ds = tsnets.TSDataSet(
    src='ts.feather', N_pred=1, nx=4, N_y=3, N_u=3, test=False)
test_ds = tsnets.TSDataSet(src='ts.feather', N_pred=1,
                           nx=4, N_y=3, N_u=3, test=True)

loader = tdata.DataLoader(dataset=train_ds, batch_size=128, shuffle=True)
test_dataloader = tdata.DataLoader(test_ds, batch_size=128)

params = train_ds.get_params()
model = tsnets.TSNet(**params)

phase1_loss = tsnets.TSLoss(10, 0.3, 0)
phase2_loss = tsnets.TSLoss(0, 10, 1)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)


def train(dataloader, model, loss_fn, optimizer, N_epochs, losses):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(N_epochs):
        for batch, X in enumerate(dataloader):
            X = [x.to(device) for x in X]

            # Compute prediction error
            # pred = model(*X)
            # out = model.get_x_kp1(*X[:2])
            # loss = loss_fn(pred, out)

            loss = model.train_loss(*X)

            

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if batch % 100 == 0:
                # loss, current = loss.item(), batch * len(X)
                current = batch * len(X)

                print(f"loss: {losses[-1]:>7f}  [{current:>5d}/{size:>5d}]")


losses = []

train(dataloader=loader, model=model, loss_fn=phase1_loss,
      optimizer=optimizer, N_epochs=300, losses=losses)
train(dataloader=loader, model=model, loss_fn=phase2_loss,
      optimizer=optimizer, N_epochs=300, losses=losses)

model.eval()

res = []
for I in test_dataloader:
    with torch.no_grad():
        preds = model.predict(*I[:2])
        _, _, ground_truth = model.get_x_kp1(*I[:2])

        res.append([preds.numpy().reshape(-1,),
                   ground_truth.numpy().reshape(-1,)])

        import pandas as pd
        df = pd.DataFrame()
        df['ypred'], df['y'] = res
        df.to_feather('out.feather')

# for outs in zip(*res):


pass
