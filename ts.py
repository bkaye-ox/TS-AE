import pandas as pd
import numpy as np
import torch
import tsnets
import torch.utils.data as tdata

import plotly.express as px

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
#
main_settings = dict(
    src='synth.feather',
    N_pred=2,
    nx=6,
    N_y=3,
    N_u=2,

)
train_ds = tsnets.TSDataSet(**main_settings, test=False)
test_ds = tsnets.TSDataSet(**main_settings, test=True)

loader = tdata.DataLoader(
    dataset=train_ds, batch_size=128, shuffle=True, drop_last=True)
test_dataloader = tdata.DataLoader(test_ds, batch_size=128, drop_last=True)

params = train_ds.get_params() | dict(loss_fn=torch.nn.L1Loss())
model = tsnets.TSNet(**params)

phase1_loss = dict(a=10.0, b=0.3, c=0)
phase2_loss = dict(a=1, b=10, c=0.1)


optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)


def train(dataloader, model,  loss_params, optimizer, N_epochs, losses):
    size = len(dataloader.dataset)

    # model.set_loss_fn(loss_fn,)

    model.train()
    for epoch in range(N_epochs):
        for batch, X in enumerate(dataloader):
            X = [x.to(device) for x in X]

            # Compute prediction error
            # pred = model(*X)
            # out = model.get_x_kp1(*X[:2])
            # loss = loss_fn(pred, out)

            loss = model.train_loss(X, loss_params)

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

num_epochs = 1

train(dataloader=loader, model=model, loss_params=phase1_loss,
      optimizer=optimizer, N_epochs=num_epochs, losses=losses)
train(dataloader=loader, model=model, loss_params=phase2_loss,
      optimizer=optimizer, N_epochs=num_epochs, losses=losses)

model.eval()

px.line(y=losses, log_y=True).show()
# input()
res = []
for I in test_dataloader:
    with torch.no_grad():
        I_km1 = I[0]
        u_k = model._u_k(I_k=I[1])
        preds = model.predict1(I_km1=I_km1, u_k=u_k)
        ground_truth = model._y_k(I_k=I[1])

        res.append([preds.numpy().reshape(-1,),
                   ground_truth.numpy().reshape(-1,)])


res = [np.concatenate(l) for l in list(zip(*res))]


df = pd.DataFrame()
df['ypred'], df['y'] = res
df.to_feather('out.feather')
# for outs in zip(*res):


pass
