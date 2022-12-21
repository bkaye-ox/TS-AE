import torch
import torch.nn as nn
import torch.utils.data.dataset as dset
import pandas as pd


class LinNet(torch.nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.renet = nn.Sequential(
            nn.Linear(in_features=dim_in, out_features=30),
            nn.SiLU(),
            nn.Linear(in_features=30, out_features=30),
            nn.SiLU(),
            nn.Linear(in_features=30, out_features=30),
            nn.SiLU()
        )
        self.lin = nn.Linear(in_features=30, out_features=dim_out)

    def forward(self, x):
        y = self.flatten(x)
        p = self.renet(y)
        return self.lin(p)


class BlockNet(nn.Module):
    def __init__(self, nz, m, n) -> None:
        super().__init__()

        self.m = m
        self.n = n
        self.lnet = LinNet(nz, m*n)

    def forward(self, x):
        y = self.lnet(x)
        return y.reshape((y.shape[0], self.m, self.n))


class FNet(torch.nn.Module):
    def __init__(self, nx, nu) -> None:
        super().__init__()
        self.nx = nx
        self.nu = nu

        self.A = BlockNet(nz=nx+nu, m=nx, n=nx+1)
        self.B = BlockNet(nz=nx+nu, m=nx, n=nu)

    def forward(self, X):
        x = X[:, :self.nx]
        u = X[:, self.nx:]

        A = self.A(X)
        B = self.B(X)

        return (torch.bmm(A, torch.cat((x, torch.ones((x.shape[0], 1))), axis=1).reshape(-1, A.shape[2], 1)) + torch.bmm(B, u.reshape((-1,u.shape[1],1)))).reshape(-1,A.shape[1])


class DNet(torch.nn.Module):
    def __init__(self, nx, nu, ny) -> None:
        super().__init__()

        self.nx = nx
        self.C = BlockNet(nz=nx+nu, m=ny, n=nx+1)

    def forward(self, X):
        x = X[:, :self.nx]

        C = self.C(X)
        return torch.bmm(C, torch.cat((x, torch.ones((x.shape[0], 1))), axis=1).reshape(-1,C.shape[2],1)).reshape(-1,C.shape[1])


class ENet(torch.nn.Module):
    def __init__(self, nI, nx) -> None:
        super().__init__()
        self.encoder = LinNet(nI, nx)

    def forward(self, x):
        return self.encoder(x)


class TSNet(torch.nn.Module):
    def __init__(self, nx, nu, ny, N_y, N_u, ) -> None:
        super().__init__()

        self.ny = ny
        self.nu = nu
        self.N_y = N_y

        nI = ny*N_y + nu*N_u  # length of Information vector

        self.e = ENet(nI=nI, nx=nx)
        self.d = DNet(nx=nx, nu=nu, ny=ny)
        self.f = FNet(nx=nx, nu=nu)

    def forward(self, I_k, I_kp1):
        x_k_est = self.e(I_k)

        u_k = self._u_k(I_k)
        z_k = torch.cat((x_k_est, u_k), dim=1)

        u_kp1 = self._u_k(I_kp1)
        x_kp1_pred = self.f(z_k)
        z_kp1_pred = torch.cat((x_kp1_pred, u_kp1), dim=1)

        x_kp1_est = self.e(I_kp1)
        z_kp1_est = torch.cat((x_kp1_est, u_kp1), dim=1)

        y_k_est = self.d(z_k)
        y_kp1_pred = self.d(z_kp1_pred)
        y_kp1_est = self.d(z_kp1_est)

        return y_k_est, x_kp1_pred, y_kp1_pred, y_kp1_est

    def get_x_kp1(self, I_k, I_kp1):
        with torch.no_grad():
            x_kp1 = self.e(I_kp1)
            y_k, y_kp1 = self._y_k(I_k), self._y_k(I_kp1)
            return y_k, x_kp1, y_kp1

    def _u_k(self, I_k):
        lb = self.N_y*self.ny
        ub = lb + self.nu

        return I_k[:, lb:ub]

    def _y_k(self, I_k):
        lb = 0
        ub = self.ny
        return I_k[:, lb:ub]


class TSDataSet(dset.Dataset):
    def __init__(self, src, N_pred, nx, N_y, N_u) -> None:
        self.N_pred = N_pred

        self.nx = nx
        self.N_y = N_y
        self.N_u = N_u

        if type(src) is str:
            df = pd.read_feather(src)
            self.setup(df)

    def __len__(self):
        return self.len

    def __getitem__(self, torch_idx):
        y, u = self.data

        idx = torch_idx + max(self.N_y, self.N_u)

        I_list = [torch.tensor(([g for yk in y[idx+k:idx+k-self.N_y:-1] for g in yk] + (
            [g for uk in u[idx+k:idx+k-self.N_u:-1] for g in uk]))) for k in range(self.N_pred + 1)]

        return I_list

    def get_params(self,):
        return {
            'nx': self.nx,
            'ny': self.ny,
            'nu': self.nu,
            'N_y': self.N_y,
            'N_u': self.N_u,
        }

    def setup(self, df):

        y_cols = ['o2pp']
        u_cols = ['o2duty', 'apduty']

        y = list(zip(*(df[c] for c in y_cols)))
        u = list(zip(*(df[c] for c in u_cols)))

        self.ny = len(y_cols)
        self.nu = len(u_cols)

        self.data = (y, u)

        self.idx_offset = max(self.N_y, self.N_u)
        self.len = len(df) - self.N_pred - self.idx_offset


class TSLoss(nn.Module):
    def __init__(self, alpha, beta, gamma) -> None:
        super().__init__()
        self.mae = nn.L1Loss()

        self.a = alpha
        self.b = beta
        self.g = gamma

    def forward(self, out, actual):

        y_k_est, x_kp1_pred, y_kp1_pred, y_kp1_est = out
        y_k, x_kp1, y_kp1 = actual

        loss = self.a*(self.mae(y_k, y_k_est) + self.mae(y_kp1, y_kp1_est)) + self.b*self.mae(x_kp1_pred, x_kp1) + self.g*self.mae(y_kp1,
                                                                                                                                   y_kp1_pred)

        return loss
