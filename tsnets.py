import torch
import torch.nn as nn
import torch.utils.data.dataset as dset
import pandas as pd

import io


class LinNet(torch.nn.Module):
    def __init__(self, dim_in, dim_out, hidden_struct=[50, 50], activation=nn.SiLU) -> None:
        super().__init__()
        self.flatten = nn.Flatten()

        hidden_struct.insert(0, dim_in)
        args = [z for x, y in zip(hidden_struct[:-1], hidden_struct[1:])
                for z in (nn.Linear(in_features=x, out_features=y), activation())]

        self.mlp = nn.Sequential(
            *args,
        )
        self.lin = nn.Linear(
            in_features=hidden_struct[-1], out_features=dim_out)

    def forward(self, x):
        y = self.flatten(x)
        p = self.mlp(y)
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

        return (torch.bmm(A, torch.cat((x, torch.ones((x.shape[0], 1))), axis=1).reshape(-1, A.shape[2], 1)) + torch.bmm(B, u.reshape((-1, u.shape[1], 1)))).reshape(-1, A.shape[1])


class DecoderNet(torch.nn.Module):
    def __init__(self, nx, nu, ny) -> None:
        super().__init__()

        self.nx = nx
        self.C = BlockNet(nz=nx+nu, m=ny, n=nx+1)

    def forward(self, X):
        x = X[:, :self.nx]

        C = self.C(X)
        return torch.bmm(C, torch.cat((x, torch.ones((x.shape[0], 1))), axis=1).reshape(-1, C.shape[2], 1)).reshape(-1, C.shape[1])


class EncoderNet(torch.nn.Module):
    def __init__(self, nI, nx) -> None:
        super().__init__()
        self.encoder = LinNet(nI, nx)

    def forward(self, x):
        return self.encoder(x)


class SNet(nn.Module):
    def __init__(self, nx, ny, nu) -> None:
        super().__init__()
        self.lnet = LinNet(nx+nu+ny, nx)

    def forward(self, xyu_k):
        return self.lnet(xyu_k)


class TSNet(torch.nn.Module):
    def __init__(self, nx, nu, ny, N_y, N_u, loss_fn) -> None:
        super().__init__()

        print('warning/note: was using wrong returns for _encode_decode, review all usage of it')

        self.ny = ny
        self.nu = nu
        self.nx = nx
        self.N_y = N_y
        self.N_u = N_u

        nI = ny*N_y + nu*N_u  # length of Information vector

        self.enet = EncoderNet(nI=nI, nx=nx)
        self.dnet = DecoderNet(nx=nx, nu=nu, ny=ny)
        self.fnet = FNet(nx=nx, nu=nu)

        self.snet = SNet(nx=nx, nu=nu, ny=ny)

        self.loss = loss_fn

    def forward_1(self, I_km1, I_k, I_kp1):
        x_k_est = self._encode(I_km1=I_km1)

        u_k = self._u_k(I_k=I_k)
        u_kp1 = self._u_k(I_k=I_kp1)

        z_k = self._z_k(I_k=I_k, x_k=x_k_est)

        y_k = self._y_k(I_k)

        # TODO include if training neural observer
        # xyu_k = torch.cat((x_k_est, y_k, u_k), dim=1)
        # x_kp1_obs = self.s(xyu_k)
        # z_kp1_obs = torch.cat((x_kp1_obs, u_kp1), dim=1)
        # y_kp1_obs = self.dnet(z_kp1_obs)

        x_kp1_pred = self._transition(z_k=z_k)
        z_kp1_pred = self._z_k(I_k=I_kp1, x_k=x_kp1_pred)

        x_kp1_est = self._encode(I_km1=I_k)
        # z_kp1_est = torch.cat((x_kp1_est, u_kp1), dim=1)
        z_kp1_est = self._z_k(x_k=x_kp1_est, I_k=I_kp1)

        y_k_est = self._decode(z_k=z_k)
        y_kp1_pred = self._decode(z_k=z_kp1_pred)
        y_kp1_est = self._decode(z_k=z_kp1_est)

        # est <=> ground truth, pred <=> prediction
        return y_k_est, x_kp1_pred, y_kp1_pred, y_kp1_est

    def predict1(self, I_km1, u_k):
        '''return predicted y(k) given I(k-1), u(k)'''
        x_k = self._encode(I_km1=I_km1)
        z_k = TSNet._z_k_alt(u_k=u_k, x_k=x_k)
        y_k = self._decode(z_k=z_k)
        # x_kp1 = self._transition(z_km1=z_k)
        # self.decode/

        return y_k

    def get_x_kp1(self, I_k, I_kp1):
        with torch.no_grad():
            x_kp1 = self.enet(I_kp1)
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

    def forward_F(self, Is):

        F = len(Is) - 2
        # if F <= 0:
        #     raise Exception('insufficient Is supplied')

        xres_list = []
        yres_list = []
        xtruth_list = []
        ytruth_list = []
        for f in range(1, F+1):  # 1 to F-1
            # if len(Is[f-1:]) - (F-f+1) - 2 != 0:
            #     print('warning')
            x_fs_pred, y_fs_pred, x_truths, y_truths = self.predict_F_steps(
                Is[f-1:]
            )

            xres_list.append(x_fs_pred)
            yres_list.append(y_fs_pred)
            xtruth_list.append(x_truths)
            ytruth_list.append(y_truths)

        return xres_list, yres_list, xtruth_list, ytruth_list

    def predict_F_steps(self, Is):
        # TODO rewrite, wtf was I doing?

        F = len(Is[1:-1])

        I_km1, I_k = Is[0], Is[1]
        x_f = self._encode(I_km1=I_km1)  # x(0)
        z_f = self._z_k(x_k=x_f, I_k=I_k)  # z(0)
        # _f regers to k+f index

        batch_size = I_km1.shape[0]

        y_fs = torch.zeros((batch_size, self.ny, F))
        x_fs = torch.zeros((batch_size, self.nx, F))
        y_fs_truth = torch.zeros((batch_size, self.ny, F))
        x_fs_truth = torch.zeros((batch_size, self.nx, F))
        # y_fs[:, :, 0] = y_fs

        # assert  == F
        for f, (I_f, I_fp1) in enumerate(zip(Is[1:-1], Is[2:])):
            # for f, I_fp1 in enumerate(Is[2:]):

            x_fp1 = self._transition(z_km1=z_f)  # x(f+1)

            z_fp1_pred = self._z_k(x_k=x_fp1, I_k=I_fp1)
            y_fp1_pred = self._decode(z_k=z_fp1_pred)

            x_fp1_truth = self._encode(I_km1=I_f)
            y_fp1_truth = self._y_k(I_fp1)

            # raise Exception('using wrong returns for _encode_decode')

            x_fs[:, :, f] = x_fp1
            y_fs[:, :, f] = y_fp1_pred

            x_fs_truth[:, :, f] = x_fp1_truth
            y_fs_truth[:, :, f] = y_fp1_truth

            x_f = x_fp1
            z_f = z_fp1_pred

        # predicted state vector k+1 to k+F, and predicted output k+1 to k+F
        return x_fs, y_fs, x_fs_truth, y_fs_truth

    def _z_k(self, I_k, x_k):
        '''return z(k) = [x(k), u(k)]'''
        u_k = self._u_k(I_k)
        return torch.cat((x_k, u_k), dim=1)

    def _z_k_alt(x_k, u_k):
        '''return z(k) = [x(k), u(k)]'''
        return torch.cat((x_k, u_k), dim=1)

    def train_loss(self, Is, loss_params):
        if self.loss is None:
            raise Exception('set_loss_fn() must be called')

        ykhat, yk = self._encode_decode(I_km1=Is[0], I_k=Is[1])
        res = self.forward_F(Is)
        res = [torch.cat(t, dim=2) for t in res]
        xkpred, ykpred, xkpredtruth, ykpredtruth = res

        a, b, c = loss_params['a'], loss_params['b'], loss_params['c']

        loss = a*self.loss(yk, ykhat) + b*self.loss(xkpred, xkpredtruth) + \
            c*self.loss(ykpred, ykpredtruth)

        return loss

    def _encode_decode(self, I_km1, I_k):
        '''returns x(k), reconstructed y hat(k), and y(k)'''
        x_k = self._encode(I_km1=I_km1)
        z_k = self._z_k(I_k=I_k, x_k=x_k)
        y_k_hat = self._decode(z_k=z_k)

        return x_k, y_k_hat, self._y_k(I_k=I_k)

    def _encode(self, I_km1):
        '''returns x(k) = e(I(k-1))'''
        return self.enet(I_km1)

    def _decode(self, z_k):
        '''returns y(k) hat = C(z(k))x(k)'''
        return self.dnet(z_k)

    def _transition(self, z_km1):
        '''returns x(k) = f(z(k-1))'''
        return self.fnet(z_km1)

    def _u_seq(self, Is):
        batch_sz = Is[0].shape[0]

        F = len(Is)
        u_seq = torch.zeros(batch_sz, self.nu, F)
        for k, I_k in enumerate(Is):
            u_seq[:, :, k] = self._u_k(I_k)
        return u_seq

    def set_loss_fn(self, loss_fn):
        self.loss = loss_fn
        # def loss(self):

    def foward(self, x):
        print('warning: not implemented')
        return x

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self.eval()
        params = {
            'ny': self.ny,
            'nu': self.nu,
            'nx': self.nx,
            'N_y': self.N_y,
            'N_u': self.N_u,
        }

        destination['init_params'] = params

        return super()._save_to_state_dict(destination, prefix, keep_vars, )

    def load(state):
        model = TSNet(**state['init_params'], loss_fn=nn.L1Loss())
        del state['init_params']
        model.load_state_dict(state)
        return model


class TSDataSet(dset.Dataset):
    def __init__(self, src, N_pred, nx, N_y, N_u, test=False) -> None:
        self.N_pred = N_pred

        self.nx = nx
        self.N_y = N_y
        self.N_u = N_u

        if type(src) is str:
            df = pd.read_feather(src)
            self.setup(df, test)

        # self.loss = loss_fn

    def __len__(self):
        return self.len

    def __getitem__(self, torch_idx, return_shape=False):
        y, u = self.data

        idx = torch_idx + self.idx_offset

        I_list = [torch.tensor(([g for yk in y[idx+k:idx+k-self.N_y:-1] for g in yk] + (
            [g for uk in u[idx+k:idx+k-self.N_u:-1] for g in uk]))) for k in range(-1, self.N_pred + 1)]
        # returns { I_km1, I_k, I_kp1 ... }

        bad_shape = any(
            [bool(x.shape[-1] != (self.N_u*self.nu + self.N_y*self.ny)) for x in I_list])
        # if bad_shape:
        #     print('bad shape')

        if return_shape:
            return I_list, bad_shape
        else:
            return I_list

    def get_params(self,):
        return {
            'nx': self.nx,
            'ny': self.ny,
            'nu': self.nu,
            'N_y': self.N_y,
            'N_u': self.N_u,
        }

    def setup(self, df, test):

        split = int(len(df)*0.9)
        df = df[split:] if test else df[:split]

        y_cols = ['o2pp']
        u_cols = ['o2duty', 'apduty']

        y = list(zip(*(df[c] for c in y_cols)))
        u = list(zip(*(df[c] for c in u_cols)))

        self.ny = len(y_cols)
        self.nu = len(u_cols)

        self.data = (y, u)

        self.idx_offset = max(self.N_y, self.N_u) + 1
        self.len = len(df) - self.N_pred - self.idx_offset - 1

        good_indices = [k for k in range(
            len(df)) if not self.__getitem__(k, return_shape=True)[1]]

        print('index div::', self.len - len(good_indices))


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
