from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim

import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

import sys
sys.path.append('..')

from VarDrop import efficient_sampler, CachedAdaptiveSampler


class Exp_Long_Term_Forecast_Efficient(Exp_Basic):

    def __init__(self, args):
        super(
            Exp_Long_Term_Forecast_Efficient,
            self
        ).__init__(args)

        # ----------------------------------------------------
        # VarDrop configuration
        # ----------------------------------------------------

        self.vardrop_version = int(
            getattr(args, 'vardrop_version', 0)
        )

        self.target_tokens = getattr(
            args,
            'target_tokens',
            None
        )

        self.sampler_refresh = int(
            getattr(args, 'sampler_refresh', 64)
        )

        self.vardrop_sampler = None

        # Version 3 uses cached adaptive sampling
        if self.vardrop_version == 3:

            if self.target_tokens is None:
                raise ValueError(
                    "target_tokens is required for "
                    "vardrop_version=3"
                )

            self.vardrop_sampler = CachedAdaptiveSampler(
                k=args.k,
                group_size=args.group_size,
                freq_list=range(1, 25),
                target_tokens=args.target_tokens,
                refresh_every=self.sampler_refresh
            )

    # ========================================================
    # Model
    # ========================================================

    def _build_model(self):

        model = self.model_dict[
            self.args.model
        ].Model(self.args).float()

        if (
            self.args.use_multi_gpu
            and self.args.use_gpu
        ):
            model = nn.DataParallel(
                model,
                device_ids=self.args.device_ids
            )

        return model

    # ========================================================
    # Data
    # ========================================================

    def _get_data(self, flag):

        data_set, data_loader = data_provider(
            self.args,
            flag
        )

        return data_set, data_loader

    # ========================================================
    # Optimizer
    # ========================================================

    def _select_optimizer(self):

        model_optim = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate
        )

        return model_optim

    # ========================================================
    # Criterion
    # ========================================================

    def _select_criterion(self):

        return nn.MSELoss()

    # ========================================================
    # VarDrop
    # ========================================================

    def _sample_variates(self, batch_x):

        indices = efficient_sampler(
            batch_x,
            k=self.args.k,
            group_size=self.args.group_size,
            freq_list=range(1, 25),
            version=self.vardrop_version,
            target_tokens=self.target_tokens,
            sampler=self.vardrop_sampler
        )

        return np.asarray(
            indices,
            dtype=np.int64
        )

    # ========================================================
    # Validation
    # ========================================================

    def vali(
        self,
        vali_data,
        vali_loader,
        criterion,
        partial_train=False
    ):

        total_loss = []

        self.model.eval()

        with torch.no_grad():

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark
            ) in enumerate(vali_loader):

                batch_x = (
                    batch_x.float()
                    .to(self.device)
                )

                batch_y = (
                    batch_y.float()
                )

                if (
                    'PEMS' in self.args.data
                    or
                    'Solar' in self.args.data
                ):

                    batch_x_mark = None
                    batch_y_mark = None

                else:

                    batch_x_mark = (
                        batch_x_mark.float()
                        .to(self.device)
                    )

                    batch_y_mark = (
                        batch_y_mark.float()
                        .to(self.device)
                    )

                # ------------------------------------------------
                # IMPORTANT:
                # Validation must use the same variate selection
                # logic as training.
                # ------------------------------------------------

                if partial_train:

                    sparse_indices = (
                        self._sample_variates(
                            batch_x
                        )
                    )

                    batch_x = (
                        batch_x[:, :, sparse_indices]
                    )

                    batch_y = (
                        batch_y[:, :, sparse_indices]
                    )

                # Decoder input
                dec_inp = torch.zeros_like(
                    batch_y[
                        :,
                        -self.args.pred_len:,
                        :
                    ]
                ).float()

                dec_inp = torch.cat(
                    [
                        batch_y[
                            :,
                            :self.args.label_len,
                            :
                        ],
                        dec_inp
                    ],
                    dim=1
                ).float().to(self.device)

                # Encoder-decoder
                if self.args.use_amp:

                    with torch.cuda.amp.autocast():

                        if self.args.output_attention:

                            outputs = self.model(
                                batch_x,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark
                            )[0]

                        else:

                            outputs = self.model(
                                batch_x,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark
                            )

                else:

                    if self.args.output_attention:

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )[0]

                    elif self.args.channel_independence:

                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape

                        if batch_x_mark is None:

                            outputs = self.model(
                                batch_x.permute(
                                    0, 2, 1
                                ).reshape(
                                    B * N,
                                    Tx,
                                    1
                                ),
                                batch_x_mark,
                                dec_inp.permute(
                                    0, 2, 1
                                ).reshape(
                                    B * N,
                                    Ty,
                                    1
                                ),
                                batch_y_mark
                            ).reshape(
                                B,
                                N,
                                -1
                            ).permute(
                                0, 2, 1
                            )

                        else:

                            outputs = self.model(
                                batch_x.permute(
                                    0, 2, 1
                                ).reshape(
                                    B * N,
                                    Tx,
                                    1
                                ),
                                batch_x_mark.repeat(
                                    N,
                                    1,
                                    1
                                ),
                                dec_inp.permute(
                                    0, 2, 1
                                ).reshape(
                                    B * N,
                                    Ty,
                                    1
                                ),
                                batch_y_mark.repeat(
                                    N,
                                    1,
                                    1
                                )
                            ).reshape(
                                B,
                                N,
                                -1
                            ).permute(
                                0,
                                2,
                                1
                            )

                    else:

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )

                f_dim = (
                    -1
                    if self.args.features == 'MS'
                    else 0
                )

                outputs = outputs[
                    :,
                    -self.args.pred_len:,
                    f_dim:
                ]

                batch_y = (
                    batch_y[
                        :,
                        -self.args.pred_len:,
                        f_dim:
                    ]
                    .to(self.device)
                )

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(
                    pred,
                    true
                )

                total_loss.append(
                    loss.item()
                )

        total_loss = np.average(
            total_loss
        )

        self.model.train()

        return total_loss

    # ========================================================
    # Train
    # ========================================================

    def train(self, setting):

        train_data, train_loader = (
            self._get_data('train')
        )

        vali_data, vali_loader = (
            self._get_data('val')
        )

        test_data, test_loader = (
            self._get_data('test')
        )

        path = os.path.join(
            self.args.checkpoints,
            setting
        )

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(
            train_loader
        )

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True
        )

        model_optim = (
            self._select_optimizer()
        )

        criterion = (
            self._select_criterion()
        )

        if self.args.use_amp:

            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(
            self.args.train_epochs
        ):

            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark
            ) in enumerate(train_loader):

                iter_count += 1

                model_optim.zero_grad()

                batch_x = (
                    batch_x.float()
                    .to(self.device)
                )

                batch_y = (
                    batch_y.float()
                    .to(self.device)
                )

                if (
                    'PEMS' in self.args.data
                    or
                    'Solar' in self.args.data
                ):

                    batch_x_mark = None
                    batch_y_mark = None

                else:

                    batch_x_mark = (
                        batch_x_mark.float()
                        .to(self.device)
                    )

                    batch_y_mark = (
                        batch_y_mark.float()
                        .to(self.device)
                    )

                # =================================================
                # VarDrop
                # =================================================

                sparse_indices = (
                    self._sample_variates(
                        batch_x
                    )
                )

                batch_x = (
                    batch_x[
                        :,
                        :,
                        sparse_indices
                    ]
                )

                batch_y = (
                    batch_y[
                        :,
                        :,
                        sparse_indices
                    ]
                )

                # =================================================
                # Decoder input
                # =================================================

                dec_inp = torch.zeros_like(
                    batch_y[
                        :,
                        -self.args.pred_len:,
                        :
                    ]
                ).float()

                dec_inp = torch.cat(
                    [
                        batch_y[
                            :,
                            :self.args.label_len,
                            :
                        ],
                        dec_inp
                    ],
                    dim=1
                ).float().to(self.device)

                # =================================================
                # Forward
                # =================================================

                if self.args.use_amp:

                    with torch.cuda.amp.autocast():

                        if self.args.output_attention:

                            outputs = self.model(
                                batch_x,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark
                            )[0]

                        else:

                            outputs = self.model(
                                batch_x,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark
                            )

                        f_dim = (
                            -1
                            if self.args.features == 'MS'
                            else 0
                        )

                        outputs = outputs[
                            :,
                            -self.args.pred_len:,
                            f_dim:
                        ]

                        batch_y_loss = batch_y[
                            :,
                            -self.args.pred_len:,
                            f_dim:
                        ]

                        loss = criterion(
                            outputs,
                            batch_y_loss
                        )

                        train_loss.append(
                            loss.item()
                        )

                else:

                    if self.args.output_attention:

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )[0]

                    elif self.args.channel_independence:

                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape

                        if batch_x_mark is None:

                            outputs = self.model(
                                batch_x.permute(
                                    0, 2, 1
                                ).reshape(
                                    B * N,
                                    Tx,
                                    1
                                ),
                                batch_x_mark,
                                dec_inp.permute(
                                    0, 2, 1
                                ).reshape(
                                    B * N,
                                    Ty,
                                    1
                                ),
                                batch_y_mark
                            ).reshape(
                                B,
                                N,
                                -1
                            ).permute(
                                0,
                                2,
                                1
                            )

                        else:

                            outputs = self.model(
                                batch_x.permute(
                                    0, 2, 1
                                ).reshape(
                                    B * N,
                                    Tx,
                                    1
                                ),
                                batch_x_mark.repeat(
                                    N,
                                    1,
                                    1
                                ),
                                dec_inp.permute(
                                    0,
                                    2,
                                    1
                                ).reshape(
                                    B * N,
                                    Ty,
                                    1
                                ),
                                batch_y_mark.repeat(
                                    N,
                                    1,
                                    1
                                )
                            ).reshape(
                                B,
                                N,
                                -1
                            ).permute(
                                0,
                                2,
                                1
                            )

                    else:

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )

                    f_dim = (
                        -1
                        if self.args.features == 'MS'
                        else 0
                    )

                    outputs = outputs[
                        :,
                        -self.args.pred_len:,
                        f_dim:
                    ]

                    batch_y_loss = batch_y[
                        :,
                        -self.args.pred_len:,
                        f_dim:
                    ]

                    loss = criterion(
                        outputs,
                        batch_y_loss
                    )

                    train_loss.append(
                        loss.item()
                    )

                # =================================================
                # Logging
                # =================================================

                if (i + 1) % 100 == 0:

                    print(
                        "\titers: {0}, epoch: {1} | "
                        "loss: {2:.7f}".format(
                            i + 1,
                            epoch + 1,
                            loss.item()
                        )
                    )

                    speed = (
                        time.time() - time_now
                    ) / iter_count

                    left_time = (
                        speed *
                        (
                            (
                                self.args.train_epochs
                                - epoch
                            ) *
                            train_steps
                            - i
                        )
                    )

                    print(
                        "\tspeed: {:.4f}s/iter; "
                        "left time: {:.4f}s".format(
                            speed,
                            left_time
                        )
                    )

                    iter_count = 0
                    time_now = time.time()

                # =================================================
                # Backward
                # =================================================

                if self.args.use_amp:

                    scaler.scale(
                        loss
                    ).backward()

                    scaler.step(
                        model_optim
                    )

                    scaler.update()

                else:

                    loss.backward()
                    model_optim.step()

            # =====================================================
            # Epoch result
            # =====================================================

            print(
                "Epoch: {} cost time: {}".format(
                    epoch + 1,
                    time.time() - epoch_time
                )
            )

            train_loss = np.average(
                train_loss
            )

            # Keep original behavior:
            # validation/test use ALL variables unless partial_train=True.
            vali_loss = self.vali(
                vali_data,
                vali_loader,
                criterion,
                partial_train=False
            )

            test_loss = self.vali(
                test_data,
                test_loader,
                criterion,
                partial_train=False
            )

            print(
                "Epoch: {0}, Steps: {1} | "
                "Train Loss: {2:.7f} "
                "Vali Loss: {3:.7f} "
                "Test Loss: {4:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    test_loss
                )
            )

            early_stopping(
                vali_loss,
                self.model,
                path
            )

            if early_stopping.early_stop:

                print("Early stopping")
                break

            adjust_learning_rate(
                model_optim,
                epoch + 1,
                self.args
            )

        # =========================================================
        # Load best model
        # =========================================================

        best_model_path = os.path.join(
            path,
            'checkpoint.pth'
        )

        self.model.load_state_dict(
            torch.load(
                best_model_path
            )
        )

        return self.model

    # ========================================================
    # Test
    # ========================================================

    def test(self, setting, test=0):

        test_data, test_loader = (
            self._get_data('test')
        )

        if test:

            print('loading model')

            self.model.load_state_dict(
                torch.load(
                    os.path.join(
                        './checkpoints/',
                        setting,
                        'checkpoint.pth'
                    )
                )
            )

        preds = []
        trues = []

        folder_path = (
            './test_results/'
            + setting
            + '/'
        )

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        with torch.no_grad():

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark
            ) in enumerate(test_loader):

                batch_x = (
                    batch_x.float()
                    .to(self.device)
                )

                batch_y = (
                    batch_y.float()
                    .to(self.device)
                )

                if (
                    'PEMS' in self.args.data
                    or
                    'Solar' in self.args.data
                ):

                    batch_x_mark = None
                    batch_y_mark = None

                else:

                    batch_x_mark = (
                        batch_x_mark.float()
                        .to(self.device)
                    )

                    batch_y_mark = (
                        batch_y_mark.float()
                        .to(self.device)
                    )

                # ------------------------------------------------
                # IMPORTANT:
                # Test trained model on ALL variables,
                # matching your current experimental protocol.
                # ------------------------------------------------

                dec_inp = torch.zeros_like(
                    batch_y[
                        :,
                        -self.args.pred_len:,
                        :
                    ]
                ).float()

                dec_inp = torch.cat(
                    [
                        batch_y[
                            :,
                            :self.args.label_len,
                            :
                        ],
                        dec_inp
                    ],
                    dim=1
                ).float().to(self.device)

                # Encoder-decoder
                if self.args.use_amp:

                    with torch.cuda.amp.autocast():

                        if self.args.output_attention:

                            outputs = self.model(
                                batch_x,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark
                            )[0]

                        else:

                            outputs = self.model(
                                batch_x,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark
                            )

                else:

                    if self.args.output_attention:

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )[0]

                    elif self.args.channel_independence:

                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape

                        if batch_x_mark is None:

                            outputs = self.model(
                                batch_x.permute(
                                    0, 2, 1
                                ).reshape(
                                    B * N,
                                    Tx,
                                    1
                                ),
                                batch_x_mark,
                                dec_inp.permute(
                                    0, 2, 1
                                ).reshape(
                                    B * N,
                                    Ty,
                                    1
                                ),
                                batch_y_mark
                            ).reshape(
                                B,
                                N,
                                -1
                            ).permute(
                                0,
                                2,
                                1
                            )

                        else:

                            outputs = self.model(
                                batch_x.permute(
                                    0, 2, 1
                                ).reshape(
                                    B * N,
                                    Tx,
                                    1
                                ),
                                batch_x_mark.repeat(
                                    N,
                                    1,
                                    1
                                ),
                                dec_inp.permute(
                                    0,
                                    2,
                                    1
                                ).reshape(
                                    B * N,
                                    Ty,
                                    1
                                ),
                                batch_y_mark.repeat(
                                    N,
                                    1,
                                    1
                                )
                            ).reshape(
                                B,
                                N,
                                -1
                            ).permute(
                                0,
                                2,
                                1
                            )

                    else:

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )

                f_dim = (
                    -1
                    if self.args.features == 'MS'
                    else 0
                )

                outputs = outputs[
                    :,
                    -self.args.pred_len:,
                    f_dim:
                ]

                batch_y = batch_y[
                    :,
                    -self.args.pred_len:,
                    f_dim:
                ]

                outputs = (
                    outputs.detach()
                    .cpu()
                    .numpy()
                )

                batch_y = (
                    batch_y.detach()
                    .cpu()
                    .numpy()
                )

                if (
                    test_data.scale
                    and self.args.inverse
                ):

                    shape = outputs.shape

                    outputs = (
                        test_data
                        .inverse_transform(
                            outputs.squeeze(0)
                        )
                        .reshape(shape)
                    )

                    batch_y = (
                        test_data
                        .inverse_transform(
                            batch_y.squeeze(0)
                        )
                        .reshape(shape)
                    )

                preds.append(outputs)
                trues.append(batch_y)

                if i % 20 == 0:

                    input_data = (
                        batch_x.detach()
                        .cpu()
                        .numpy()
                    )

                    if (
                        test_data.scale
                        and self.args.inverse
                    ):

                        shape = input_data.shape

                        input_data = (
                            test_data
                            .inverse_transform(
                                input_data.squeeze(0)
                            )
                            .reshape(shape)
                        )

                    gt = np.concatenate(
                        (
                            input_data[
                                0, :, -1
                            ],
                            batch_y[
                                0, :, -1
                            ]
                        ),
                        axis=0
                    )

                    pd = np.concatenate(
                        (
                            input_data[
                                0, :, -1
                            ],
                            outputs[
                                0, :, -1
                            ]
                        ),
                        axis=0
                    )

                    visual(
                        gt,
                        pd,
                        os.path.join(
                            folder_path,
                            str(i) + '.pdf'
                        )
                    )

        # =========================================================
        # Metrics
        # =========================================================

        preds = np.array(preds)
        trues = np.array(trues)

        print(
            'test shape:',
            preds.shape,
            trues.shape
        )

        preds = preds.reshape(
            -1,
            preds.shape[-2],
            preds.shape[-1]
        )

        trues = trues.reshape(
            -1,
            trues.shape[-2],
            trues.shape[-1]
        )

        print(
            'test shape:',
            preds.shape,
            trues.shape
        )

        folder_path = (
            './results/'
            + setting
            + '/'
        )

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(
            preds,
            trues
        )

        print(
            'mse:{}, mae:{}'.format(
                mse,
                mae
            )
        )

        with open(
            "result_long_term_forecast.txt",
            'a'
        ) as f:

            f.write(
                setting + "\n"
            )

            f.write(
                'mse:{}, mae:{}'.format(
                    mse,
                    mae
                )
            )

            f.write(
                '\n\n'
            )

        np.save(
            folder_path + 'metrics.npy',
            np.array([
                mae,
                mse,
                rmse,
                mape,
                mspe
            ])
        )

        np.save(
            folder_path + 'pred.npy',
            preds
        )

        np.save(
            folder_path + 'true.npy',
            trues
        )

        return

    # ========================================================
    # Predict
    # ========================================================

    def predict(self, setting, load=False):

        pred_data, pred_loader = (
            self._get_data('pred')
        )

        if load:

            path = os.path.join(
                self.args.checkpoints,
                setting
            )

            best_model_path = os.path.join(
                path,
                'checkpoint.pth'
            )

            self.model.load_state_dict(
                torch.load(
                    best_model_path
                )
            )

        preds = []

        self.model.eval()

        with torch.no_grad():

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark
            ) in enumerate(pred_loader):

                batch_x = (
                    batch_x.float()
                    .to(self.device)
                )

                batch_y = (
                    batch_y.float()
                )

                batch_x_mark = (
                    batch_x_mark.float()
                    .to(self.device)
                )

                batch_y_mark = (
                    batch_y_mark.float()
                    .to(self.device)
                )

                dec_inp = torch.zeros_like(
                    batch_y[
                        :,
                        -self.args.pred_len:,
                        :
                    ]
                ).float()

                dec_inp = torch.cat(
                    [
                        batch_y[
                            :,
                            :self.args.label_len,
                            :
                        ],
                        dec_inp
                    ],
                    dim=1
                ).float().to(self.device)

                if self.args.use_amp:

                    with torch.cuda.amp.autocast():

                        if self.args.output_attention:

                            outputs = self.model(
                                batch_x,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark
                            )[0]

                        else:

                            outputs = self.model(
                                batch_x,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark
                            )

                else:

                    if self.args.output_attention:

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )[0]

                    else:

                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark
                        )

                outputs = (
                    outputs.detach()
                    .cpu()
                    .numpy()
                )

                if (
                    pred_data.scale
                    and self.args.inverse
                ):

                    shape = outputs.shape

                    outputs = (
                        pred_data
                        .inverse_transform(
                            outputs.squeeze(0)
                        )
                        .reshape(shape)
                    )

                preds.append(outputs)

        preds = np.array(preds)

        preds = preds.reshape(
            -1,
            preds.shape[-2],
            preds.shape[-1]
        )

        folder_path = (
            './results/'
            + setting
            + '/'
        )

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(
            folder_path + 'real_prediction.npy',
            preds
        )

        return