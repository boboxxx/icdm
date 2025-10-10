import torch
import torch.nn.functional as F
import math


class NoiseScheduleVP:
    def __init__(
        self,
        schedule="discrete",
        betas=None,
        alphas_cumprod=None,
        continuous_beta_0=0.1,
        continuous_beta_1=20.0,
        dtype=torch.float32,
    ):
        """Create a wrapper class for the forward SDE (VP type).
        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***
        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:
            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)
        Moreover, as lambda(t) is an invertible function, we also support its inverse function:
            t = self.inverse_lambda(lambda_t)
        ===============================================================
        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).
        1. For discrete-time DPMs:
            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.
            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)
            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.
            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).
        2. For continuous-time DPMs:
            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:
            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.
        ===============================================================
        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).

        ===============================================================
        Example:
        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)
        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)
        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)
        """

        if schedule not in ["discrete", "linear", "cosine"]:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(
                    schedule
                )
            )

        self.schedule = schedule
        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.0
            self.t_array = (
                torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            )
            self.log_alpha_array = log_alphas.reshape(
                (
                    1,
                    -1,
                )
            ).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.0
            self.cosine_t_max = (
                math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi)
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            self.cosine_log_alpha_0 = math.log(
                math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0)
            )
            self.schedule = schedule
            if schedule == "cosine":
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.0

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == "discrete":
            return interpolate_fn(
                t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)
            ).reshape((-1))
        elif self.schedule == "linear":
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == "cosine":
            log_alpha_fn = lambda s: torch.log(
                torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0)
            )
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == "linear":
            tmp = (
                2.0
                * (self.beta_1 - self.beta_0)
                * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            )
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == "discrete":
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2.0 * lamb)
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                torch.flip(self.t_array.to(lamb.device), [1]),
            )
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            t_fn = (
                lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0))
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            t = t_fn(log_alpha)
            return t


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.0,
    classifier_fn=None,
    classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model."""

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == "discrete":
            return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(
                t_continuous
            ), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(
                t_continuous
            ), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * sigma_t * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1.0 or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class UniPC:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="data_prediction",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.0,
        dynamic_thresholding_ratio=0.995,
        variant="bh1",
    ):
        """Construct a UniPC.

        We support both data_prediction and noise_prediction.
        """
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["data_prediction", "noise_prediction"]

        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn

        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

        self.variant = variant
        self.predict_x0 = algorithm_type == "data_prediction"

    def dynamic_thresholding_fn(self, x0, t=None):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(
            torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims
        )
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(
            t
        )
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling."""
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(
                device
            )
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = (
                torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1)
                .pow(t_order)
                .to(device)
            )
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(
                    skip_type
                )
            )

    def get_orders_and_timesteps_for_singlestep_solver(
        self, steps, order, skip_type, t_T, t_0, device
    ):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (
                    K - 2
                ) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (
                    K - 1
                ) + [1]
            else:
                orders = [3,] * (
                    K - 1
                ) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [
                    2,
                ] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (
                    K - 1
                ) + [1]
        elif order == 1:
            K = steps
            orders = [
                1,
            ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == "logSNR":
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[
                torch.cumsum(
                    torch.tensor(
                        [
                            0,
                        ]
                        + orders
                    ),
                    0,
                ).to(device)
            ]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def multistep_uni_pc_update(
        self, x, model_prev_list, t_prev_list, t, order, A=None, Ap=None, deg_feature=None, **kwargs
    ):
        if len(t.shape) == 0:
            t = t.view(-1)
        if "bh" in self.variant:
            return self.multistep_uni_pc_bh_update(
                x,
                model_prev_list,
                t_prev_list,
                t,
                order,
                A=A,
                Ap=Ap,
                deg_feature=deg_feature,
                **kwargs,
            )
        else:
            assert self.variant == "vary_coeff"
            return self.multistep_uni_pc_vary_update(
                x, model_prev_list, t_prev_list, t, order, **kwargs
            )

    def multistep_uni_pc_vary_update(
        self, x, model_prev_list, t_prev_list, t, order, use_corrector=True
    ):
        print(f"using unified predictor-corrector with order {order} (solver type: vary coeff)")
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=x.device)

        K = len(rks)
        # build C matrix
        C = []

        col = torch.ones_like(rks)
        for k in range(1, K + 1):
            C.append(col)
            col = col * rks / (k + 1)
        C = torch.stack(C, dim=1)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            C_inv_p = torch.linalg.inv(C[:-1, :-1])
            A_p = C_inv_p

        if use_corrector:
            print("using corrector")
            C_inv = torch.linalg.inv(C)
            A_c = C_inv

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_ks = []
        factorial_k = 1
        h_phi_k = h_phi_1
        for k in range(1, K + 2):
            h_phi_ks.append(h_phi_k)
            h_phi_k = h_phi_k / hh - 1 / factorial_k
            factorial_k *= k + 1

        model_t = None
        if self.predict_x0:
            x_t_ = sigma_t / sigma_prev_0 * x - alpha_t * h_phi_1 * model_prev_0
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * torch.einsum(
                        "bkchw,k->bchw", D1s, A_p[k]
                    )
            # now corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = model_t - model_prev_0
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * torch.einsum(
                        "bkchw,k->bchw", D1s, A_c[k][:-1]
                    )
                x_t = x_t - alpha_t * h_phi_ks[K] * (D1_t * A_c[k][-1])
        else:
            log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(
                t_prev_0
            ), ns.marginal_log_mean_coeff(t)
            x_t_ = (torch.exp(log_alpha_t - log_alpha_prev_0)) * x - (
                sigma_t * h_phi_1
            ) * model_prev_0
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - sigma_t * h_phi_ks[k + 1] * torch.einsum(
                        "bkchw,k->bchw", D1s, A_p[k]
                    )
            # now corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = model_t - model_prev_0
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    x_t = x_t - sigma_t * h_phi_ks[k + 1] * torch.einsum(
                        "bkchw,k->bchw", D1s, A_c[k][:-1]
                    )
                x_t = x_t - sigma_t * h_phi_ks[K] * (D1_t * A_c[k][-1])
        return x_t, model_t

    def multistep_uni_pc_bh_update(
        self,
        x,
        model_prev_list,
        t_prev_list,
        t,
        order,
        x_t=None,
        use_corrector=True,
        A=None,
        Ap=None,
        deg_feature=None,
    ):
        # print(f'using unified predictor-corrector with order {order} (solver type: B(h))')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(
            t_prev_0
        ), ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)
        alpha_prev_0 = torch.exp(log_alpha_prev_0)
        post_mean_coeff1 = torch.square(sigma_prev_0) * torch.sqrt(alpha_t) / (1 - alpha_prev_0)
        post_mean_coeff2 = (1 - alpha_t) * torch.sqrt(alpha_t / alpha_prev_0) / (1 - alpha_prev_0)
        post_sqrt_variance = torch.sqrt(
            torch.square(sigma_prev_0) * (1 - alpha_t) / (1 - alpha_prev_0)
        )
        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi = torch.exp(hh)  # h_phi=e^h
        # print(h_phi-h_phi_1)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == "bh1":
            B_h = hh
        elif self.variant == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.cat(b)

        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if use_corrector:
            # print('using corrector')
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_t = None
        if self.predict_x0:
            x_t_ = sigma_t / sigma_prev_0 * x - alpha_t * h_phi_1 * model_prev_0

            if x_t is None:

                if use_predictor:
                    pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)
                else:
                    pred_res = 0
                x_t = x_t_ - alpha_t * B_h * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    corr_res = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = model_t - model_prev_0
                x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            # print('\nx:',x.shape)
            # print('model_prev_0',model_prev_0.shape)
            x_t_ = (
                torch.exp(log_alpha_t - log_alpha_prev_0) * x
                - sigma_t * h_phi * model_prev_0  # h_phi_1 --> h_phi
            )  ###### 这个是对当前xt最直接的估计

            # print(1)
            # noise=torch.randn_like(x_t_)
            # x_t = post_mean_coeff1*x_t_+post_mean_coeff2*x+post_sqrt_variance*noise
            residule = sigma_t * model_prev_0

            x_t_ = x_t_ + residule
            # x_t_ = Ap(deg_feature) + x_t_ - Ap(A(x_t_))

            if x_t is None:  # 这里每一次采样都是True

                if use_predictor:
                    pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)
                else:
                    pred_res = 0
                x_t = x_t_ - sigma_t * B_h * pred_res
                # x_t = x_t + sigma_t * model_prev_0

                # deg_feature = alpha_t * deg_feature + sigma_t * noise

                # x_t=x_temp+temp- sigma_t * B_h * pred_res
                # print(x_t-x_temp)
            ## 在这里换和在上面换是一样的，都是对预测xt的值域空间赋值，然后用corrector进行矫正
            ## 这并不能保证初始的退化是一致的，因为有corrector进行矫正，只能保证接近
            # predict : 用前order-1阶做估计当前xt
            # corrector : 预测出当前x_t后，在用当前的xt做一次矫正
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:  # always True
                    corr_res = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = model_t - model_prev_0
                x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)

                x_t = x_t - sigma_t * B_h * (rhos_c[-1] * D1_t)
                if A is not None:
                    x_t = Ap * deg_feature + x_t - Ap * A * (x_t)
        return x_t, model_t

    def sample(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        lower_order_final=True,
        denoise_to_zero=False,
        atol=0.0078,
        rtol=0.05,
        return_intermediate=False,
        A=None,
        Ap=None,
        deg_feature=None,
    ):
        """
        Compute the sample at time `t_end` by UniPC, given the initial `x` at time `t_start`.
        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == "multistep":
                assert steps >= order
                ## 获得所有采样点的time
                timesteps = self.get_time_steps(
                    skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device
                )
                # print(timesteps.shape)
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]

                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]  # len = 1, model_output_x
                # print(model_prev_list[0].shape)
                # 这两个if都是False
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                    # print(1)
                if return_intermediate:
                    intermediates.append(x)
                    # print(2)
                # Init the first `order` values by lower order multistep UniPC.
                for step in range(1, order):
                    t = timesteps[step]
                    x, model_x = self.multistep_uni_pc_update(
                        x,
                        model_prev_list,
                        t_prev_list,
                        t,
                        step,
                        use_corrector=True,
                        A=A,
                        Ap=Ap,
                        deg_feature=deg_feature,
                    )
                    if model_x is None:
                        # print('enter')
                        model_x = self.model_fn(x, t)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(model_x)

                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    if lower_order_final:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    if step == steps:
                        # print('do not run corrector at the last step')
                        use_corrector = False
                    else:
                        use_corrector = True
                    x, model_x = self.multistep_uni_pc_update(
                        x,
                        model_prev_list,
                        t_prev_list,
                        t,
                        step_order,
                        use_corrector=use_corrector,
                        A=A,
                        Ap=Ap,
                        deg_feature=deg_feature,
                    )
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        if model_x is None:
                            model_x = self.model_fn(x, t)
                        model_prev_list[-1] = model_x
            else:
                raise ValueError("Got wrong method {}".format(method))

            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x


#############################################################
# other utility functions
#############################################################


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(
        2
    )
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]


class Joint_UniPC:
    def __init__(
        self,
        model_fn_x,
        model_fn_n,
        noise_schedule,
        algorithm_type="data_prediction",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.0,
        dynamic_thresholding_ratio=0.995,
        variant="bh1",
    ):
        """Construct a UniPC.

        We support both data_prediction and noise_prediction.
        """
        self.model_x = lambda x, t: model_fn_x(x, t.expand((x.shape[0])))
        self.model_n = lambda x, t: model_fn_n(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["data_prediction", "noise_prediction"]

        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

        self.variant = variant
        self.predict_x0 = algorithm_type == "data_prediction"

    def dynamic_thresholding_fn(self, x0, t=None):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(
            torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims
        )
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(
            t
        )
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return x0

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling."""
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(
                device
            )
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = (
                torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1)
                .pow(t_order)
                .to(device)
            )
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(
                    skip_type
                )
            )

    def get_orders_and_timesteps_for_singlestep_solver(
        self, steps, order, skip_type, t_T, t_0, device
    ):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (
                    K - 2
                ) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (
                    K - 1
                ) + [1]
            else:
                orders = [3,] * (
                    K - 1
                ) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [
                    2,
                ] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (
                    K - 1
                ) + [1]
        elif order == 1:
            K = steps
            orders = [
                1,
            ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == "logSNR":
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[
                torch.cumsum(
                    torch.tensor(
                        [
                            0,
                        ]
                        + orders
                    ),
                    0,
                ).to(device)
            ]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def multistep_uni_pc_update(
        self,
        x,
        model_prev_list_x,
        n,
        model_prev_list_n,
        t_prev_list,
        t,
        order,
        Lambda=1,
        Beta=1,
        SNR=None,
        SINR=None,
        deg_feature=None,
        h=None,
        **kwargs,
    ):
        if len(t.shape) == 0:
            t = t.view(-1)

        return self.multistep_uni_pc_bh_update(
            x,
            model_prev_list_x,
            n,
            model_prev_list_n,
            t_prev_list,
            t,
            order,
            Lambda=Lambda,
            Beta=Beta,
            SNR=SNR,
            SINR=SINR,
            deg_feature=deg_feature,
            h_channel=h,
            **kwargs,
        )

    def multistep_uni_pc_bh_update(
        self,
        x,
        model_prev_list_x,
        n,
        model_prev_list_n,
        t_prev_list,
        t,
        order,
        Lambda=1,
        Beta=1,
        x_t=None,
        use_corrector=True,
        SNR=None,
        SINR=None,
        deg_feature=None,
        h_channel=None,
    ):
        # print(f'using unified predictor-corrector with order {order} (solver type: B(h))')
        ns = self.noise_schedule
        assert order <= len(model_prev_list_x)
        assert len(model_prev_list_x) == len(model_prev_list_n)
        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0_x = model_prev_list_x[-1]
        model_prev_0_n = model_prev_list_n[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(
            t_prev_0
        ), ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)
        alpha_prev_0 = torch.exp(log_alpha_prev_0)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s_x = []
        D1s_n = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i_x = model_prev_list_x[-(i + 1)]
            model_prev_i_n = model_prev_list_n[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s_x.append((model_prev_i_x - model_prev_0_x) / rk)
            D1s_n.append((model_prev_i_n - model_prev_0_n) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi = torch.exp(hh)
        # print(h_phi-h_phi_1)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == "bh1":
            B_h = hh
        elif self.variant == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.cat(b)

        # now predictor
        use_predictor = len(D1s_x) > 0 and x_t is None
        if len(D1s_x) > 0:
            D1s_x = torch.stack(D1s_x, dim=1)  # (B, K)
            D1s_n = torch.stack(D1s_n, dim=1)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s_x = None
            D1s_n = None

        if use_corrector:
            # print('using corrector')
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_tx = None
        model_tn = None

        # print('\nx:',x.shape)
        # print('model_prev_0',model_prev_0.shape)
        x_t_ = (
            torch.exp(log_alpha_t - log_alpha_prev_0) * x
            - sigma_t * h_phi * model_prev_0_x  # h_phi_1 --> h_phi
        )
        n_t_ = (
            torch.exp(log_alpha_t - log_alpha_prev_0) * n
            - sigma_t * h_phi * model_prev_0_n  # h_phi_1 --> h_phi
        )
        ##-------------------------------------- DPS
        # x0 = torch.exp(-log_alpha_prev_0) * (x - sigma_prev_0 * self.model_x(x, t_prev_0))
        # n0 = torch.exp(-log_alpha_prev_0) * (n - sigma_prev_0 * self.model_n(n, t_prev_0))

        # sigma_square = 1 / (10 ** (SNR / 10))
        # A = 1
        # Ap = math.sqrt((1 / (10 ** (SINR / 10)))) - (1 / (10 ** (SNR / 10)))

        # Ws = torch.abs(h_channel) ** 2 / (torch.abs(h_channel) ** 2 + sigma_square)

        # Ws = torch.cat((Ws, Ws), dim=2)

        # Wz = torch.conj(h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
        # WzH = (h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
        # L = n0.shape[2]
        # n0_complex = n0[:, :, : L // 2, :] + n0[:, :, L // 2 :, :] * 1j
        # n0_complex = n0_complex * Wz
        # x0_coeef = Ws * A * x0
        # n0_coeef = Ap * n0_complex
        # n0_coeef = torch.cat((torch.real(n0_coeef), torch.real(n0_coeef.imag)), dim=2)
        # gx = (
        #     Ws
        #     * A
        #     * (
        #         # (1 / alpha_prev_0)
        #         (deg_feature - x0_coeef - n0_coeef)
        #         / torch.sqrt(torch.mean(torch.abs(deg_feature - x0_coeef - n0_coeef) ** 2))
        #     )
        # )
        # mu = deg_feature - x0_coeef - n0_coeef
        # mu_complex = mu[:, :, : L // 2, :] + mu[:, :, L // 2 :, :] * 1j
        # mu_coeef = WzH * mu_complex
        # mu_coeef = torch.cat((torch.real(mu_coeef), torch.real(mu_coeef.imag)), dim=2)
        # gn = Ap * (
        #     # (1 / alpha_prev_0)
        #     mu_coeef
        #     / torch.sqrt(torch.mean(torch.abs(deg_feature - x0_coeef - n0_coeef) ** 2))
        # )

        ##------------------------------------------GDM
        # x0 = torch.exp(-log_alpha_prev_0) * (x - sigma_prev_0 * self.model_x(x, t_prev_0))
        # n0 = torch.exp(-log_alpha_prev_0) * (n - sigma_prev_0 * self.model_n(n, t_prev_0))

        # sigma_square = 1 / (10 ** (SNR / 10))
        # A = 1
        # Ap = math.sqrt((1 / (10 ** (SINR / 10)))) - (1 / (10 ** (SNR / 10)))
        # An = math.sqrt((1 / (10 ** (SINR / 10))))
        # Ws = torch.abs(h_channel) ** 2 / (torch.abs(h_channel) ** 2 + sigma_square)

        # Ws = torch.cat((Ws, Ws), dim=2)

        # Wz = torch.conj(h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
        # WzWzH = torch.abs(h_channel) ** 2 / (torch.abs(h_channel) ** 2 + sigma_square) ** 2
        # WzWzH = torch.cat((WzWzH, WzWzH), dim=2)
        # WzH = (h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
        # L = n0.shape[2]
        # n0_complex = n0[:, :, : L // 2, :] + n0[:, :, L // 2 :, :] * 1j
        # n0_complex = n0_complex * Wz
        # x0_coeef = Ws * A * x0
        # n0_coeef = Ap * n0_complex
        # n0_coeef = torch.cat((torch.real(n0_coeef), torch.real(n0_coeef.imag)), dim=2)
        # mu = deg_feature - x0_coeef - n0_coeef
        # mu_complex = mu[:, :, : L // 2, :] + mu[:, :, L // 2 :, :] * 1j
        # mu_coeef = WzH * mu_complex  # Wz or WzH
        # mu_coeef = torch.cat((torch.real(mu_coeef), torch.real(mu_coeef.imag)), dim=2)

        # rt_square = sigma_prev_0**2 / (sigma_prev_0**2 + 1)
        # coeef = A**2 * Ws * Ws + Ap**2 * WzWzH + (A**2 + An**2) / rt_square
        # gx = A * Ws * mu / coeef
        # gn = Ap * mu_coeef / coeef
        ##------------------------------------------end
        ##-------------------projection：\sigma_t:60 step without:60 step
        # sigma_square = 1 / (10 ** (SNR / 10))
        # A = 1
        # Ap = math.sqrt((1 / (10 ** (SINR / 10)))) - (1 / (10 ** (SNR / 10)))

        # sigma_square = 1 / (10 ** (SNR / 10))

        # Ws = torch.abs(h_channel) ** 2 / (torch.abs(h_channel) ** 2 + sigma_square)

        # Ws = torch.cat((Ws, Ws), dim=2)

        # Wz = torch.conj(h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
        # WzH = (h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
        # L = n.shape[2]
        # n_t_complex = n[:, :, : L // 2, :] + n[:, :, L // 2 :, :] * 1j
        # n_t_complex = n_t_complex * Wz
        # x_t_coeef = Ws * A * x
        # n_t_coeef = Ap * n_t_complex
        # n_t_coeef = torch.cat((torch.real(n_t_coeef), torch.real(n_t_coeef.imag)), dim=2)
        # noise = torch.randn_like(n_t_coeef)

        # deg_feature_t = alpha_prev_0 * deg_feature + sigma_prev_0 * noise
        # # deg_feature_t = deg_feature * alpha_t
        # mu = deg_feature_t - x_t_coeef - n_t_coeef
        # mu_complex = mu[:, :, : L // 2, :] + mu[:, :, L // 2 :, :] * 1j
        # mu_coeef = WzH * mu_complex
        # mu_coeef = torch.cat((torch.real(mu_coeef), torch.real(mu_coeef.imag)), dim=2)
        # gx = Ws * (deg_feature_t - x_t_coeef - n_t_coeef)
        # gn = Ap * mu_coeef
        # ----------------end

        x_t_ = x_t_ + sigma_t * model_prev_0_x
        n_t_ = n_t_ + sigma_t * model_prev_0_n

        # x_t_ = deg_feature / (A+Ap) + x_t_ - (A*x_t_+Ap*n_t_)/(A+Ap)
        # n_t_ = deg_feature / (A+Ap) + n_t_ - (A*x_t_+Ap*n_t_)/(A+Ap)
        # x_t_ = deg_feature / (Ap) + x_t_ - (A*x_t_+Ap*n_t_)/(Ap)
        # n_t_ = deg_feature / (1/Ap-1) + n_t_ - (A*x_t_+Ap*n_t_)/(1/Ap-1)
        ###### 这个是对当前xt最直接的估计
        # x_t_ = Ap(deg_feature) + x_t_ - Ap(A(x_t_))
        # noise=torch.randn_like(x_t_)
        # x_t = post_mean_coeff1*x_t_+post_mean_coeff2*x+post_sqrt_variance*noise
        # residule=sigma_t*model_prev_0

        # x_t_=x_t_+residule

        if x_t is None:  # 这里每一次采样都是True

            if use_predictor:

                pred_res_x = torch.einsum("k,bkchw->bchw", rhos_p, D1s_x)
                pred_res_n = torch.einsum("k,bkchw->bchw", rhos_p, D1s_n)
            else:
                pred_res_x = 0
                pred_res_n = 0
            x_t = x_t_ - sigma_t * B_h * pred_res_x
            n_t = n_t_ - sigma_t * B_h * pred_res_n
            # x_t = x_t + sigma_t * model_prev_0

            # deg_feature = alpha_t * deg_feature + sigma_t * noise

            # x_t=x_temp+temp- sigma_t * B_h * pred_res
            # print(x_t-x_temp)
        ## 在这里换和在上面换是一样的，都是对预测xt的值域空间赋值，然后用corrector进行矫正
        ## 这并不能保证初始的退化是一致的，因为有corrector进行矫正，只能保证接近
        # predict : 用前order-1阶做估计当前xt
        # corrector : 预测出当前x_t后，在用当前的xt做一次矫正
        if use_corrector:
            model_tx = self.model_x(x_t, t)
            model_tn = self.model_n(n_t, t)

            if D1s_x is not None:  # always True
                corr_res_x = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s_x)
                corr_res_n = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s_n)
            else:
                corr_res_x = 0
                corr_res_n = 0
            D1_tx = model_tx - model_prev_0_x
            D1_tn = model_tn - model_prev_0_n
            x_t = x_t_ - sigma_t * B_h * (corr_res_x + rhos_c[-1] * D1_tx)
            n_t = n_t_ - sigma_t * B_h * (corr_res_n + rhos_c[-1] * D1_tn)

            ## -----------------DPS(0.5,0.3) + GDM(1.5,1.5)+projection(1.1,0.5)
            # model_tx = model_tx - 0.5 * gx
            # model_tn = model_tn - 0.3 * gn
            # -----------------end
            ##-------------------projection：\sigma_t:60 step without:60 step
            # sigma_square = 1 / (10 ** (SNR / 10))
            # A = 1
            # Ap = math.sqrt((1 / (10 ** (SINR / 10)))) - (1 / (10 ** (SNR / 10)))

            # sigma_square = 1 / (10 ** (SNR / 10))

            # Ws = torch.abs(h_channel) ** 2 / (torch.abs(h_channel) ** 2 + sigma_square)

            # Ws = torch.cat((Ws, Ws), dim=2)

            # Wz = torch.conj(h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
            # WzH = (h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
            # L = n_t.shape[2]
            # n_t_complex = n[:, :, : L // 2, :] + n[:, :, L // 2 :, :] * 1j
            # n_t_complex = n_t_complex * Wz
            # x_t_coeef = Ws * A * x
            # n_t_coeef = Ap * n_t_complex
            # n_t_coeef = torch.cat((torch.real(n_t_coeef), torch.real(n_t_coeef.imag)), dim=2)
            # noise = torch.randn_like(n_t_coeef)

            # deg_feature_t = alpha_prev_0 * deg_feature + sigma_prev_0 * noise
            # # deg_feature_t = deg_feature * alpha_t
            # mu = deg_feature_t - x_t_coeef - n_t_coeef
            # mu_complex = mu[:, :, : L // 2, :] + mu[:, :, L // 2 :, :] * 1j
            # mu_coeef = WzH * mu_complex
            # mu_coeef = torch.cat((torch.real(mu_coeef), torch.real(mu_coeef.imag)), dim=2)
            # model_tx = model_tx - 0.7 * Ws * (deg_feature_t - x_t_coeef - n_t_coeef)
            # model_tn = model_tn - 0.5 * Ap * mu_coeef
            # --------------------end

            # -------------- normal
            # sigma_square = 1 / (10 ** (SNR / 10))
            # A = 1
            # Ap = math.sqrt((1 / (10 ** (SINR / 10)))) - (1 / (10 ** (SNR / 10)))

            # sigma_square = 1 / (10 ** (SNR / 10))

            # Wn_square_inv = (torch.abs(h_channel) ** 2 + sigma_square) ** 2 / torch.abs(
            #     h_channel
            # ) ** 2
            # Wn_square_inv = torch.cat((Wn_square_inv, Wn_square_inv), dim=2)
            # Ws = torch.abs(h_channel) ** 2 / (torch.abs(h_channel) ** 2 + sigma_square)

            # Ws = torch.cat((Ws, Ws), dim=2)

            # Wz = torch.conj(h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
            # WzH = (h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
            # L = n_t.shape[2]
            # n_t_complex = n_t[:, :, : L // 2, :] + n_t[:, :, L // 2 :, :] * 1j
            # n_t_complex = n_t_complex * Wz
            # x_t_coeef = Ws * A * x_t
            # n_t_coeef = Ap * n_t_complex
            # n_t_coeef = torch.cat((torch.real(n_t_coeef), torch.real(n_t_coeef.imag)), dim=2)
            # mu = deg_feature - x_t_coeef - n_t_coeef
            # mu_complex = mu[:, :, : L // 2, :] + mu[:, :, L // 2 :, :] * 1j
            # mu_coeef = WzH * mu_complex
            # mu_coeef = torch.cat((torch.real(mu_coeef), torch.real(mu_coeef.imag)), dim=2)
            # model_tx = model_tx - 1 * Ws * (deg_feature - x_t_coeef - n_t_coeef)
            # model_tn = model_tn - 0.3 * Ap * mu_coeef

            ###------------------- end
            # ------------------ previous for predict ------
            # zeta = 1 / (2 - alpha_prev_0**2)  # (sigma_t**2 + 1)
            # A = 1
            # Ap = math.sqrt((1 / (10 ** (SINR / 10)))) - (1 / (10 ** (SNR / 10)))
            # deg_feature_t = deg_feature / zeta

            # if h_channel[0, 0, 0, 0] == h_channel[0, 0, 0, 1] == 1:
            #     x_t_coeef = A * x
            #     n_t_coeef = Ap * n
            #     model_tx = model_tx - zeta**2 * Lambda * (deg_feature_t - x_t_coeef - n_t_coeef)
            #     model_tn = model_tn - Ap * zeta**2 * Beta * (
            #         deg_feature_t - x_t_coeef - n_t_coeef
            #     )
            # # print(1)
            # else:

            #     sigma_square = 1 / (10 ** (SNR / 10))

            #     Wn_square_inv = (torch.abs(h_channel) ** 2 + sigma_square) ** 2 / torch.abs(
            #         h_channel
            #     ) ** 2
            #     Wn_square_inv = torch.cat((Wn_square_inv, Wn_square_inv), dim=2)
            #     Ws = torch.abs(h_channel) ** 2 / (torch.abs(h_channel) ** 2 + sigma_square)

            #     Ws = torch.cat((Ws, Ws), dim=2)

            #     Wz = torch.conj(h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
            #     WzH = (h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
            #     L = n_t.shape[2]
            #     n_t_complex = n[:, :, : L // 2, :] + n[:, :, L // 2 :, :] * 1j
            #     n_t_complex = n_t_complex * Wz
            #     x_t_coeef = Ws * A * x
            #     n_t_coeef = Ap * n_t_complex
            #     n_t_coeef = torch.cat((torch.real(n_t_coeef), torch.real(n_t_coeef.imag)), dim=2)
            #     mu = deg_feature_t - x_t_coeef - n_t_coeef
            #     mu_complex = mu[:, :, : L // 2, :] + mu[:, :, L // 2 :, :] * 1j
            #     mu_coeef = WzH * mu_complex
            #     mu_coeef = torch.cat((torch.real(mu_coeef), torch.real(mu_coeef.imag)), dim=2)
            #     model_tx = model_tx - zeta**2 * Lambda * Ws * Wn_square_inv * (
            #         deg_feature_t - x_t_coeef - n_t_coeef
            #     )
            #     model_tn = model_tn - Ap * zeta**2 * Beta * Wn_square_inv * mu_coeef
            ## ------------------------------ proposed
            zeta = 1 / (2 - alpha_t**2)  # (sigma_t**2 + 1)
            A = 1
            Ap = math.sqrt((1 / (10 ** (SINR / 10)))) - (1 / (10 ** (SNR / 10)))
            deg_feature_t = deg_feature / zeta

            if h_channel[0, 0, 0, 0] == h_channel[0, 0, 0, 1] == 1:
                x_t_coeef = A * x_t
                n_t_coeef = Ap * n_t
                model_tx = model_tx - zeta**2 * Lambda * (deg_feature_t - x_t_coeef - n_t_coeef)
                model_tn = model_tn - Ap * zeta**2 * Beta * (
                    deg_feature_t - x_t_coeef - n_t_coeef
                )
            # print(1)
            else:

                sigma_square = 1 / (10 ** (SNR / 10))

                Wn_square_inv = (torch.abs(h_channel) ** 2 + sigma_square) ** 2 / torch.abs(
                    h_channel
                ) ** 2
                Wn_square_inv = torch.cat((Wn_square_inv, Wn_square_inv), dim=2)
                Ws = torch.abs(h_channel) ** 2 / (torch.abs(h_channel) ** 2 + sigma_square)

                Ws = torch.cat((Ws, Ws), dim=2)

                Wz = torch.conj(h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
                WzH = (h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
                L = n_t.shape[2]
                n_t_complex = n_t[:, :, : L // 2, :] + n_t[:, :, L // 2 :, :] * 1j
                n_t_complex = n_t_complex * Wz
                x_t_coeef = Ws * A * x_t
                n_t_coeef = Ap * n_t_complex
                n_t_coeef = torch.cat((torch.real(n_t_coeef), torch.real(n_t_coeef.imag)), dim=2)
                mu = deg_feature_t - x_t_coeef - n_t_coeef
                mu_complex = mu[:, :, : L // 2, :] + mu[:, :, L // 2 :, :] * 1j
                mu_coeef = WzH * mu_complex
                mu_coeef = torch.cat((torch.real(mu_coeef), torch.real(mu_coeef.imag)), dim=2)
                model_tx = model_tx - zeta**2 * Lambda * Ws * Wn_square_inv * (
                    deg_feature_t - x_t_coeef - n_t_coeef
                )
                model_tn = model_tn - Ap * zeta**2 * Beta * Wn_square_inv * mu_coeef
            # ------------------------------ end

        return x_t, model_tx, n_t, model_tn

    ## - 以后公开的时候统一写--
    def guide_guidance(
        self, x_t, n_t, deg_feature, t, zeta, Ws, Wz, WzH, SNR, SINR, h_channel, model_tx, model_tn
    ):
        sigma_square = 1 / (10 ** (SNR / 10))
        A = 1
        Ap = math.sqrt((1 / (10 ** (SINR / 10)))) - (1 / (10 ** (SNR / 10)))

        sigma_square = 1 / (10 ** (SNR / 10))

        Wn_square_inv = (torch.abs(h_channel) ** 2 + sigma_square) ** 2 / torch.abs(h_channel) ** 2
        Wn_square_inv = torch.cat((Wn_square_inv, Wn_square_inv), dim=2)
        Ws = torch.abs(h_channel) ** 2 / (torch.abs(h_channel) ** 2 + sigma_square)

        Ws = torch.cat((Ws, Ws), dim=2)

        Wz = torch.conj(h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
        WzH = (h_channel) / (torch.abs(h_channel) ** 2 + sigma_square)
        L = n_t.shape[2]
        n_t_complex = n_t[:, :, : L // 2, :] + n_t[:, :, L // 2 :, :] * 1j
        n_t_complex = n_t_complex * Wz
        x_t_coeef = Ws * A * x_t
        n_t_coeef = Ap * n_t_complex
        n_t_coeef = torch.cat((torch.real(n_t_coeef), torch.real(n_t_coeef.imag)), dim=2)
        mu = deg_feature - x_t_coeef - n_t_coeef
        mu_complex = mu[:, :, : L // 2, :] + mu[:, :, L // 2 :, :] * 1j
        mu_coeef = WzH * mu_complex
        mu_coeef = torch.cat((torch.real(mu_coeef), torch.real(mu_coeef.imag)), dim=2)
        model_tx = model_tx - 0.7 * Ws * (deg_feature - x_t_coeef - n_t_coeef)
        model_tn = model_tn - 0.3 * Ap * mu_coeef
        return model_tx, model_tn

    # pass

    def sample(
        self,
        x,
        n,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        lower_order_final=True,
        denoise_to_zero=False,
        atol=0.0078,
        rtol=0.05,
        return_intermediate=False,
        SNR=None,
        SINR=None,
        deg_feature=None,
        h=None,
        Lambda=1,
        Beta=1,
    ):
        # return_intermediate = True
        """
        Compute the sample at time `t_end` by UniPC, given the initial `x` at time `t_start`.
        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        intermediates_n = []
        with torch.no_grad():
            if method == "multistep":
                assert steps >= order
                ## 获得所有采样点的time
                timesteps = self.get_time_steps(
                    skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device
                )
                # print(timesteps.shape)
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]

                t_prev_list = [t]
                # deg_feature_init=deg_feature/sigma_t
                A = 1
                Ap = math.sqrt((1 / (10 ** (SINR / 10)))) - (1 / (10 ** (SNR / 10)))
                model_prev_init_x = self.model_x(x, t)
                model_prev_init_n = self.model_n(n, t)

                model_prev_list_x = [model_prev_init_x]  # len = 1, model_output_x
                model_prev_list_n = [model_prev_init_n]
                # print(model_prev_list[0].shape)

                # print(2)
                # Init the first `order` values by lower order multistep UniPC.
                # print(A,Ap)
                for step in range(1, order):
                    t = timesteps[step]
                    x, model_x, n, model_n = self.multistep_uni_pc_update(
                        x,
                        model_prev_list_x,
                        n,
                        model_prev_list_n,
                        t_prev_list,
                        t,
                        step,
                        Lambda=Lambda,
                        Beta=Beta,
                        use_corrector=True,
                        SNR=SNR,
                        SINR=SINR,
                        deg_feature=deg_feature,
                        h=h,
                    )
                    if model_x is None:
                        # print('enter')
                        model_x = self.model_fn(x, t)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                        intermediates_n.append(n)
                    t_prev_list.append(t)
                    model_prev_list_x.append(model_x)
                    model_prev_list_n.append(model_n)
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    if lower_order_final:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    if step == steps:
                        # print('do not run corrector at the last step')
                        use_corrector = False
                    else:
                        use_corrector = True
                    x, model_x, n, model_n = self.multistep_uni_pc_update(
                        x,
                        model_prev_list_x,
                        n,
                        model_prev_list_n,
                        t_prev_list,
                        t,
                        step_order,
                        Lambda=Lambda,
                        Beta=Beta,
                        use_corrector=use_corrector,
                        SNR=SNR,
                        SINR=SINR,
                        deg_feature=deg_feature,
                        h=h,
                    )
                    # if self.correcting_xt_fn is not None:
                    #     x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                        intermediates_n.append(n)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list_x[i] = model_prev_list_x[i + 1]
                        model_prev_list_n[i] = model_prev_list_n[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        # if model_x is None:
                        #     model_x = self.model_fn(x, t)
                        model_prev_list_x[-1] = model_x
                        model_prev_list_n[-1] = model_n
            else:
                raise ValueError("Got wrong method {}".format(method))

            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x, n
