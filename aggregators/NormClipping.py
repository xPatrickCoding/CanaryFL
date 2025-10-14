import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common import FitRes

class NormClippingFedAvg(FedAvg):
    def __init__(self, clip_norm=5.0, **kwargs):
        super().__init__(**kwargs)
        self.clip_norm = clip_norm

    def aggregate_fit(self, rnd, results, failures):
        clipped_results = []
        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)

            flat = np.concatenate([w.ravel() for w in weights])
            norm = np.linalg.norm(flat) #norm clipping

            scale = 1.0 / max(1.0, norm / self.clip_norm)
            weights = [w * scale for w in weights]

            clipped_params = ndarrays_to_parameters(weights)

            # Building new FitRes Object with new parameters
            new_fit_res = FitRes(
                parameters=clipped_params,
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
                status=fit_res.status,
            )
            clipped_results.append((client, new_fit_res))

        return super().aggregate_fit(rnd, clipped_results, failures)
