"""Keep the requested CPU thread limit after Ultralytics initializes its backend."""
import torch


class ThreadLimit:
    def __init__(self,threads):
        if threads<1:raise ValueError('threads must be positive')
        self.threads=threads

    def __call__(self,predictor):
        if torch.get_num_threads()!=self.threads:torch.set_num_threads(self.threads)


def configure_model(model,threads):
    limiter=ThreadLimit(threads)
    model.add_callback('on_predict_start',limiter)
    return model
