import torch
from balltrack.runtime import ThreadLimit


def test_requested_threads_are_restored_after_backend_reset():
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        ThreadLimit(1)(None)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original)
