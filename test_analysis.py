from pytest import approx, mark
from scipy.stats import kurtosis as scipy_kurtosis
from torch import randn, rand
from torch.nn import Module, Conv2d
from torch.nn.functional import relu

from metrics import compute_avg_and_std, kurtosis, activation_hooks, flatten_dict


def test_compute_avg_and_std():
    input_1 = [3, 4]
    output_1 = compute_avg_and_std(input_1)
    assert output_1['avg'] == 3.5
    assert output_1['std'] == 0.5

    input_2 = [5, 12]
    output_2 = compute_avg_and_std(input_2)
    assert output_2['avg'] == 8.5
    assert output_2['std'] == 3.5


def test_kurtosis():
    input = randn(10000)
    normal_kurtosis = kurtosis(input.unsqueeze(0)).item()
    s_kurtosis = scipy_kurtosis(input.numpy())
    assert normal_kurtosis == approx(0., abs=0.1)
    assert normal_kurtosis == approx(s_kurtosis, abs=0.001)

def test_flatten_dict():
    input = {'a': {'b': 1, 'c': 2}}
    assert flatten_dict(input) == {'a/b': 1, 'a/c': 2}

    input = {'a': {'b': {'c': 1}}}
    assert flatten_dict(input) == {'a/b/c': 1}

class Net(Module):
    """Simple two layer conv net"""
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 8, kernel_size=(5, 5), stride=(2, 2))
        self.conv2 = Conv2d(8, 8, kernel_size=(3, 3), stride=(2, 2))

    def forward(self, x):
        y = relu(self.conv1(x))
        z = relu(self.conv2(y))
        return z


@mark.parametrize("acts_to_save", [None, "conv1,conv2"])
def test_register_activation_hooks(acts_to_save):
    mdl = Net()
    to_save = None if acts_to_save is None else acts_to_save.split(',')

    # register fwd hooks in specified layers
    saved_activations, deregister_hooks = activation_hooks(mdl, layers_to_save=to_save)

    # run 4 times, but unhook after 2nd iter
    # should only save 2
    num_saved = 2
    num_fwd = 4
    images = [randn(1, 3, 256, 256) for _ in range(num_fwd)]
    for i in range(num_fwd):
        mdl(images[i])
        if i == num_saved - 1:
            deregister_hooks()

    if to_save is not None:
        assert len(saved_activations) == len(to_save)
    for activation in saved_activations:
        assert len(saved_activations[activation]) == 3
        print(activation)
        assert saved_activations[activation][0] == num_saved
        assert saved_activations[activation][1]**2 >= 0.