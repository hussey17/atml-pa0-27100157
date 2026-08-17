import torch

from vae_pa0.modeling import MNISTVAE, negative_elbo


def test_vae_shapes_and_probability_range() -> None:
    model = MNISTVAE(input_dim=784, hidden_dim=32, latent_dim=3)
    output = model(torch.rand(4, 1, 28, 28))
    assert output.logits.shape == (4, 784)
    assert output.reconstruction.shape == (4, 784)
    assert output.mu.shape == (4, 3)
    assert output.logvar.shape == (4, 3)
    assert torch.all((0 <= output.reconstruction) & (output.reconstruction <= 1))


def test_reparameterization_uses_log_variance() -> None:
    model = MNISTVAE(hidden_dim=8, latent_dim=2)
    mu = torch.tensor([[1.0, -1.0]])
    logvar = torch.log(torch.tensor([[4.0, 9.0]]))
    eps = torch.tensor([[0.5, -2.0]])
    sample = model.reparameterize(mu, logvar, eps)
    assert torch.allclose(sample, torch.tensor([[2.0, -7.0]]))


def test_zero_mean_unit_variance_has_zero_kl() -> None:
    logits = torch.zeros(2, 784)
    target = torch.zeros(2, 1, 28, 28)
    mu = torch.zeros(2, 5)
    logvar = torch.zeros(2, 5)
    loss = negative_elbo(logits, target, mu, logvar)
    assert torch.allclose(loss.kl, torch.tensor(0.0))
    assert torch.allclose(loss.total, loss.reconstruction)


def test_beta_scales_only_kl_term() -> None:
    logits = torch.zeros(1, 784)
    target = torch.zeros(1, 784)
    mu = torch.ones(1, 2)
    logvar = torch.zeros(1, 2)
    standard = negative_elbo(logits, target, mu, logvar, beta=1.0)
    doubled = negative_elbo(logits, target, mu, logvar, beta=2.0)
    assert torch.allclose(doubled.total - standard.total, standard.kl)
