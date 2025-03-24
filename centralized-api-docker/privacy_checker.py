def check_privacy(config):
    if config.dp_optimizer_selection == "old":
        from opacus.privacy_analysis import compute_rdp, get_privacy_spent
    else:
        from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))

    delta = 1.0 / config.n_client_data
    sample_rate = config.batch_size / config.n_client_data
    noise_multiplier = config.noise_multiplier

    steps = config.n_rounds * config.n_epochs / sample_rate
    rdps = compute_rdp(q=sample_rate, noise_multiplier=noise_multiplier, steps=steps, orders=orders)

    #rdps = compute_rdp(sample_rate, noise_multiplier, steps, orders)
    epsilon, alpha = get_privacy_spent(orders=orders, rdp=rdps, delta=delta)

    return epsilon, alpha
