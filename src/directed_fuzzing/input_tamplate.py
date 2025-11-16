class InputTemplate:
    def __init__(self):
        # For each symbolic input: name -> {type, distribution, constraints}
        self.inputs = {}

    def register_unknown(self, name, distribution):
        self.inputs[name] = distribution

    def set_concrete(self, name, value):
        # Overrides the distribution
        self.inputs[name].set_fixed(value)

    def sample_all(self):
        sampled = {}
        for name, dist in self.inputs.items():
            sampled[name] = dist.sample()
        return sampled

    def update_posteriors(self, reward, sampled_inputs):
        for name, dist in self.inputs.items():
            dist.update(reward, sampled_inputs[name])

