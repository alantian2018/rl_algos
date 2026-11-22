import torch.nn as nn
import torch
import torch.nn.functional as F
from .rssm import RSSM
from .networks import RewardDecoder, ContinueDecoder, ObservationHandler, DreamerState

class WorldModel(nn.Module):
    def __init__(self, deterministic_dim, stochastic_dim, action_dim,  obs_dim, obs_channels, latent_obs_size, hidden_dim,
                 loss_weights=(1.0, 1.0, 10.0, 1.0), free_nats=0.0):
        super().__init__()

        hidden_state_size = deterministic_dim + stochastic_dim
        
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.loss_weights = loss_weights
        self.free_nats = free_nats
        self.rssm = RSSM(deterministic_dim, stochastic_dim, action_dim, hidden_dim, latent_obs_size)
        self.obs_handler = ObservationHandler(obs_dim, obs_channels, latent_obs_size, hidden_state_size, hidden_dim)
        self.reward_decoder = RewardDecoder(deterministic_dim, stochastic_dim, hidden_dim)
        self.continue_decoder = ContinueDecoder(deterministic_dim, stochastic_dim, hidden_dim)

    def observe_step(self, obs, previous_state : DreamerState, previous_action):
        encoded = self.obs_handler.encode(obs)
        return self.rssm.observe(previous_state, previous_action, encoded)

    def initial_state(self, batch_size: int) -> DreamerState:
        gru = self.rssm.sequential_model.gru
        reference = gru.weight_ih_l0
        return DreamerState(
            deterministic=reference.new_zeros(batch_size, self.deterministic_dim),
            stochastic=reference.new_zeros(batch_size, self.stochastic_dim),
        )

    def observe_sequence(self, obs, actions, terminated, truncated, next_obs, initial_state=None):
        batch_size, sequence_length = obs.shape[:2]
        if sequence_length < 1 or next_obs.shape != obs.shape:
            raise ValueError("obs and next_obs must have matching nonempty sequence shapes")
        if terminated.shape == (batch_size, sequence_length, 1):
            terminated = terminated[..., 0]
        if truncated.shape == (batch_size, sequence_length, 1):
            truncated = truncated[..., 0]
        if actions.shape[:2] != (batch_size, sequence_length):
            raise ValueError("actions must match observation batch/time dimensions")
        if terminated.shape != (batch_size, sequence_length) or truncated.shape != (batch_size, sequence_length):
            raise ValueError("terminated and truncated must have shape [B, T] or [B, T, 1]")
        episode_ends = terminated.to(device=actions.device, dtype=torch.bool) | truncated.to(device=actions.device, dtype=torch.bool)
        encoded = self.obs_handler.encode(next_obs.flatten(0, 1))
        encoded = encoded.reshape(batch_size, sequence_length, -1)
        state = self.initial_state(batch_size) if initial_state is None else initial_state
        zero_action = actions.new_zeros(batch_size, actions.shape[-1])
        state, _, _ = self.observe_step(obs[:, 0], state, zero_action)
        deterministic, stochastic = [], []
        prior_means, prior_stds = [], []
        posterior_means, posterior_stds = [], []

        for t in range(sequence_length):
            action = actions[:, t]
            if t > 0:
                reset = episode_ends[:, t - 1].unsqueeze(-1)
                if reset.any():
                    reset_state, _, _ = self.observe_step(
                        obs[:, t], self.initial_state(batch_size), zero_action
                    )
                    state = DreamerState(
                        torch.where(reset, reset_state.deterministic, state.deterministic),
                        torch.where(reset, reset_state.stochastic, state.stochastic),
                    )

            state, prior, posterior = self.rssm.observe(state, action, encoded[:, t])
            deterministic.append(state.deterministic)
            stochastic.append(state.stochastic)
            prior_means.append(prior.loc)
            prior_stds.append(prior.scale)
            posterior_means.append(posterior.loc)
            posterior_stds.append(posterior.scale)

        states = DreamerState(
            torch.stack(deterministic, dim=1),
            torch.stack(stochastic, dim=1),
        )
        priors = torch.distributions.Normal(
            torch.stack(prior_means, dim=1), torch.stack(prior_stds, dim=1)
        )
        posteriors = torch.distributions.Normal(
            torch.stack(posterior_means, dim=1), torch.stack(posterior_stds, dim=1)
        )
        return states, priors, posteriors

    def predict(self, states: DreamerState, decode_obs=True):
        observations = None
        if decode_obs:
            leading_shape = states.deterministic.shape[:-1]
            flat_states = DreamerState(
                states.deterministic.reshape(-1, self.deterministic_dim),
                states.stochastic.reshape(-1, self.stochastic_dim),
            )
            decoded = self.obs_handler.decode(flat_states)
            observations = decoded.reshape(*leading_shape, *decoded.shape[1:])

        return {
            "reconstructions": observations,
            "rewards": self.reward_decoder(states),
            "continuations": self.continue_decoder(states),
        }

    def step_env(self, state: DreamerState, actions):
        next_state, _ = self.rssm.imagine(state, actions)
        rewards = self.reward_decoder(next_state)
        continuations = self.continue_decoder(next_state).probs
        return next_state, rewards, continuations

    def loss(self, obs, actions, rewards, terminated, truncated, next_obs, initial_state=None, prediction_mask=None):
        states, priors, post = self.observe_sequence(obs, actions, terminated, truncated, next_obs, initial_state=initial_state)

        # dynamics loss between the prior and post
        kl = torch.distributions.kl_divergence(post, priors)  # [B, T, stochastic_dim]
        raw_kl = kl.sum(dim=-1).mean()
        L_kl = raw_kl.clamp_min(self.free_nats)

        predictions = self.predict(states, decode_obs=True)

        #reconstruction loss 
        reconstructions, pred_rewards, continuations = predictions['reconstructions'], predictions['rewards'], predictions['continuations']
        sq_error =  0.5 * ((reconstructions - next_obs) ** 2)
        image_loss = sq_error.sum((-3, -2, -1))
        L_reconstruction = image_loss.mean()


        targets = 1.0 - terminated.to(continuations.logits)
        if prediction_mask is None:
            prediction_mask = torch.ones_like(pred_rewards[..., 0], dtype=torch.bool)
        if prediction_mask.shape != pred_rewards.shape[:2]:
            raise ValueError("prediction_mask must have shape [B, T]")
        valid = prediction_mask.to(device=pred_rewards.device, dtype=torch.bool).unsqueeze(-1)
        count = valid.sum().clamp_min(1)
        L_rewards = 0.5 * (pred_rewards[valid] - rewards[valid]).square().sum() / count
        L_continuation = F.binary_cross_entropy_with_logits(
            continuations.logits[valid], targets[valid], reduction="sum"
        ) / count

        reconstruction_weight, reward_weight, continuation_weight, kl_weight = self.loss_weights
        total_loss = (
            reconstruction_weight * L_reconstruction + reward_weight * L_rewards
            + continuation_weight * L_continuation + kl_weight * L_kl
        )
        metrics = {
            "reconstruction_loss": L_reconstruction.detach(),
            "reward_loss": L_rewards.detach(),
            "continuation_loss": L_continuation.detach(),
            "kl_loss": L_kl.detach(),
            "raw_kl": raw_kl.detach(),
        }
        return total_loss, metrics, states
   



