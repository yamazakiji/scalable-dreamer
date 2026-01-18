"""Action encoder for dynamics model."""
import torch
from torch import nn, Tensor


class ActionEncoder(nn.Module):
    """
    Encode discrete actions into tokens via embedding lookup.

    Following Dreamer 4 (arXiv:2509.24527), actions are encoded into S_a tokens
    and summed with a learned embedding. For unlabeled videos, only the learned
    embedding is used.
    """

    def __init__(
        self,
        num_actions: int,
        embed_dim: int,
        num_tokens: int = 1,
    ):
        """
        Args:
            num_actions: Size of action vocabulary (number of discrete actions)
            embed_dim: Embedding dimension
            num_tokens: Number of tokens per action (S_a from paper)
        """
        super().__init__()

        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # Action embedding lookup
        self.action_embed = nn.Embedding(num_actions, embed_dim)

        # Learned embedding summed with action (used alone for unlabeled videos)
        self.learned_token = nn.Parameter(torch.randn(1, 1, num_tokens, embed_dim) * 0.02)

    def forward(self, actions: Tensor, has_actions: bool = True) -> Tensor:
        """
        Encode actions into token embeddings.

        Args:
            actions: (B, T) discrete action indices, or None for unlabeled
            has_actions: Whether actions are provided. If False, returns only
                         learned embeddings (for unlabeled video training)

        Returns:
            tokens: (B, T, num_tokens, embed_dim) action token embeddings
        """
        if not has_actions or actions is None:
            # Unlabeled video: return only learned embedding
            B, T = actions.shape if actions is not None else (1, 1)
            return self.learned_token.expand(B, T, -1, -1)

        B, T = actions.shape

        # Embed actions: (B, T) -> (B, T, embed_dim)
        action_emb = self.action_embed(actions)

        # Expand to num_tokens: (B, T, num_tokens, embed_dim)
        action_emb = action_emb.unsqueeze(2).expand(-1, -1, self.num_tokens, -1)

        # Sum with learned token
        tokens = action_emb + self.learned_token.expand(B, T, -1, -1)

        return tokens
