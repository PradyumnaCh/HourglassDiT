from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.utils.backbone_utils import BackboneConfigMixin

logger = logging.get_logger(__name__)

# TODO: Setup our own configuration for HDiT like this!
NAT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shi-labs/nat-mini-in1k-224": "https://huggingface.co/shi-labs/nat-mini-in1k-224/resolve/main/config.json",
    # See all Nat models at https://huggingface.co/models?filter=nat
}


class HDiTConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HDiT Model`]. It is used to instantiate a Nat model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Nat

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:

        input_size (`int`, *optional*, defaults to 128):
        `int`: The size of the input image.

        patch_size (`int`, *optional*, defaults to 4):
            The size (resolution) of each patch. NOTE: Only patch size of 4 is supported at the moment.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embed_dim (`int`, *optional*, defaults to 64):
            Dimensionality of patch embedding.
        cond_dim (`int`, *optional*, defaults to 768):
            Dimensionality of conditioning embedding
        levels (`List[int]`, *optional*, defaults to `[1, 1]`):
            Number of levels in the Hourglass Structure
        depths (`List[int]`, *optional*, defaults to `[2,11]`):
            Number of layers in each level of the encoder.
        widths (`List[int]`, *optional*, defaults to `[384,764]`):
            Width of the transformer layer at each level
        num_heads (`List[int]`, *optional*, defaults to `[6,12]`):
            Number of attention heads in each layer of the Transformer
        attn_head_dim (`int`, *optional*, defaults to 64):
            The dimension of each attention head.
        kernel_size (`int`, *optional*, defaults to 7):
            Neighborhood Attention kernel size.
        mlp_depth (`int`, *optional*, defaults to 2):
            The depth of the mlp.
        mlp_dim (`int`, *optional*, defaults to 768):
            The dimension of the MLP
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not a learnable bias should be added to the queries, keys and values.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
    Example:

    ```python
    >>> from transformers import HDiTConfig, HDiTModel
    >>> # Initializing a HDiT configuration
    >>> configuration = HDiTConfig()

    >>> # Initializing a model (with random weights) from the default configuration
    >>> model = HDiT(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "hdit"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        input_size=128,
        patch_size=4,
        num_channels=3,
        num_classes=1000,
        cond_dim=768,
        levels=[1, 1],
        depths=[2, 11],
        widths=[384, 768],
        num_heads=[6, 12],
        attn_head_dim=64,
        kernel_size=7,
        mapping_depth=2,
        mapping_dim=768,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.cond_dim = cond_dim
        self.levels = levels
        self.depths = depths
        self.widths = widths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.kernel_size = kernel_size
        self.mapping_depth = mapping_depth
        self.mapping_dim = mapping_dim
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
