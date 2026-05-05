"""TransXNet-GGG core ablation: MUDD only, with MCA disabled."""

from .transxnetggg import TransXNet


def _clean_kwargs(kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("use_mca", None)
    kwargs.pop("use_mudd", None)
    kwargs.pop("use_differential_attn", None)
    return kwargs


def transxnet_t(pretrained=False, pretrained_cfg=None, **kwargs):
    kwargs = _clean_kwargs(kwargs)
    return TransXNet(
        arch="t",
        use_mca=False,
        use_mudd=True,
        use_differential_attn=False,
        **kwargs,
    )


def transxnet_s(pretrained=False, pretrained_cfg=None, **kwargs):
    kwargs = _clean_kwargs(kwargs)
    return TransXNet(
        arch="s",
        use_mca=False,
        use_mudd=True,
        use_differential_attn=False,
        **kwargs,
    )


def transxnet_b(pretrained=False, pretrained_cfg=None, **kwargs):
    kwargs = _clean_kwargs(kwargs)
    return TransXNet(
        arch="b",
        use_mca=False,
        use_mudd=True,
        use_differential_attn=False,
        **kwargs,
    )
