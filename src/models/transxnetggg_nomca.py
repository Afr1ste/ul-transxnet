"""TransXNet-GGG ablation variant with MCA explicitly disabled.

This wrapper reuses the current corrected TransXNet-GGG implementation while
forcing ``use_mca=False`` at construction time. It is intended for clean
without-MCA comparison runs, without editing the with-MCA production module.
"""

from .transxnetggg import TransXNet


def _drop_forced_use_mca(kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("use_mca", None)
    return kwargs


def transxnet_t(pretrained=False, pretrained_cfg=None, **kwargs):
    kwargs = _drop_forced_use_mca(kwargs)
    return TransXNet(
        arch="t",
        use_mca=False,
        use_mudd=True,
        use_differential_attn=True,
        **kwargs,
    )


def transxnet_s(pretrained=False, pretrained_cfg=None, **kwargs):
    kwargs = _drop_forced_use_mca(kwargs)
    return TransXNet(
        arch="s",
        use_mca=False,
        use_mudd=True,
        use_differential_attn=True,
        **kwargs,
    )


def transxnet_b(pretrained=False, pretrained_cfg=None, **kwargs):
    kwargs = _drop_forced_use_mca(kwargs)
    return TransXNet(
        arch="b",
        use_mca=False,
        use_mudd=True,
        use_differential_attn=True,
        **kwargs,
    )
