import imageio


def permute(lst, pmt):
    assert len(lst) == len(pmt)
    return [lst[i] for i in pmt]


def _read_exr(path):
    # Download freeimage dll, will only download once if not present
    # from https://imageio.readthedocs.io/en/stable/format_exr-fi.html#exr-fi
    imageio.plugins.freeimage.download()

    kwargs = {"flags": imageio.plugins.freeimage.IO_FLAGS.EXR_ZIP}
    return imageio.v2.imread(path, format="exr", **kwargs)
