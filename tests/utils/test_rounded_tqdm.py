import io
from oasislmf.utils.rounded_tqdm import rounded_tqdm


def test_output_does_not_round_up():
    buf = io.StringIO()
    with rounded_tqdm(total=1000, file=buf) as bar:
        bar.update(999)
    assert '100%' not in buf.getvalue()
    assert '99%' in buf.getvalue()


def test_output_does_not_overround():
    buf = io.StringIO()
    with rounded_tqdm(total=1000, file=buf) as bar:
        bar.update(1000)
    assert '100%' in buf.getvalue()
