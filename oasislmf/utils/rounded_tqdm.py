from tqdm import tqdm
import math


# Hopefully remove when option is added to tqdm
class rounded_tqdm(tqdm):
    def __init__(self, *args, **kwargs):
        self.bar_format = '{desc}: {rounded_percent:3d}%|{bar}{r_bar}]'
        kwargs.setdefault('bar_format', self.bar_format)
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        d = super().format_dict
        total = d.get('total')
        n = d.get('n')
        d['rounded_percent'] = math.floor(n / total * 100) if total else 0
        return d
