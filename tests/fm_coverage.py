#!/usr/bin/env python3
# Prints out the names of all unskipped FM test cases in ``tests/acceptance/test_fm_acceptance.py``

import io
import os
import re

if __name__ == '__main__':
    fm_acceptance_tests_fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'acceptance', 'test_fm_acceptance.py')
    with io.open(fm_acceptance_tests_fp, 'r', encoding='utf-8') as f:
        lines = [
            l.strip() for l in f.readlines()
            if l.strip().startswith('@pytest.mark.skip') or l.strip().startswith('def test')
        ]
    #import ipdb; ipdb.set_trace()
    for i, line in enumerate(lines):
        if line.startswith('def') and not lines[i - 1].startswith('@pytest.mark.skip'):
            print(re.match(r'def test_(\w+\d+)\(self\):$', line).groups()[0])


