#!/usr/bin/env python3
# Prints out the names of all FM test cases in `validation/examples`
# which are covered by the acceptance tests - all of these must have
# expected data defined

import os

if __name__ == '__main__':
    validation_fp = os.path.join(os.getcwd(), 'examples')
    for c in sorted(
        [
            fn for fn in os.listdir(validation_fp)
            if os.path.isdir(os.path.join(validation_fp, fn)) and 'expected' in os.listdir(os.path.join(validation_fp, fn))
        ]
    ):
        print(c)

