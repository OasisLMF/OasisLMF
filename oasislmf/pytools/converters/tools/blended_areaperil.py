from oasislmf.pytools.common.data import (blended_areaperil_float_dtype, blended_areaperil_float_fmt,
                                          blended_areaperil_float_headers)

# The record layout depends on the weight type in the file's header, so these describe the
# float form only, for listings; the converters in csvtobin/bintocsv utils read the header.
headers = blended_areaperil_float_headers
dtype = blended_areaperil_float_dtype
fmt = blended_areaperil_float_fmt
cli_support = ['bintocsv', 'csvtobin']
