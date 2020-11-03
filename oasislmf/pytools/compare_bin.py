if __name__ == "__main__":
    from fm import compare
    import sys

    print(sys.argv)
    with open(sys.argv[1], 'rb') as gul, open(sys.argv[2], 'rb') as fm1, open(sys.argv[3], 'rb') as fm2:
        print(compare.compare_streams(gul, fm1, fm2, precision=float(sys.argv[4]) if len(sys.argv) >= 4 else 1e-3))