import sys


def resolve_file(path, mode, stack):
    """_summary_

    Args:
        path (str | os.PathLike): File path or "-" indicationg standard input/output.
        mode (str): Mode to open file ("r", "rb, "w", "wb").
        stack (ExitStack): Context manager stack used to manage file lifecycle.

    Returns:
        file (IO): A file-like object opened in the specified mode.
    """
    is_read = "r" in mode
    is_binary = "b" in mode

    if str(path) == "-":
        if is_read:
            return sys.stdin.buffer if is_binary else sys.stdin
        else:
            return sys.stdout.buffer if is_binary else sys.stdout
    else:
        return stack.enter_context(open(path, mode))
