import ctypes
import io
import os
import sys
import tempfile
from contextlib import contextmanager
from functools import partial
from typing import Any, Iterator

WIN = os.name == "nt"
if WIN:
    libc = ctypes.cdll.msvcrt

    kernel32 = ctypes.WinDLL("kernel32")  # type: ignore
    STD_OUTPUT_HANDLE = -11
    hStandardOutput = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    c_stdout = None
else:
    libc = ctypes.CDLL(None)
    c_stdout = ctypes.c_void_p.in_dll(libc, "stdout")
    # sidenote: on mac this would be:
    # c_stdout = ctypes.c_void_p.in_dll(libc, "__stdoutp")


@contextmanager
def stdout_redirected(stream: Any = os.devnull) -> Iterator[None]:
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd: int) -> None:
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        if WIN:
            kernel32.SetStdHandle(STD_OUTPUT_HANDLE, libc._get_osfhandle(to_fd))

        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, "wb"))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode="w+b")
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)


LOG = io.BytesIO()
caplog = partial(stdout_redirected, LOG)