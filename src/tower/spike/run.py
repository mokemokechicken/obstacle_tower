from pathlib import Path
import sys


if __name__ == '__main__':
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.path.append(str(Path(__file__).absolute().parents[2]))

    from tower.spike.main import main
    main()

"""
# MOVE: [NO, forword, back]
# CAMERA: [NO, rotate left, rotate right]
# Jump: [NO, jump]
# MOVE: [NO, right, left]
"""
