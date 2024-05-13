import numpy as np


def p2b(
    line: str, start_pose: int = 1, change_line: bool = True
) -> "tuple[list[str],int,int,int,int]":
    """points to bounding box

    Args:
        line (str): original label: head_1 head2 ... (x1,y1) (x2 y2) ...
        start_pose (int, optional): first index of (x,y). Defaults to 1.
        change_line (bool, optional): including \\n or not. Defaults to True.

    Returns:
        tuple[list[str],int,int,int,int]: bounding box label: list(head1 head2 ...) (x,y,w,h)
    """

    inf = line.split(" ")
    arg_num = len(inf)
    if change_line:
        arg_num -= 1
    xs = inf[start_pose:arg_num:2]
    xs = list(map(float, xs))
    xs = sorted(xs)

    ys = inf[start_pose + 1 : arg_num : 2]
    ys = list(map(float, ys))
    ys = sorted(ys)

    x = xs[0]
    y = ys[0]
    w = xs[-1] - xs[0]
    h = ys[-1] - ys[0]

    # 进行修正
    # x -= 0.35 * w
    # w *= 1.7
    # y -= 0.22 * h
    # h *= 1.44

    x -= 0.5 * w
    w *= 2
    y -= 0.8 * h
    h *= 2.4

    heads = list()
    for i in range(start_pose):
        heads.append(inf[i])

    return heads, max(int(x), 0), max(int(y), 0), max(int(w), 0), max(int(h), 0)
