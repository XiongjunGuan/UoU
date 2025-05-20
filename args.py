'''
Description: 
Author: Xiongjun Guan
Date: 2025-05-14 17:31:15
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-05-19 16:58:24

Copyright (C) 2025 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
"""
Description:
Author: Xiongjun Guan
Date: 2025-05-14
version: 0.0.1
"""

import argparse
from types import SimpleNamespace


def dict_to_namespace(d):
    """
    将嵌套的字典转换为 SimpleNamespace 对象，以支持通过 `.` 的形式访问。
    """
    if isinstance(d, dict):
        # 如果是字典，递归将内部的键值对也转化为 SimpleNamespace
        return SimpleNamespace(**{
            k: dict_to_namespace(v)
            for k, v in d.items()
        })
    elif isinstance(d, list):
        # 如果是列表，递归将列表中的元素也处理（假设列表中元素是字典或其他可转换对象）
        return [dict_to_namespace(v) for v in d]
    else:
        # 如果是普通值（如 int, str, 等），直接返回
        return d


def get_args():
    parser = argparse.ArgumentParser(description="Train parameters")
    # train settings
    parser.add_argument(
        "--config_name",
        type=str,
        default="config",
    )
    parser.add_argument(
        "--cuda_ids",
        "-c",
        type=str,
        default="0,1",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=24,
    )

    return parser.parse_args()
