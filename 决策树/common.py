# -*- ecoding: utf-8 -*-
# @ModuleName: common
# @Function: 
# @Author: Yuxuan Xi
# @Time: 2020/6/7 17:09

import functools
import time

def Timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start=time.time()
        func_ret=func(*args, **kwargs)
        end=time.time()
        return end-start,func_ret
    return wrapper