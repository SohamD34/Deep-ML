import numpy as np
import math

def pos_encoding(position: int, d_model: int):

    np.set_printoptions(suppress=True, precision=6)
    
    if position == 0 or d_model <= 0:
        return -1
    
    pos_encoding = []

    for pos in range(position):

        v = []

        for i in range(d_model):
            # for even position
            if(i%2 == 0):
                enc = np.sin(pos / 10000**(2*(i//2)/d_model))
            # for odd position
            else:
                enc = np.cos(pos / 10000**(2*(i//2)/d_model))
            v.append(enc)
        
        pos_encoding.append(v)

    pos_encoding = np.float16(pos_encoding)
    return pos_encoding



position = 2
d_model = 8
print(pos_encoding(position, d_model))

