import numpy as np

def calculate_brightness(img):
	
    cummulative = 0
    curr_h = len(img)
    if curr_h == 0:
        return -1

    curr_w = len(img[0])
    if curr_w==0:
        return -1

    for i in range(curr_h):
        w = len(img[i])

        if(w == curr_w):
            for j in img[i]:
                if j >=0 and j <= 255:
                    cummulative += j
                else:
                    return -1
        else:
            return -1
    
    return cummulative / (curr_h * curr_w)


# img = [[100, 200], [50, 150, 60]]
# print(calculate_brightness(img))
print(calculate_brightness([]))
print(calculate_brightness([[100, 200], [150]])) 
print(calculate_brightness([[100, 300]])) 
print(calculate_brightness([[128]])) 
print(calculate_brightness([[100, 200], [50, 150]])) 