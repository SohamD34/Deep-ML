def cross_product(a, b):
    
    x = a[1]*b[2] - a[2]*b[1]
    y = a[2]*b[0] - a[0]*b[2]
    z = a[0]*b[1] - a[1]*b[0]

    return [x,y,z]

print(cross_product([1, 2, 3], [4, 5, 6]))
# Expected
# [-3, 6, -3]
# Your Output
# [-3, -9, -3]