def disorder(apples: list) -> float:
    """
    Compute the disorder in a basket of apples.
    """
    unique_colors = list(set(apples))
    freq = {i:apples.count(i) for i in unique_colors}
    total = sum(freq.values())
    return 1-sum((count/total) * (count/total) for count in freq.values())

print(disorder([0,0,0,0]))
print(disorder([1,1,0,0]))
print(disorder([0,1,2,3]))
print(disorder([0,0,0,0,0,1,2,3]))
print(disorder([0,0,1,1,2,2,3,3]))