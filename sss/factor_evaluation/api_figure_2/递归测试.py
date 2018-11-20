def func(l):
    l.append(1)
    if len(l)==10:
        return l
    return func(l)

l=[]
print(func(l),len(l))

a=[0.8,0.85,0.9,0.95]
b=[0.025,0.05,0.075,0.1]
l=[]
for x in a:
    for y in b:
        print(x,y)
        l.append((x,y))

print(l)
