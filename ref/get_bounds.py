from numpy import nan

f = open('bounds.log', 'r')
S = f.read()
f.close()

min_d = {}
sum_d = {}
max_d = {}
n_d = {}
L1 = S.split('\n')
N = 0
for i, s in enumerate(L1):
    l = s.split(' ')
    k = l[0]
    if k == '':
        continue
    if len(k) > N:
        N = len(k)

    m = float(l[1])
    a = float(l[2])
    M = float(l[3])
    
    if '-nan' in l or 'nan' in l:
        continue
    
    if not min_d.has_key(k) or m < min_d[k]:
        min_d[k] = m
    
    if not max_d.has_key(k) or M > max_d[k]:
        max_d[k] = M
    
    if not sum_d.has_key(k):
        sum_d[k] = a
    else:
        sum_d[k] += a
    
    if not n_d.has_key(k):
        n_d[k] = 1.0
    else:
        n_d[k] += 1.0

for k in max_d.iterkeys():
    fmt = "%" + str(N+1) + "s %10.2g %10.2g %10.2g"
    print fmt % (k, min_d[k], sum_d[k]/n_d[k], max_d[k])
