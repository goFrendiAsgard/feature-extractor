from FE_Base import feature_extracting, shuffle_record, test_phenotype
from gogenpy import utils
    
randomizer = utils.randomizer
records = []
i = 0
while i<3:
    for x in xrange(-10,10):
        for y in xrange(-10,10):
            r = (x**2+y**2) ** 0.5
            if r<6.4:
                c = 'kecil'
            elif r<9.2:
                c = 'sedang'
            else:
                c = 'besar'
            records.append([x,y,c])
    i+=1

variables = ['x','y']
records = shuffle_record(records)

'''
test_phenotype(records, variables, 'sqr(x)+sqr(y)')
test_phenotype(records, variables, 'x*x+sqr(y)')
test_phenotype(records, variables, 'sqrt(sqr(x)+sqr(y))')
test_phenotype(records, variables, 'x')
test_phenotype(records, variables, 'y')
test_phenotype(records, variables, 'sqr(sqr(y + sqr(sqr(sqr(sqr(x - x + y + sqr(y))))) + y + sqr(sqr(sqr(x))) - x))')
test_phenotype(records, variables, 'sqr(sqr(y + sqr(y) + x - x + x * x - y))')
test_phenotype(records, variables, 'sqr(sqr(y) + sqr(x) - x)')
test_phenotype(records, variables, 'sqr(sqr(sqr(y + sqr(cos(y) / cos(x) * sqr(abs(x) - x + y)) - x)))')
'''
feature_extracting(records, variables, label='Hypotenuse-5-Fold', fold=5)
feature_extracting(records, variables, label='Hypotenuse-5-Fold', fold=3)
feature_extracting(records, variables, label='Hypotenuse-1-Fold', fold=1)
