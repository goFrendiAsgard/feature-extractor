from FE_Base import GE_Multi_Fitness
from gogenpy import utils
    
randomizer = utils.randomizer
records = []
training_data = []
training_target = []
i = 0
kecil_count = 0
sedang_count = 0
besar_count = 0
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
            training_data.append([x,y])
            training_target.append(c)
    i+=1


variables = ['x','y']

ge_multi_fitness = GE_Multi_Fitness()
ge_multi_fitness.classes = ['kecil','sedang','besar']
ge_multi_fitness.variables = variables
ge_multi_fitness.training_data = training_data
ge_multi_fitness.training_target = training_target

def print_fitness(phenotype):
    individual = {'phenotype': phenotype}
    print individual['phenotype']
    #print ge_multi_fitness._calculate_projection_attribute(phenotype)
    print ge_multi_fitness.do_calculate_fitness(individual)
    print ''

print_fitness('sqr(x)+sqr(y)')
print_fitness('x*x+sqr(y)')
print_fitness('sqrt(sqr(x)+sqr(y))')
print_fitness('x')
print_fitness('y')
print_fitness('sqr(y)*x')
print_fitness('sqr(sqr(sqrt(sqrt(sqr(sqr(sqrt(sqr(x) + sqr(y))) - y) - sqr(y)))))')
print_fitness('sqrt(sqr(sqr(sqr(sqr(x))) + y + x + y))')