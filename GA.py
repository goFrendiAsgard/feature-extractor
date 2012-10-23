'''
Created on Oct 23, 2012

@author: gofrendi
'''
from GA_Base import GA_Base

class GA(GA_Base):
    '''
    Genetics Algorithm class
    '''

    def __init__(self):
        '''
        Constructor
        '''
        GA_Base.__init__(self)

if __name__ == '__main__':
    ga = GA()
    ga.process()
    ga.show()