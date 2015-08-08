cdef extern from "crandom.h":
    int add_one(int value)

def generate_normal_deviates(int value):
    print 'generating via crandom...'
    result = add_one(value)
    return result
